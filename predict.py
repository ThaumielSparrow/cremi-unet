import argparse
from pathlib import Path
from typing import cast

import cv2 as cv
import albumentations as A
import h5py
import numpy as np
import torch
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

from dataload import image_index
from model import MODEL_NAME_UNET, create_model, default_model_config
from preprocess import CREMI


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE_TYPE = DEVICE
USE_AMP = DEVICE_TYPE == 'cuda'
DEFAULT_VOLUME_DEPTH = 125


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_name = checkpoint.get('model_name', MODEL_NAME_UNET)
        model_config = checkpoint.get('model_config', default_model_config(model_name))
        state_dict = checkpoint['state_dict']
        image_height = checkpoint.get('image_height', 200 if model_name == MODEL_NAME_UNET else 512)
        image_width = checkpoint.get('image_width', 200 if model_name == MODEL_NAME_UNET else 512)
        slice_radius = checkpoint.get('slice_radius', 0)
        volume_depth = checkpoint.get('volume_depth', DEFAULT_VOLUME_DEPTH)
    else:
        model_name = MODEL_NAME_UNET
        model_config = default_model_config(model_name)
        state_dict = checkpoint
        image_height = 200
        image_width = 200
        slice_radius = 0
        volume_depth = DEFAULT_VOLUME_DEPTH

    model = create_model(model_name, model_config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    metadata = {
        'model_name': model_name,
        'model_config': model_config,
        'image_height': int(image_height),
        'image_width': int(image_width),
        'slice_radius': int(slice_radius),
        'volume_depth': int(volume_depth) if volume_depth is not None else None,
    }
    return model, metadata


def build_transform(channels, height, width):
    transforms: list[TransformType] = [
        A.Resize(height=height, width=width),
        A.Normalize(
            mean=(0.0,) * channels,
            std=(1.0,) * channels,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def ensure_channels(image, channels):
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], channels, axis=2)
    elif image.shape[-1] == 4:
        image = image[:, :, :3]

    if image.shape[-1] == channels:
        return image

    if channels == 3 and image.shape[-1] > 3:
        return image[:, :, :3]

    grayscale = image[:, :, 0]
    return np.repeat(grayscale[:, :, None], channels, axis=2)


def predict_array(model, transform, image, threshold):
    original_height, original_width = image.shape[:2]
    transformed = transform(image=image)
    tensor = transformed['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.autocast(device_type=DEVICE_TYPE, enabled=USE_AMP):
            logits = model(tensor)
            probs = logits.sigmoid()

    mask = (probs.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
    if mask.shape != (original_height, original_width):
        mask = cv.resize(mask, (original_width, original_height), interpolation=cv.INTER_NEAREST)

    return mask


def numeric_index(path, fallback):
    try:
        return image_index(path)
    except ValueError:
        return fallback


def image_context(input_path):
    image_paths = [path for path in iter_inputs(input_path) if path.suffix.lower() in IMAGE_EXTENSIONS]
    indexed = [(numeric_index(path, idx), path) for idx, path in enumerate(image_paths)]
    context = {}
    path_to_index = {}
    for idx, path in indexed:
        context.setdefault(idx, path)
        path_to_index[path] = idx
    return image_paths, context, path_to_index


def volume_bounds(center_idx, volume_depth):
    if volume_depth is None:
        return None

    volume_start = (center_idx // volume_depth) * volume_depth
    return volume_start, volume_start + volume_depth - 1


def nearest_context_path(context, center_idx, image_idx, volume_depth):
    bounds = volume_bounds(center_idx, volume_depth)
    if bounds is not None:
        image_idx = min(max(image_idx, bounds[0]), bounds[1])

    if image_idx in context:
        return context[image_idx]

    candidate_indices = list(context)
    if bounds is not None:
        candidate_indices = [idx for idx in context if bounds[0] <= idx <= bounds[1]]
        if not candidate_indices:
            raise FileNotFoundError(f'No context images found within volume bounds {bounds}')

    nearest_idx = min(candidate_indices, key=lambda idx: abs(idx - image_idx))
    return context[nearest_idx]


def image_stack_from_context(image_path, channels, slice_radius, volume_depth, context, path_to_index):
    if slice_radius <= 0:
        return ensure_channels(np.array(Image.open(image_path)), channels)

    center_idx = path_to_index[image_path]
    images = []
    for offset in range(-slice_radius, slice_radius + 1):
        context_path = nearest_context_path(context, center_idx, center_idx + offset, volume_depth)
        images.append(np.array(Image.open(context_path).convert('L')))
    return ensure_channels(np.stack(images, axis=-1), channels)


def predict_image_file(model, transform, image_path, output_dir, threshold, channels, slice_radius, volume_depth, context, path_to_index):
    image = image_stack_from_context(image_path, channels, slice_radius, volume_depth, context, path_to_index)
    mask = predict_array(model, transform, image, threshold)
    output_path = output_dir / f'{image_path.stem}_pred.png'
    cv.imwrite(str(output_path), mask)
    return output_path


def hdf_stack(raw, z_index, channels, slice_radius):
    if slice_radius <= 0:
        return ensure_channels(np.asarray(raw[z_index, :, :]), channels)

    images = []
    max_z = raw.shape[0] - 1
    for offset in range(-slice_radius, slice_radius + 1):
        context_z = min(max(z_index + offset, 0), max_z)
        images.append(np.asarray(raw[context_z, :, :]))
    return ensure_channels(np.stack(images, axis=-1), channels)


def predict_hdf_file(model, transform, hdf_path, output_dir, threshold, raw_dataset, channels, slice_radius):
    output_paths = []
    with h5py.File(hdf_path, 'r') as dataset:
        if raw_dataset not in dataset:
            raise KeyError(f'{hdf_path} does not contain {raw_dataset}')

        raw = cast(h5py.Dataset, dataset[raw_dataset])
        for z_index in tqdm(range(raw.shape[0]), desc=f'Predicting {hdf_path.name}'):
            image = hdf_stack(raw, z_index, channels, slice_radius)
            mask = predict_array(model, transform, image, threshold)
            output_path = output_dir / f'{hdf_path.stem}_z{z_index:04d}_pred.png'
            cv.imwrite(str(output_path), mask)
            output_paths.append(output_path)

    return output_paths


def iter_inputs(input_path):
    if input_path.is_file():
        yield input_path
        return

    for path in sorted(input_path.rglob('*')):
        if path.is_file() and (path.suffix.lower() in IMAGE_EXTENSIONS or path.suffix.lower() in {'.h5', '.hdf', '.hdf5'}):
            yield path


def main():
    parser = argparse.ArgumentParser(description='Run EM membrane segmentation inference from a trained U-Net checkpoint.')
    parser.add_argument('input', type=Path, help='Image file, image folder, HDF file, or folder containing HDF/images.')
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoint.pth.tar'), help='Path to trained checkpoint.')
    parser.add_argument('--output-dir', type=Path, default=Path('predictions'), help='Directory for predicted mask PNGs.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Sigmoid threshold for binary masks.')
    parser.add_argument('--raw-dataset', default=CREMI.RAW_DATASET, help='HDF dataset path containing raw EM volume.')
    parser.add_argument('--volume-depth', type=int, default=None, help='Number of slices per original PNG volume for 2.5D folder inference.')
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    if not args.input.exists():
        raise FileNotFoundError(f'Input path not found: {args.input}')

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, metadata = load_model(args.checkpoint, DEVICE)
    channels = int(metadata['model_config']['in_chan'])
    slice_radius = metadata['slice_radius']
    volume_depth = args.volume_depth if args.volume_depth is not None else metadata['volume_depth']
    transform = build_transform(channels, metadata['image_height'], metadata['image_width'])
    image_paths, context, path_to_index = image_context(args.input)

    written = []
    for input_path in iter_inputs(args.input):
        suffix = input_path.suffix.lower()
        if suffix in {'.h5', '.hdf', '.hdf5'}:
            written.extend(
                predict_hdf_file(
                    model,
                    transform,
                    input_path,
                    args.output_dir,
                    args.threshold,
                    args.raw_dataset,
                    channels,
                    slice_radius,
                )
            )
        elif suffix in IMAGE_EXTENSIONS:
            written.append(
                predict_image_file(
                    model,
                    transform,
                    input_path,
                    args.output_dir,
                    args.threshold,
                    channels,
                    slice_radius,
                    volume_depth,
                    context,
                    path_to_index,
                )
            )

    if not written:
        raise FileNotFoundError(f'No supported image or HDF files found in {args.input}')

    print(f'Saved {len(written)} prediction mask(s) to {args.output_dir}')
    print(f'Device: {DEVICE}, checkpoint: {args.checkpoint}')
    print(f"Model: {metadata['model_name']}, input size: {metadata['image_height']}x{metadata['image_width']}")
    print(f'Slice radius: {slice_radius}, volume depth: {volume_depth}, channels: {channels}, threshold: {args.threshold}')


if __name__ == '__main__':
    main()
