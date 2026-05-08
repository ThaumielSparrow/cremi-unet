from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from config import DEFAULT_BLOCK_SHAPE, DEFAULT_HALO, DEFAULT_OFFSETS, RAW_DATASET_KEY
from data import iter_hdf_paths, normalize_raw
from model import create_model


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(checkpoint["model_name"], checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    offsets = tuple(tuple(offset) for offset in checkpoint.get("offsets", DEFAULT_OFFSETS))
    normalization = checkpoint.get("normalization", "scale01")
    target_mode = checkpoint.get("target_mode", "affinity")
    return model, offsets, normalization, target_mode, checkpoint


def block_positions(length: int, block_size: int) -> list[tuple[int, int]]:
    starts = list(range(0, length, block_size))
    return [(start, min(start + block_size, length)) for start in starts]


def parse_roi(values: list[int] | None) -> tuple[int, int, int, int, int, int] | None:
    if values is None:
        return None
    if len(values) != 6:
        raise argparse.ArgumentTypeError("ROI must be z0 z1 y0 y1 x0 x1")
    z0, z1, y0, y1, x0, x1 = [int(value) for value in values]
    if z0 < 0 or y0 < 0 or x0 < 0 or z1 <= z0 or y1 <= y0 or x1 <= x0:
        raise ValueError(f"Invalid ROI: {values}")
    return z0, z1, y0, y1, x0, x1


def roi_shape(roi: tuple[int, int, int, int, int, int] | None, source_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if roi is None:
        return source_shape
    z0, z1, y0, y1, x0, x1 = roi
    if z1 > source_shape[0] or y1 > source_shape[1] or x1 > source_shape[2]:
        raise ValueError(f"ROI {roi} exceeds source shape {source_shape}")
    return z1 - z0, y1 - y0, x1 - x0


def expanded_slices(
    output_start: tuple[int, int, int],
    output_stop: tuple[int, int, int],
    source_shape: tuple[int, int, int],
    roi_start: tuple[int, int, int],
    halo: tuple[int, int, int],
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    input_slices = []
    crop_slices = []
    for start, stop, roi_axis_start, size, pad in zip(output_start, output_stop, roi_start, source_shape, halo):
        source_output_start = roi_axis_start + start
        source_output_stop = roi_axis_start + stop
        input_start = max(source_output_start - pad, 0)
        input_stop = min(source_output_stop + pad, size)
        crop_start = source_output_start - input_start
        crop_stop = crop_start + (stop - start)
        input_slices.append(slice(input_start, input_stop))
        crop_slices.append(slice(crop_start, crop_stop))
    return tuple(input_slices), tuple(crop_slices)


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    stem = input_path.stem
    return output_dir / f"{stem}_affinities.h5"


def predict_file(
    input_path: Path,
    output_path: Path,
    model,
    offsets,
    raw_key: str,
    output_key: str,
    block_shape: tuple[int, int, int],
    halo: tuple[int, int, int],
    device: torch.device,
    use_amp: bool,
    normalization: str,
    target_mode: str,
    roi: tuple[int, int, int, int, int, int] | None,
) -> None:
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        if raw_key not in src:
            raise KeyError(f"{input_path} does not contain {raw_key}")

        raw_dataset = src[raw_key]
        source_shape = tuple(raw_dataset.shape[:3])
        volume_shape = roi_shape(roi, source_shape)
        roi_start = (0, 0, 0) if roi is None else (roi[0], roi[2], roi[4])
        out = dst.create_dataset(
            output_key,
            shape=(len(offsets), *volume_shape),
            dtype=np.uint8,
            chunks=(1, min(block_shape[0], volume_shape[0]), min(block_shape[1], volume_shape[1]), min(block_shape[2], volume_shape[2])),
            compression="gzip",
            compression_opts=4,
        )
        out.attrs["offsets"] = np.asarray(offsets, dtype=np.int32)
        out.attrs["source_file"] = str(input_path)
        out.attrs["raw_key"] = raw_key
        out.attrs["target_mode"] = target_mode
        if roi is not None:
            out.attrs["roi"] = np.asarray(roi, dtype=np.int64)

        z_blocks = block_positions(volume_shape[0], block_shape[0])
        y_blocks = block_positions(volume_shape[1], block_shape[1])
        x_blocks = block_positions(volume_shape[2], block_shape[2])
        total_blocks = len(z_blocks) * len(y_blocks) * len(x_blocks)

        with torch.no_grad(), tqdm(total=total_blocks, desc=f"Predicting {input_path.name}") as progress:
            for z0, z1 in z_blocks:
                for y0, y1 in y_blocks:
                    for x0, x1 in x_blocks:
                        output_start = (z0, y0, x0)
                        output_stop = (z1, y1, x1)
                        input_slices, crop_slices = expanded_slices(output_start, output_stop, source_shape, roi_start, halo)
                        raw = normalize_raw(np.asarray(raw_dataset[input_slices]), normalization)
                        tensor = torch.from_numpy(np.ascontiguousarray(raw[None, None], dtype=np.float32)).to(device)

                        with torch.autocast(device_type=device_type, enabled=use_amp):
                            probs = model(tensor).sigmoid()
                        probs = probs[0, :, crop_slices[0], crop_slices[1], crop_slices[2]]
                        probs_uint8 = (probs.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
                        out[:, z0:z1, y0:y1, x0:x1] = probs_uint8
                        progress.update(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict CREMI affinity maps from raw HDF volumes.")
    parser.add_argument("input", type=Path, help="HDF file or folder containing HDF files.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/affinity/best.pth.tar"))
    parser.add_argument("--output-dir", type=Path, default=Path("predictions/affinities"))
    parser.add_argument("--raw-key", default=RAW_DATASET_KEY)
    parser.add_argument("--output-key", default="affinities")
    parser.add_argument("--block-shape", nargs=3, type=int, default=DEFAULT_BLOCK_SHAPE, metavar=("Z", "Y", "X"))
    parser.add_argument("--halo", nargs=3, type=int, default=DEFAULT_HALO, metavar=("Z", "Y", "X"))
    parser.add_argument("--roi", nargs=6, type=int, default=None, metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.input.exists():
        raise FileNotFoundError(f"Input path not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    model, offsets, normalization, target_mode, _ = load_checkpoint(args.checkpoint, device)
    roi = parse_roi(args.roi)

    for input_path in iter_hdf_paths(args.input):
        output_path = output_path_for(input_path, args.output_dir)
        predict_file(
            input_path=input_path,
            output_path=output_path,
            model=model,
            offsets=offsets,
            raw_key=args.raw_key,
            output_key=args.output_key,
            block_shape=tuple(args.block_shape),
            halo=tuple(args.halo),
            device=device,
            use_amp=use_amp,
            normalization=normalization,
            target_mode=target_mode,
            roi=roi,
        )


if __name__ == "__main__":
    main()
