from pathlib import Path

import albumentations as A
from albumentations.core.composition import TransformType
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from model import MODEL_NAME_MEMBRANE_2P5D, create_model, default_model_config
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)

# Hyperparameters
MODEL_NAME = MODEL_NAME_MEMBRANE_2P5D
MODEL_CONFIG = default_model_config(MODEL_NAME)
SLICE_RADIUS = 2
VOLUME_DEPTH = 125
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE_TYPE = DEVICE
USE_AMP = DEVICE_TYPE == 'cuda'
BATCH_SIZE = 2
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = DEVICE_TYPE == 'cuda'
LOAD_MODEL = False
DEEP_SUPERVISION_WEIGHTS = (1.0, 0.5, 0.25, 0.125)

TRAIN_IMAGE_DIR = 'data/train/EM/'
TRAIN_MASK_DIR = 'data/train/SEG/'
TEST_IMAGE_DIR = 'data/test/EM/'
TEST_MASK_DIR = 'data/test/SEG/'
PREDICTION_DIR = Path('saved_images')


def model_channels():
    return int(MODEL_CONFIG['in_chan'])


def deep_supervision_loss(predictions, targets, loss_fn):
    if not isinstance(predictions, (list, tuple)):
        return loss_fn(predictions, targets)

    weights = DEEP_SUPERVISION_WEIGHTS[:len(predictions)]
    total_weight = sum(weights)
    first_weight = weights[0]
    first_pred = predictions[0]
    loss = first_weight * loss_fn(first_pred, targets)
    for weight, pred in zip(weights[1:], predictions[1:]):
        loss = loss + weight * loss_fn(pred, targets)
    return loss / total_weight


def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    
    for _, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward
        with torch.autocast(device_type=DEVICE_TYPE, enabled=USE_AMP):
            predictions = model(data)
            loss = deep_supervision_loss(predictions, targets, loss_fn)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def get_train_transform():
    channels = model_channels()
    transforms: list[TransformType] = [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=(0.0,) * channels,
            std=(1.0,) * channels,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def get_val_transform():
    channels = model_channels()
    transforms: list[TransformType] = [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=(0.0,) * channels,
            std=(1.0,) * channels,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def main():
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    model = create_model(MODEL_NAME, MODEL_CONFIG).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # Use Cross-Entropy loss and add Sigmoid operation if output desires multiple channels
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMAGE_DIR,
        TRAIN_MASK_DIR,
        TEST_IMAGE_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
        slice_radius=SLICE_RADIUS,
        volume_depth=VOLUME_DEPTH,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load('checkpoint.pth.tar', map_location=DEVICE), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.GradScaler(device=DEVICE_TYPE, enabled=USE_AMP)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)
        
        checkpoint = {
            'model_name': MODEL_NAME,
            'model_config': MODEL_CONFIG,
            'slice_radius': SLICE_RADIUS,
            'volume_depth': VOLUME_DEPTH,
            'image_height': IMAGE_HEIGHT,
            'image_width': IMAGE_WIDTH,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)

        save_predictions_as_imgs(val_loader, model, dir=str(PREDICTION_DIR), device=DEVICE)


if __name__ == "__main__":
    main()
