from pathlib import Path

import torch
import torchvision
from dataload import LoadData
from torch.utils.data import DataLoader


def final_logits(outputs):
    return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    print('=> Current state saved')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print('=> Loaded state')
    model.load_state_dict(checkpoint['state_dict'])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    slice_radius=0,
    train_context_dirs=None,
    val_context_dirs=None,
    volume_depth=None,
):
    if slice_radius > 0:
        default_context_dirs = [train_dir, val_dir]
        train_context_dirs = train_context_dirs or default_context_dirs
        val_context_dirs = val_context_dirs or default_context_dirs

    train_ds = LoadData(
        em_dir=train_dir,
        seg_dir=train_maskdir,
        transform=train_transform,
        slice_radius=slice_radius,
        context_dirs=train_context_dirs,
        volume_depth=volume_depth,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = LoadData(
        em_dir=val_dir,
        seg_dir=val_maskdir,
        transform=val_transform,
        slice_radius=slice_radius,
        context_dirs=val_context_dirs,
        volume_depth=volume_depth,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device='cuda'):
    num_correct=0
    num_pixels=0
    dice_score=0
    
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = final_logits(model(x)).sigmoid()
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += preds.numel()
            dice_score += (2 * (preds*y).sum()) / (preds+y).sum() + 1e-8

    print(f'{num_correct}/{num_pixels} correct\nAccuracy: {((num_correct/num_pixels)*100):.2f}%')
    print(f'Dice score: {dice_score/len(loader):.3f}')

    model.train()

def save_predictions_as_imgs(loader, model, dir='saved_images/', device='cuda'):
    Path(dir).mkdir(parents=True, exist_ok=True)
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = final_logits(model(x)).sigmoid()
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f'{dir}/pred_{idx}.png')
        torchvision.utils.save_image(y.unsqueeze(1), f'{dir}/{idx}.png')
    
    model.train()
