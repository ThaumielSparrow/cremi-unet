from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEFAULT_NORMALIZATION,
    DEFAULT_OFFSETS,
    DEFAULT_PATCH_SHAPE,
    DEFAULT_SCALE_FACTORS,
    DEFAULT_TARGET_MODE,
    LABEL_DATASET_KEY,
    MODEL_NAME_AFFINITY,
    RAW_DATASET_KEY,
)
from data import CremiAffinityDataset
from losses import create_affinity_loss
from model import create_model


def parse_shape(values: list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected exactly three integers: z y x")
    return int(values[0]), int(values[1]), int(values[2])


def parse_scale_factors(value: str | None) -> tuple[tuple[int, int, int], ...]:
    if value is None:
        return DEFAULT_SCALE_FACTORS
    parsed = json.loads(value)
    return tuple(tuple(int(axis) for axis in scale) for scale in parsed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def move_batch(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device):
    raw, targets, mask = batch
    return raw.to(device, non_blocking=True), targets.to(device, non_blocking=True), mask.to(device, non_blocking=True)


def validate(model, loader, loss_fn, device: torch.device, use_amp: bool) -> float:
    device_type = "cuda" if device.type == "cuda" else "cpu"
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            raw, targets, mask = move_batch(batch, device)
            with torch.autocast(device_type=device_type, enabled=use_amp):
                logits = model(raw)
                loss = loss_fn(logits, targets, mask)
            total += float(loss.item())
            count += 1
    model.train()
    return total / max(count, 1)


def save_checkpoint(path: Path, model, optimizer, scaler, args, step: int, best_val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": MODEL_NAME_AFFINITY,
        "model_config": model.config,
        "offsets": DEFAULT_OFFSETS,
        "raw_key": args.raw_key,
        "label_key": args.label_key,
        "patch_shape": tuple(args.patch_shape),
        "normalization": args.normalize,
        "target_mode": args.target_mode,
        "loss_name": args.loss,
        "step": step,
        "best_val_loss": best_val_loss,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    torch.save(checkpoint, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a 3D anisotropic CREMI affinity U-Net.")
    parser.add_argument("--train-dir", type=Path, default=Path("samples/train"), help="Folder or HDF file with labeled CREMI data.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/affinity"))
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a previous checkpoint.")
    parser.add_argument("--raw-key", default=RAW_DATASET_KEY)
    parser.add_argument("--label-key", default=LABEL_DATASET_KEY)
    parser.add_argument("--patch-shape", nargs=3, type=int, default=DEFAULT_PATCH_SHAPE, metavar=("Z", "Y", "X"))
    parser.add_argument("--scale-factors", default=None, help='JSON scale factors, e.g. "[[1,3,3],[1,3,3],[2,2,2],[2,2,2],[2,2,2]]".')
    parser.add_argument("--initial-features", type=int, default=32)
    parser.add_argument("--gain", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=1024, help="Use 0 for no feature cap.")
    parser.add_argument("--block-type", choices=("torch_em", "residual"), default="torch_em")
    parser.add_argument("--norm", choices=("InstanceNorm", "GroupNorm", "BatchNorm", "none"), default="InstanceNorm")
    parser.add_argument("--activation", choices=("ReLU", "SiLU", "GELU"), default="ReLU")
    parser.add_argument("--anisotropic-kernel", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--train-patches", type=int, default=1024, help="Random patches per loader pass.")
    parser.add_argument("--val-patches", type=int, default=32)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--loss", choices=("dice", "bce_dice", "bce", "soft_dice"), default="dice")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when supported.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--normalize", choices=("scale01", "standardize"), default=DEFAULT_NORMALIZATION)
    parser.add_argument("--target-mode", choices=("affinity", "disaffinity"), default=DEFAULT_TARGET_MODE)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--p-drop-slice", type=float, default=0.025)
    parser.add_argument("--p-low-contrast", type=float, default=0.025)
    parser.add_argument("--p-deform-slice", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.patch_shape = parse_shape(args.patch_shape)
    scale_factors = parse_scale_factors(args.scale_factors)

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = device.type == "cuda" and not args.no_amp
    device_type = "cuda" if device.type == "cuda" else "cpu"

    train_ds = CremiAffinityDataset(
        args.train_dir,
        patch_shape=args.patch_shape,
        raw_key=args.raw_key,
        label_key=args.label_key,
        offsets=DEFAULT_OFFSETS,
        patches_per_epoch=args.train_patches,
        augment=not args.no_augment,
        deterministic=False,
        seed=args.seed,
        normalize=args.normalize,
        target_mode=args.target_mode,
        p_drop_slice=args.p_drop_slice,
        p_low_contrast=args.p_low_contrast,
        p_deform_slice=args.p_deform_slice,
    )
    val_ds = CremiAffinityDataset(
        args.train_dir,
        patch_shape=args.patch_shape,
        raw_key=args.raw_key,
        label_key=args.label_key,
        offsets=DEFAULT_OFFSETS,
        patches_per_epoch=args.val_patches,
        augment=False,
        deterministic=True,
        seed=args.seed + 10_000,
        normalize=args.normalize,
        target_mode=args.target_mode,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model_config = {
        "in_channels": 1,
        "out_channels": len(DEFAULT_OFFSETS),
        "initial_features": args.initial_features,
        "gain": args.gain,
        "max_features": args.max_features,
        "scale_factors": scale_factors,
        "block_type": args.block_type,
        "norm": None if args.norm == "none" else args.norm,
        "activation": args.activation,
        "anisotropic_kernel": args.anisotropic_kernel,
    }
    model = create_model(MODEL_NAME_AFFINITY, model_config).to(device)
    if args.compile:
        model = torch.compile(model)

    loss_fn = create_affinity_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device=device_type, enabled=use_amp)

    start_step = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        state_model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint.get("step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

    model.train()
    train_iter = cycle(train_loader)
    progress = tqdm(range(start_step + 1, args.steps + 1), desc=f"Training on {device}")
    running_loss = 0.0

    for step in progress:
        raw, targets, mask = move_batch(next(train_iter), device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type, enabled=use_amp):
            logits = model(raw)
            loss = loss_fn(logits, targets, mask)

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        progress.set_postfix(loss=running_loss / max(1, step - start_step))

        if step % args.save_every == 0:
            state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(args.checkpoint_dir / "last.pth.tar", state_model, optimizer, scaler, args, step, best_val_loss)

        if step % args.val_every == 0:
            val_loss = validate(model, val_loader, loss_fn, device, use_amp)
            progress.write(f"step={step} val_loss={val_loss:.5f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                save_checkpoint(args.checkpoint_dir / "best.pth.tar", state_model, optimizer, scaler, args, step, best_val_loss)

    state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_checkpoint(args.checkpoint_dir / "last.pth.tar", state_model, optimizer, scaler, args, args.steps, best_val_loss)


if __name__ == "__main__":
    main()
