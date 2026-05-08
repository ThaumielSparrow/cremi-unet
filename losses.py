from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


AVAILABLE_AFFINITY_LOSSES = ("dice", "bce_dice", "bce", "soft_dice")


class DiceLoss(nn.Module):
    """Masked Dice loss following torch-em's squared-denominator Dice formulation."""

    def __init__(self, eps: float = 1e-7, reduce_channel: str = "mean"):
        super().__init__()
        if reduce_channel not in {"mean", "sum"}:
            raise ValueError('reduce_channel must be "mean" or "sum"')
        self.eps = eps
        self.reduce_channel = reduce_channel

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = logits.sigmoid() * mask
        targets = targets * mask
        dims = (0, 2, 3, 4)
        numerator = (probs * targets).sum(dim=dims)
        denominator = (probs * probs).sum(dim=dims) + (targets * targets).sum(dim=dims)
        valid_channels = mask.sum(dim=dims) > 0
        if not valid_channels.any():
            return logits.sum() * 0.0

        dice_error = 1.0 - 2.0 * numerator[valid_channels] / denominator[valid_channels].clamp_min(self.eps)
        if self.reduce_channel == "sum":
            return dice_error.sum()
        return dice_error.mean()


class SoftDiceLoss(nn.Module):
    """Classic soft Dice with linear denominator, kept for experimentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        probs = logits.sigmoid() * mask
        targets = targets * mask
        dims = (2, 3, 4)
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        valid_channels = mask.sum(dim=dims) > 0
        if not valid_channels.any():
            return logits.sum() * 0.0
        return 1.0 - dice[valid_channels].mean()


class BCELoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * mask
        denominator = mask.sum().clamp_min(1.0)
        return loss.sum() / denominator


class WeightedLoss(nn.Module):
    def __init__(self, losses: list[nn.Module], weights: tuple[float, ...]):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        total = logits.new_tensor(0.0)
        total_weight = sum(self.weights)
        for weight, loss_fn in zip(self.weights, self.losses):
            total = total + weight * loss_fn(logits, targets, mask)
        return total / total_weight


def create_affinity_loss(name: str) -> nn.Module:
    if name == "dice":
        return DiceLoss()
    if name == "soft_dice":
        return SoftDiceLoss()
    if name == "bce":
        return BCELoss()
    if name == "bce_dice":
        return WeightedLoss(
            [BCELoss(), DiceLoss()],
            weights=(0.5, 0.5),
        )
    raise ValueError(f'Unknown affinity loss "{name}". Available losses: {", ".join(AVAILABLE_AFFINITY_LOSSES)}')
