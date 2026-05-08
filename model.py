from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEFAULT_SCALE_FACTORS, MODEL_NAME_AFFINITY


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return min(groups, channels)
    return 1


def norm_layer(name: str | None, channels: int) -> nn.Module:
    if name is None or name.lower() == "none":
        return nn.Identity()
    if name == "InstanceNorm":
        return nn.InstanceNorm3d(channels, affine=True)
    if name == "GroupNorm":
        return nn.GroupNorm(group_count(channels), channels)
    if name == "BatchNorm":
        return nn.BatchNorm3d(channels)
    raise ValueError(f"Unknown norm: {name}")


def act_layer(name: str) -> nn.Module:
    if name == "ReLU":
        return nn.ReLU(inplace=True)
    if name == "SiLU":
        return nn.SiLU(inplace=True)
    if name == "GELU":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


def scale_kernel(
    scale_factor: tuple[int, int, int],
    anisotropic_kernel: bool,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if not anisotropic_kernel:
        return (3, 3, 3), (1, 1, 1)
    kernel = tuple(1 if factor == 1 else 3 for factor in scale_factor)
    padding = tuple(0 if factor == 1 else 1 for factor in scale_factor)
    return kernel, padding


def capped_features(initial_features: int, gain: int, depth: int, max_features: int | None) -> tuple[int, ...]:
    features = tuple(initial_features * (gain**idx) for idx in range(depth + 1))
    if max_features is None or max_features <= 0:
        return features
    return tuple(min(feature, max_features) for feature in features)


class ConvBlock3d(nn.Module):
    """torch-em style pre-activation two-convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        padding: tuple[int, int, int] = (1, 1, 1),
        norm: str | None = "InstanceNorm",
        activation: str = "ReLU",
    ):
        super().__init__()
        self.block = nn.Sequential(
            norm_layer(norm, in_channels),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            act_layer(activation),
            norm_layer(norm, out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            act_layer(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        padding: tuple[int, int, int] = (1, 1, 1),
        norm: str | None = "GroupNorm",
        activation: str = "SiLU",
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            norm_layer(norm, out_channels),
            act_layer(activation),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            norm_layer(norm, out_channels),
        )
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(norm, out_channels),
            )
        self.activation = act_layer(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv2(self.conv1(x)) + self.shortcut(x))


class UpBlock3d(nn.Module):
    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        block_type: str,
        kernel_size: tuple[int, int, int],
        padding: tuple[int, int, int],
        norm: str | None,
        activation: str,
    ):
        super().__init__()
        self.project = nn.Conv3d(decoder_channels, skip_channels, kernel_size=1)
        block_impl = ConvBlock3d if block_type == "torch_em" else ResBlock3d
        self.block = block_impl(
            skip_channels * 2,
            skip_channels,
            kernel_size=kernel_size,
            padding=padding,
            norm=norm,
            activation=activation,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.project(x)
        return self.block(torch.cat((skip, x), dim=1))


class AnisotropicAffinityUNet(nn.Module):
    """torch-em inspired 3D anisotropic U-Net for CREMI affinity/disaffinity logits."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 12,
        initial_features: int = 32,
        gain: int = 2,
        max_features: int | None = 1024,
        scale_factors: tuple[tuple[int, int, int], ...] = DEFAULT_SCALE_FACTORS,
        block_type: str = "torch_em",
        norm: str | None = "InstanceNorm",
        activation: str = "ReLU",
        anisotropic_kernel: bool = False,
    ):
        super().__init__()
        if block_type not in {"torch_em", "residual"}:
            raise ValueError('block_type must be "torch_em" or "residual"')

        scale_factors = tuple(tuple(scale) for scale in scale_factors)
        depth = len(scale_factors)
        features = capped_features(initial_features, gain, depth, max_features)
        block_impl = ConvBlock3d if block_type == "torch_em" else ResBlock3d

        self.config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "initial_features": initial_features,
            "gain": gain,
            "max_features": max_features,
            "scale_factors": scale_factors,
            "block_type": block_type,
            "norm": norm,
            "activation": activation,
            "anisotropic_kernel": anisotropic_kernel,
        }

        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for feature, scale_factor in zip(features[:-1], scale_factors):
            kernel_size, padding = scale_kernel(scale_factor, anisotropic_kernel)
            self.encoders.append(
                block_impl(
                    current_channels,
                    feature,
                    kernel_size=kernel_size,
                    padding=padding,
                    norm=norm,
                    activation=activation,
                )
            )
            current_channels = feature

        self.pools = nn.ModuleList(
            [nn.MaxPool3d(kernel_size=scale, stride=scale, ceil_mode=False) for scale in scale_factors]
        )

        base_kernel, base_padding = scale_kernel(scale_factors[-1], anisotropic_kernel)
        self.base = block_impl(
            features[-2],
            features[-1],
            kernel_size=base_kernel,
            padding=base_padding,
            norm=norm,
            activation=activation,
        )

        self.decoders = nn.ModuleList()
        current_channels = features[-1]
        for skip_channels, scale_factor in zip(reversed(features[:-1]), reversed(scale_factors)):
            kernel_size, padding = scale_kernel(scale_factor, anisotropic_kernel)
            self.decoders.append(
                UpBlock3d(
                    current_channels,
                    skip_channels,
                    block_type=block_type,
                    kernel_size=kernel_size,
                    padding=padding,
                    norm=norm,
                    activation=activation,
                )
            )
            current_channels = skip_channels

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        x = self.base(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.final_conv(x)


def default_model_config(model_name: str = MODEL_NAME_AFFINITY) -> dict:
    if model_name != MODEL_NAME_AFFINITY:
        raise ValueError(f"Unknown model name: {model_name}")
    return {
        "in_channels": 1,
        "out_channels": 12,
        "initial_features": 32,
        "gain": 2,
        "max_features": 1024,
        "scale_factors": DEFAULT_SCALE_FACTORS,
        "block_type": "torch_em",
        "norm": "InstanceNorm",
        "activation": "ReLU",
        "anisotropic_kernel": False,
    }


def create_model(model_name: str = MODEL_NAME_AFFINITY, model_config: dict | None = None) -> nn.Module:
    if model_name != MODEL_NAME_AFFINITY:
        raise ValueError(f"Unknown model name: {model_name}")

    config = default_model_config(model_name)
    if model_config:
        config.update(model_config)
    config["scale_factors"] = tuple(tuple(scale) for scale in config["scale_factors"])
    return AnisotropicAffinityUNet(**config)


if __name__ == "__main__":
    model = create_model(model_config={"initial_features": 4, "max_features": 64})
    x = torch.randn(1, 1, 8, 72, 72)
    y = model(x)
    assert y.shape == (1, 12, 8, 72, 72)
    print(y.shape)
