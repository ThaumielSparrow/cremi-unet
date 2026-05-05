# pyright: reportPrivateImportUsage=false
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


MODEL_NAME_UNET = 'unet'
MODEL_NAME_MEMBRANE_2P5D = 'membrane_unet_2p5d'


class DoubleConv(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan, out_chan, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet architecture, implemented to follow schema in relevant paper: https://arxiv.org/abs/1505.04597
    
    Batch normalization added between convolve layers for reduction in covariance
    """
    def __init__(self, in_chan=3, out_chan=1, features:tuple=(64, 128, 256, 512)):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # UNet encoder blocks
        for feature in features:
            self.downs.append(DoubleConv(in_chan, feature))
            in_chan = feature

        # UNet decoder blocks
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # Misc UNet utils
        self.plateau = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_chan, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.plateau(x)
        skip_connections.reverse()

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # Barrier against error when inputs are not factors of 16 
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)


def group_count(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_chan,
                out_chan,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(group_count(out_chan), out_chan),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv1 = ConvNormAct(in_chan, out_chan)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_chan), out_chan),
        )
        if in_chan == out_chan:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False),
                nn.GroupNorm(group_count(out_chan), out_chan),
            )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.conv2(self.conv1(x)) + self.shortcut(x))


class ASPP(nn.Module):
    def __init__(self, in_chan, out_chan, dilation_rates=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvNormAct(
                    in_chan,
                    out_chan,
                    kernel_size=3,
                    padding=rate,
                    dilation=rate,
                )
                for rate in dilation_rates
            ]
        )
        self.project = ConvNormAct(
            out_chan * len(dilation_rates),
            out_chan,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x):
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class AttentionGate(nn.Module):
    def __init__(self, skip_chan, gate_chan):
        super().__init__()
        inter_chan = max(skip_chan // 2, 1)
        self.skip_proj = nn.Conv2d(skip_chan, inter_chan, kernel_size=1, bias=True)
        self.gate_proj = nn.Conv2d(gate_chan, inter_chan, kernel_size=1, bias=True)
        self.attention = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_chan, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, skip, gate):
        if skip.shape[2:] != gate.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode='bilinear', align_corners=False)
        weights = self.attention(self.skip_proj(skip) + self.gate_proj(gate))
        return skip * weights


class UpBlock(nn.Module):
    def __init__(self, decoder_chan, skip_chan, out_chan):
        super().__init__()
        self.attention = AttentionGate(skip_chan, decoder_chan)
        self.block = ResidualBlock(decoder_chan + skip_chan, out_chan)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        skip = self.attention(skip, x)
        return self.block(torch.cat((skip, x), dim=1))


class MembraneUNet2p5D(nn.Module):
    """
    2.5D U-Net variant for membrane segmentation.

    Input channels are adjacent z-slices around the target slice. During training
    the model can return deep-supervision logits. Eval mode returns only the final logits.
    """
    def __init__(
        self,
        in_chan=5,
        out_chan=1,
        features=(32, 64, 128, 256, 512),
        dilation_rates=(1, 2, 4, 8),
        deep_supervision=True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.config = {
            'in_chan': in_chan,
            'out_chan': out_chan,
            'features': tuple(features),
            'dilation_rates': tuple(dilation_rates),
            'deep_supervision': deep_supervision,
        }

        self.encoders = nn.ModuleList()
        current_chan = in_chan
        for feature in features:
            self.encoders.append(ResidualBlock(current_chan, feature))
            current_chan = feature

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ASPP(features[-1], features[-1], dilation_rates=dilation_rates)

        decoder_channels = list(reversed(features[:-1]))
        gate_channels = list(reversed(features[1:]))
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(decoder_chan, skip_chan, skip_chan)
                for decoder_chan, skip_chan in zip(gate_channels, decoder_channels)
            ]
        )
        self.final_conv = nn.Conv2d(features[0], out_chan, kernel_size=1)
        self.deep_supervision_heads = nn.ModuleList(
            [nn.Conv2d(channel, out_chan, kernel_size=1) for channel in decoder_channels[:-1]]
        )

    def forward(self, x):
        skips = []
        for idx, encoder in enumerate(self.encoders):
            x = encoder(x)
            if idx < len(self.encoders) - 1:
                skips.append(x)
                x = self.pool(x)

        x = self.bottleneck(x)
        decoder_outputs = []
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip)
            decoder_outputs.append(x)

        final_logits = self.final_conv(x)
        if not self.training or not self.deep_supervision:
            return final_logits

        logits = [final_logits]
        target_size = final_logits.shape[2:]
        for feature, head in zip(decoder_outputs[:-1], self.deep_supervision_heads):
            aux_logits = head(feature)
            aux_logits = F.interpolate(aux_logits, size=target_size, mode='bilinear', align_corners=False)
            logits.append(aux_logits)

        return logits


def default_model_config(model_name=MODEL_NAME_MEMBRANE_2P5D):
    if model_name == MODEL_NAME_MEMBRANE_2P5D:
        return {
            'in_chan': 5,
            'out_chan': 1,
            'features': (32, 64, 128, 256, 512),
            'dilation_rates': (1, 2, 4, 8),
            'deep_supervision': True,
        }
    if model_name == MODEL_NAME_UNET:
        return {
            'in_chan': 3,
            'out_chan': 1,
            'features': (64, 128, 256, 512),
        }
    raise ValueError(f'Unknown model name: {model_name}')


def create_model(model_name=MODEL_NAME_MEMBRANE_2P5D, model_config=None):
    config = default_model_config(model_name)
    if model_config:
        config.update(model_config)

    if model_name == MODEL_NAME_MEMBRANE_2P5D:
        return MembraneUNet2p5D(**config)
    if model_name == MODEL_NAME_UNET:
        return UNet(**config)
    raise ValueError(f'Unknown model name: {model_name}')


if __name__ == "__main__":
    x = torch.randn((3, 5, 150, 150))
    model = MembraneUNet2p5D()
    preds = model(x)
    final_preds = preds[0] if isinstance(preds, list) else preds
    print(final_preds.shape, x.shape)
    assert final_preds.shape == (3, 1, 150, 150)
