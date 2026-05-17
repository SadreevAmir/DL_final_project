from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10_000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class LayerNorm2d(nn.Module):
    """LayerNorm-style normalization over all non-batch dimensions for a 2D feature map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + 1.0e-5) * self.weight + self.bias


class CircularConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="circular")
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.conv1 = CircularConv2d(in_channels, out_channels, kernel_size=3)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = LayerNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CircularConv2d(out_channels, out_channels, kernel_size=3)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = CircularConv2d(channels, channels, kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = CircularConv2d(channels, channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ScoreUNet(nn.Module):
    """U-Net for epsilon prediction with clean coordinate channels in the input.

    The default widths and depth match the Kolmogorov-flow table in the SDA paper:
    channels per level (96, 192, 384), 3 residual blocks per level, circular padding,
    SiLU activations and LayerNorm-style normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels_per_level: tuple[int, ...] = (96, 192, 384),
        num_res_blocks: int = 3,
        image_size: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % (2 ** (len(channels_per_level) - 1)) != 0:
            raise ValueError("image_size must be divisible by the UNet downsampling factor")

        base_channels = channels_per_level[0]
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = CircularConv2d(in_channels, base_channels, kernel_size=3)

        ch = base_channels
        skip_channels = [ch]
        self.downs = nn.ModuleList()
        for level, out_ch in enumerate(channels_per_level):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
                skip_channels.append(ch)
            downsample = Downsample(ch) if level != len(channels_per_level) - 1 else nn.Identity()
            if level != len(channels_per_level) - 1:
                skip_channels.append(ch)
            self.downs.append(nn.ModuleDict({"blocks": blocks, "downsample": downsample}))

        self.mid1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout)

        self.ups = nn.ModuleList()
        for level, out_ch in reversed(list(enumerate(channels_per_level))):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout))
                ch = out_ch
            upsample = Upsample(ch) if level != 0 else nn.Identity()
            self.ups.append(nn.ModuleDict({"blocks": blocks, "upsample": upsample}))

        self.output = nn.Sequential(
            LayerNorm2d(ch),
            nn.SiLU(),
            CircularConv2d(ch, out_channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(t_embed)
        h = self.input(x)
        skips = [h]

        for down in self.downs:
            for block in down["blocks"]:
                h = block(h, time_emb)
                skips.append(h)
            h = down["downsample"](h)
            if not isinstance(down["downsample"], nn.Identity):
                skips.append(h)

        h = self.mid2(self.mid1(h, time_emb), time_emb)

        for up in self.ups:
            for block in up["blocks"]:
                h = torch.cat([h, skips.pop()], dim=1)
                h = block(h, time_emb)
            h = up["upsample"](h)

        return self.output(h)
