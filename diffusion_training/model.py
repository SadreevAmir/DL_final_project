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


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = _group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.norm = _group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = (c // self.heads) ** -0.5
        attention = torch.einsum("bhdn,bhdm->bhnm", q * scale, k).softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attention, v).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (16,),
        image_size: int = 64,
        dropout: float = 0.0,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if image_size % (2 ** (len(channel_mults) - 1)) != 0:
            raise ValueError("image_size must be divisible by the UNet downsampling factor")

        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        ch = base_channels
        resolution = image_size
        skip_channels = [ch]
        self.downs = nn.ModuleList()
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
                attentions.append(
                    AttentionBlock(ch, attention_heads) if resolution in attention_resolutions else nn.Identity()
                )
                skip_channels.append(ch)
            downsample = Downsample(ch) if level != len(channel_mults) - 1 else nn.Identity()
            if level != len(channel_mults) - 1:
                skip_channels.append(ch)
            self.downs.append(nn.ModuleDict({"blocks": blocks, "attentions": attentions, "downsample": downsample}))
            if level != len(channel_mults) - 1:
                resolution //= 2

        self.mid1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(ch, attention_heads)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout)

        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout))
                ch = out_ch
                attentions.append(
                    AttentionBlock(ch, attention_heads) if resolution in attention_resolutions else nn.Identity()
                )
            upsample = Upsample(ch) if level != 0 else nn.Identity()
            self.ups.append(nn.ModuleDict({"blocks": blocks, "attentions": attentions, "upsample": upsample}))
            if level != 0:
                resolution *= 2

        self.output = nn.Sequential(
            _group_norm(ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(timesteps)
        h = self.input(x)
        skips = [h]

        for down in self.downs:
            for block, attention in zip(down["blocks"], down["attentions"]):
                h = attention(block(h, time_emb))
                skips.append(h)
            h = down["downsample"](h)
            if not isinstance(down["downsample"], nn.Identity):
                skips.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, time_emb)), time_emb)

        for up in self.ups:
            for block, attention in zip(up["blocks"], up["attentions"]):
                h = torch.cat([h, skips.pop()], dim=1)
                h = attention(block(h, time_emb))
            h = up["upsample"](h)

        return self.output(h)


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)
