from __future__ import annotations

import torch
from torch import nn


class DiffusersUNet(nn.Module):
    """Thin wrapper around diffusers.UNet2DModel for epsilon prediction.

    The training code calls models as ``model(x, timestep)``. DDPM passes integer
    timesteps in ``[0, num_train_timesteps - 1]``. The VP-SDE code maps continuous
    ``t in [0, 1]`` onto the same timestep scale before calling this module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels_per_level: tuple[int, ...] = (96, 192, 384),
        num_res_blocks: int = 3,
        image_size: int = 64,
        dropout: float = 0.0,
        attention_head_dim: int = 32,
    ) -> None:
        super().__init__()
        try:
            from diffusers import UNet2DModel
        except ImportError as exc:  # pragma: no cover - dependency error path.
            raise ImportError(
                "diffusers is required for training. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        if len(channels_per_level) < 2:
            raise ValueError("channels_per_level must contain at least two levels")

        down_block_types = ["DownBlock2D"] + ["AttnDownBlock2D"] * (len(channels_per_level) - 1)
        up_block_types = ["AttnUpBlock2D"] * (len(channels_per_level) - 1) + ["UpBlock2D"]

        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=num_res_blocks,
            block_out_channels=channels_per_level,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            dropout=dropout,
            act_fn="silu",
            norm_num_groups=32,
            attention_head_dim=attention_head_dim,
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.model(x, timesteps, return_dict=False)[0]
