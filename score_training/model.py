from __future__ import annotations

from diffusion_training.unet import DiffusersUNet


# Backward-compatible name for older imports/checkpoints.
ScoreUNet = DiffusersUNet

__all__ = ["DiffusersUNet", "ScoreUNet"]
