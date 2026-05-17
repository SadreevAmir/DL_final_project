from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1_000,
        beta_schedule: str = "cosine",
        objective: str = "eps",
        alpha_cumprod_min: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if objective != "eps":
            raise ValueError("Only epsilon prediction is implemented")

        betas = _make_beta_schedule(timesteps, beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).clamp_min(alpha_cumprod_min)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.timesteps = timesteps
        self.objective = objective
        self.alpha_cumprod_min = float(alpha_cumprod_min)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def training_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        coords: torch.Tensor | None = None,
        min_snr_gamma: float = 5.0,
    ) -> torch.Tensor:
        batch = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (batch,), device=x_start.device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        if coords is not None:
            x_noisy = torch.cat([x_noisy, coords.expand(batch, -1, -1, -1)], dim=1)
        predicted_noise = model(x_noisy, t)
        if min_snr_gamma > 0:
            alpha_bar = self.alphas_cumprod[t].view(-1, *([1] * (x_start.ndim - 1)))
            snr = alpha_bar / (1.0 - alpha_bar)
            weight = torch.minimum(snr, snr.new_full((), float(min_snr_gamma))) / snr
            per_sample = ((predicted_noise.float() - noise.float()) ** 2).mean(
                dim=tuple(range(1, x_start.ndim)), keepdim=True
            )
            return (per_sample * weight).mean()
        return F.mse_loss(predicted_noise.float(), noise.float())

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device,
        sample_steps: int | None = None,
        coords: torch.Tensor | None = None,
        clip_pred_x0: float = 5.0,
    ) -> torch.Tensor:
        if sample_steps is not None and sample_steps != self.timesteps:
            return self.ddim_sample(model, shape, device, sample_steps, coords=coords, clip_pred_x0=clip_pred_x0)

        image = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            image = self.p_sample(model, image, t, i, coords=coords, clip_pred_x0=clip_pred_x0)
        return image

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        coords: torch.Tensor | None = None,
        clip_pred_x0: float = 5.0,
    ) -> torch.Tensor:
        betas_t = _extract(self.betas, t, x.shape)
        sqrt_alphas_cumprod_t = _extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = _extract(self.sqrt_recip_alphas, t, x.shape)

        model_input = x if coords is None else torch.cat([x, coords.expand(x.shape[0], -1, -1, -1)], dim=1)
        pred_noise = model(model_input, t)
        if clip_pred_x0 > 0:
            pred_x0 = ((x - sqrt_one_minus_alphas_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t).clamp(
                -clip_pred_x0, clip_pred_x0
            )
            pred_noise = (x - sqrt_alphas_cumprod_t * pred_x0) / sqrt_one_minus_alphas_cumprod_t

        model_mean = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
        if t_index == 0:
            return model_mean
        posterior_variance_t = _extract(self.posterior_variance, t, x.shape)
        return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device,
        sample_steps: int,
        eta: float = 0.0,
        coords: torch.Tensor | None = None,
        clip_pred_x0: float = 5.0,
    ) -> torch.Tensor:
        times = torch.linspace(-1, self.timesteps - 1, steps=sample_steps + 1, device=device).long()
        times = list(reversed(times.tolist()))
        image = torch.randn(shape, device=device)

        for time, next_time in zip(times[:-1], times[1:]):
            t = torch.full((shape[0],), time, device=device, dtype=torch.long)
            model_input = image if coords is None else torch.cat([image, coords.expand(shape[0], -1, -1, -1)], dim=1)
            pred_noise = model(model_input, t)
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[next_time] if next_time >= 0 else torch.tensor(1.0, device=device)
            pred_x0 = (image - torch.sqrt(1.0 - alpha) * pred_noise) / torch.sqrt(alpha)
            if clip_pred_x0 > 0:
                pred_x0 = pred_x0.clamp(-clip_pred_x0, clip_pred_x0)
            sigma = eta * torch.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
            c = torch.sqrt(torch.clamp(1 - alpha_next - sigma**2, min=0.0))
            noise = torch.randn_like(image) if next_time > 0 else torch.zeros_like(image)
            image = torch.sqrt(alpha_next) * pred_x0 + c * pred_noise + sigma * noise

        return image


def _extract(values: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    out = values.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def _make_beta_schedule(timesteps: int, schedule: str) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1.0e-4, 0.02, timesteps, dtype=torch.float32)
    if schedule == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0.0001, 0.9999).float()
    raise ValueError(f"Unknown beta schedule: {schedule}")
