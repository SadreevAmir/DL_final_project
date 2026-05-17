from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class VPCosineSDE(nn.Module):
    """Continuous VP SDE schedule used in Score-Based Data Assimilation.

    mu(t) = cos(omega t), sigma(t) = sqrt(1 - mu(t)^2),
    omega = arccos(1e-3), t in [0, 1].
    """

    def __init__(self, sigma_min_mu: float = 1.0e-3, t_eps: float = 1.0e-5) -> None:
        super().__init__()
        self.t_eps = t_eps
        self.register_buffer("omega", torch.tensor(math.acos(sigma_min_mu), dtype=torch.float32))

    def mu_sigma(self, t: torch.Tensor, x_shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        mu = torch.cos(self.omega * t).clamp_min(1.0e-4)
        sigma = torch.sqrt(torch.clamp(1.0 - mu * mu, min=1.0e-8))
        view_shape = (t.shape[0],) + (1,) * (len(x_shape) - 1)
        return mu.view(view_shape), sigma.view(view_shape)

    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        coords: torch.Tensor,
        time_embedding_scale: float = 999.0,
    ) -> torch.Tensor:
        batch = x0.shape[0]
        t = self.t_eps + (1.0 - self.t_eps) * torch.rand(batch, device=x0.device)
        noise = torch.randn_like(x0)
        mu, sigma = self.mu_sigma(t, x0.shape)
        xt = mu * x0 + sigma * noise
        model_input = torch.cat([xt, coords.expand(batch, -1, -1, -1)], dim=1)
        pred_noise = model(model_input, t * time_embedding_scale)
        return F.mse_loss(pred_noise.float(), noise.float())

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        coords: torch.Tensor,
        steps: int,
        device: torch.device,
        time_embedding_scale: float = 999.0,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        times = torch.linspace(1.0, 0.0, steps + 1, device=device)
        coords_batch = coords.expand(shape[0], -1, -1, -1)

        for t_value, next_t_value in zip(times[:-1], times[1:]):
            t = torch.full((shape[0],), float(t_value), device=device)
            t_next = torch.full((shape[0],), float(next_t_value), device=device)
            mu, sigma = self.mu_sigma(t, x.shape)
            mu_next, sigma_next = self.mu_sigma(t_next, x.shape)
            pred_noise = model(torch.cat([x, coords_batch], dim=1), t * time_embedding_scale)
            score = -pred_noise / sigma

            # Deterministic VP/DDIM update for an epsilon-prediction model.
            # Since score = -eps/sigma, this is equivalent to:
            # x_next = mu_next * x0_hat + sigma_next * eps_hat.
            x = (mu_next / mu) * x - mu_next * (sigma_next / mu_next - sigma / mu) * sigma * score

        return x
