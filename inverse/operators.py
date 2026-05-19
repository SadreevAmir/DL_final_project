from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class LinearOperator(torch.nn.Module):
    name: str

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def pinv(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} does not provide a pseudoinverse.")

    def observation_to_image(self, y: torch.Tensor) -> torch.Tensor:
        try:
            return self.pinv(y)
        except NotImplementedError:
            return y

    def add_noise(
        self,
        y: torch.Tensor,
        noise_sigma: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if noise_sigma <= 0.0:
            return y
        return y + noise_sigma * torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)


class MaskOperator(LinearOperator):
    def __init__(self, mask: torch.Tensor, name: str = "mask") -> None:
        super().__init__()
        self.name = name
        self.register_buffer("_mask", mask.float())

    @property
    def mask_tensor(self) -> torch.Tensor:
        return self._mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._mask

    def pinv(self, y: torch.Tensor) -> torch.Tensor:
        return y * self._mask

    def add_noise(
        self,
        y: torch.Tensor,
        noise_sigma: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if noise_sigma <= 0.0:
            return y
        noise = torch.randn(y.shape, device=y.device, dtype=y.dtype, generator=generator)
        return y + noise_sigma * noise * self._mask


class DownsampleOperator(LinearOperator):
    def __init__(self, factor: int = 4) -> None:
        super().__init__()
        self.factor = int(factor)
        self.name = f"downsample_x{self.factor}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=self.factor, stride=self.factor)

    def pinv(self, y: torch.Tensor) -> torch.Tensor:
        return y.repeat_interleave(self.factor, dim=-2).repeat_interleave(self.factor, dim=-1)


class PeriodicGaussianBlurOperator(LinearOperator):
    def __init__(self, channels: int, sigma: float = 2.0, truncate: float = 4.0) -> None:
        super().__init__()
        self.channels = int(channels)
        self.sigma = float(sigma)
        self.truncate = float(truncate)
        self.name = f"periodic_gaussian_blur_sigma{self.sigma:g}"
        kernel = _gaussian_kernel2d(self.sigma, self.truncate)
        self.register_buffer("kernel", kernel.view(1, 1, *kernel.shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        radius = self.kernel.shape[-1] // 2
        padded = F.pad(x, (radius, radius, radius, radius), mode="circular")
        weight = self.kernel.to(dtype=x.dtype, device=x.device).expand(x.shape[1], 1, -1, -1)
        return F.conv2d(padded, weight, groups=x.shape[1])


def _gaussian_kernel2d(sigma: float, truncate: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(truncate * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def build_sparse_grid_mask(height: int, width: int, stride: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((1, 1, height, width), device=device)
    mask[..., ::stride, ::stride] = 1.0
    return mask


def build_operator(
    name: str,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
    stride: int = 4,
    downsample_factor: int = 4,
    blur_sigma: float = 2.0,
) -> LinearOperator:
    if name == "sparse_grid":
        mask = build_sparse_grid_mask(height, width, stride=stride, device=device)
        return MaskOperator(mask=mask, name=f"sparse_grid_stride{stride}").to(device)
    if name == "downsample":
        return DownsampleOperator(factor=downsample_factor).to(device)
    if name == "blur":
        return PeriodicGaussianBlurOperator(channels=channels, sigma=blur_sigma).to(device)
    raise ValueError(f"Unknown operator: {name!r}")
