from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import LoadedDataset, load_dataset_into_ram
from .diffusion import GaussianDiffusion
from score_training.model import ScoreUNet


@dataclass
class TrainConfig:
    data_source: str
    output_dir: str = "runs"
    cache_dir: str = "data/download_cache"
    stats_cache_path: str = ""
    force_recompute_stats: bool = False
    dataset_tag: str = "kolmogorov_velocity"
    image_key: str = "images"
    val_fraction: float = 0.1
    seed: int = 123
    epochs: int = 100
    batch_size: int = 256
    val_batch_size: int = 128
    num_workers: int = 4
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-4
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    timesteps: int = 1_000
    beta_schedule: str = "cosine"
    sample_steps: int = 250
    sample_count: int = 32
    sample_every_epochs: int = 1
    display_samples_in_notebook: bool = False
    use_ema_for_validation: bool = False
    use_ema_for_sampling: bool = False
    channels_per_level: str = "96,192,384"
    num_res_blocks: int = 3
    dropout: float = 0.0
    ema_decay: float = 0.9999
    precision: str = "bf16"
    compile_model: bool = False
    max_train_batches: int = 0
    max_val_batches: int = 0
    limit_train: int = 0
    limit_val: int = 0
    save_last_every_epochs: int = 1
    download_best_in_colab: bool = False


def prepare_dataset(config: TrainConfig) -> LoadedDataset:
    print("Preparing dataset")
    print(f"Data source: {config.data_source}")
    print(f"Cache directory: {config.cache_dir}")
    dataset = load_dataset_into_ram(
        data_source=config.data_source,
        cache_dir=config.cache_dir,
        val_fraction=config.val_fraction,
        seed=config.seed,
        image_key=config.image_key,
        limit_train=config.limit_train,
        limit_val=config.limit_val,
        stats_cache_path=config.stats_cache_path,
        force_recompute_stats=config.force_recompute_stats,
    )
    print(f"Dataset ready: train={len(dataset.train)}, val={len(dataset.val)}")
    print(f"Stats: mean={dataset.stats.mean}, std={dataset.stats.std}")
    return dataset


def train_diffusion_model(config: TrainConfig, dataset: LoadedDataset | None = None) -> Path:
    _set_reproducibility(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _configure_torch_for_a100()

    if dataset is None:
        dataset = prepare_dataset(config)

    run_dir = _make_run_dir(config, dataset)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", asdict(config))
    _write_json(run_dir / "data_stats.json", dataset.stats.to_dict())
    _write_json(run_dir / "dataset_files.json", {"files": dataset.files})

    train_loader = _make_loader(dataset.train, config.batch_size, config.num_workers, shuffle=True)
    val_loader = _make_loader(dataset.val, config.val_batch_size, config.num_workers, shuffle=False)

    coords = _make_coord_grid(dataset.stats.height, dataset.stats.width, device)
    model = ScoreUNet(
        in_channels=dataset.stats.channels + 2,
        out_channels=dataset.stats.channels,
        channels_per_level=_parse_int_tuple(config.channels_per_level),
        num_res_blocks=config.num_res_blocks,
        image_size=dataset.stats.height,
        dropout=config.dropout,
    ).to(device)
    ema_model = ScoreUNet(
        in_channels=dataset.stats.channels + 2,
        out_channels=dataset.stats.channels,
        channels_per_level=_parse_int_tuple(config.channels_per_level),
        num_res_blocks=config.num_res_blocks,
        image_size=dataset.stats.height,
        dropout=config.dropout,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)

    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    diffusion = GaussianDiffusion(
        timesteps=config.timesteps,
        beta_schedule=config.beta_schedule,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16" and device.type == "cuda")

    history: list[dict[str, float | int | str]] = []
    best_val = float("inf")
    best_path = run_dir / "checkpoints" / f"best_{run_dir.name}.pt"
    global_step = 0

    print("Starting training")
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Loaded dataset in RAM: train={len(dataset.train)}, val={len(dataset.val)}")

    for epoch in range(1, config.epochs + 1):
        train_loss, global_step = _train_one_epoch(
            model=model,
            ema_model=ema_model,
            diffusion=diffusion,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            coords=coords,
            config=config,
            global_step=global_step,
            epoch=epoch,
        )
        eval_model = ema_model if config.use_ema_for_validation else _unwrap_model(model)
        sample_model = ema_model if config.use_ema_for_sampling else _unwrap_model(model)

        val_loss = _validate(
            model=eval_model,
            diffusion=diffusion,
            loader=val_loader,
            device=device,
            coords=coords,
            config=config,
            epoch=epoch,
        )

        improved = val_loss < best_val
        if improved:
            best_val = val_loss

        if config.sample_every_epochs > 0 and epoch % config.sample_every_epochs == 0:
            sample_path = run_dir / "samples" / f"epoch_{epoch:04d}_val_{val_loss:.6f}.png"
            _save_samples(
                model=sample_model,
                diffusion=diffusion,
                dataset=dataset,
                device=device,
                coords=coords,
                config=config,
                path=sample_path,
            )
            print(f"Saved unconditional samples: {sample_path}")
            if config.display_samples_in_notebook:
                _display_image_in_notebook(sample_path)

        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "improved": int(improved),
        }
        history.append(record)
        _write_json(run_dir / "history.json", {"history": history})

        should_save_last = config.save_last_every_epochs > 0 and epoch % config.save_last_every_epochs == 0
        if should_save_last:
            _save_checkpoint(
                run_dir / "checkpoints" / f"last_{run_dir.name}_epoch_{epoch:04d}.pt",
                model,
                ema_model,
                optimizer,
                config,
                dataset,
                epoch,
                global_step,
                best_val,
                history,
            )

        if improved:
            epoch_best_path = run_dir / "checkpoints" / f"best_{run_dir.name}_epoch_{epoch:04d}_val_{val_loss:.6f}.pt"
            _save_checkpoint(
                epoch_best_path,
                model,
                ema_model,
                optimizer,
                config,
                dataset,
                epoch,
                global_step,
                best_val,
                history,
            )
            shutil.copy2(epoch_best_path, best_path)
            if config.download_best_in_colab:
                _download_in_colab(best_path)

        print(
            f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best={best_val:.6f} improved={improved}"
        )

    return best_path


def _train_one_epoch(
    model: nn.Module,
    ema_model: nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader[torch.Tensor],
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    coords: torch.Tensor,
    config: TrainConfig,
    global_step: int,
    epoch: int,
) -> tuple[float, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    pending_backward = False

    for batch_idx, batch in enumerate(progress, start=1):
        if config.max_train_batches > 0 and batch_idx > config.max_train_batches:
            break
        batch = batch.to(device, non_blocking=True)
        with _autocast_context(device, config.precision):
            loss = diffusion.training_loss(model, batch, coords=coords) / config.grad_accum_steps
        scaler.scale(loss).backward()
        pending_backward = True

        if batch_idx % config.grad_accum_steps == 0:
            _optimizer_step(model, ema_model, optimizer, scaler, config)
            global_step += 1
            pending_backward = False

        loss_value = float(loss.detach().item() * config.grad_accum_steps)
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.4f}", step=global_step)

    if pending_backward:
        _optimizer_step(model, ema_model, optimizer, scaler, config)
        global_step += 1

    return float(np.mean(losses)), global_step


@torch.no_grad()
def _validate(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    coords: torch.Tensor,
    config: TrainConfig,
    epoch: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"val epoch {epoch}", leave=False), start=1):
        if config.max_val_batches > 0 and batch_idx > config.max_val_batches:
            break
        batch = batch.to(device, non_blocking=True)
        with _autocast_context(device, config.precision):
            loss = diffusion.training_loss(model, batch, coords=coords)
        losses.append(float(loss.detach().item()))
    return float(np.mean(losses))


@torch.no_grad()
def _save_samples(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    dataset: LoadedDataset,
    device: torch.device,
    coords: torch.Tensor,
    config: TrainConfig,
    path: Path,
) -> None:
    model.eval()
    shape = (
        config.sample_count,
        dataset.stats.channels,
        dataset.stats.height,
        dataset.stats.width,
    )
    with _autocast_context(device, config.precision):
        samples = diffusion.sample(model, shape, device=device, sample_steps=config.sample_steps, coords=coords)
    samples = samples.float().cpu().numpy()
    mean = np.asarray(dataset.stats.mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.asarray(dataset.stats.std, dtype=np.float32).reshape(1, -1, 1, 1)
    raw = samples * std + mean
    _save_preview_grid(raw, path)


def _save_preview_grid(images: np.ndarray, path: Path) -> None:
    count = images.shape[0]
    cols = min(8, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.65 * cols, 1.65 * rows))
    axes_np = np.atleast_1d(axes).ravel()
    for ax in axes_np:
        ax.axis("off")

    visual = _visualize_field(images)
    vmax = float(np.nanpercentile(np.abs(visual), 99.0))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    for ax, image in zip(axes_np, visual):
        ax.imshow(image, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _visualize_field(images: np.ndarray) -> np.ndarray:
    if images.shape[1] == 1:
        return images[:, 0]
    if images.shape[1] == 2:
        ux = images[:, 0]
        uy = images[:, 1]
        return 0.5 * (np.roll(uy, -1, axis=-1) - np.roll(uy, 1, axis=-1)) - 0.5 * (
            np.roll(ux, -1, axis=-2) - np.roll(ux, 1, axis=-2)
        )
    return images[:, 0]


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: AdamW,
    config: TrainConfig,
    dataset: LoadedDataset,
    epoch: int,
    global_step: int,
    best_val: float,
    history: list[dict[str, float | int | str]],
) -> None:
    payload = {
        "model": _unwrap_model(model).state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "data_stats": dataset.stats.to_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val,
        "history": history,
    }
    torch.save(payload, path)


def _make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader[torch.Tensor]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )


def _make_coord_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=0).unsqueeze(0)


def _optimizer_step(
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
) -> None:
    scaler.unscale_(optimizer)
    if config.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    _update_ema(ema_model, _unwrap_model(model), config.ema_decay)


def _make_run_dir(config: TrainConfig, dataset: LoadedDataset) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = (
        f"ddpm_{config.dataset_tag}_{dataset.stats.channels}ch_"
        f"{dataset.stats.height}x{dataset.stats.width}_coords_{config.precision}_{timestamp}"
    )
    path = Path(config.output_dir) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, ema_param in ema_params.items():
        ema_param.data.mul_(decay).add_(model_params[name].data, alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(model_buffers[name])


def _unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _download_in_colab(path: Path) -> None:
    try:
        from google.colab import files  # type: ignore

        files.download(str(path))
    except Exception as exc:  # pragma: no cover - only used in Colab.
        print(f"Could not trigger Colab download for {path}: {exc}")


def _display_image_in_notebook(path: Path) -> None:
    try:
        from IPython.display import display  # type: ignore
        from PIL import Image

        display(Image.open(path))
    except Exception as exc:  # pragma: no cover - notebook convenience only.
        print(f"Could not display sample image {path}: {exc}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _set_reproducibility(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_torch_for_a100() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
