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
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusion_training.data import LoadedDataset, load_dataset_into_ram
from .model import ScoreUNet
from .sde import VPCosineSDE


@dataclass
class ScoreTrainConfig:
    data_source: str
    output_dir: str = "runs_score"
    cache_dir: str = "data/download_cache"
    stats_cache_path: str = ""
    force_recompute_stats: bool = False
    dataset_tag: str = "kolmogorov_velocity"
    image_key: str = "images"
    val_fraction: float = 0.1
    seed: int = 123
    epochs: int = 1024
    batches_per_epoch: int = 128
    val_batches: int = 64
    batch_size: int = 32
    val_batch_size: int = 64
    num_workers: int = 4
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-3
    max_grad_norm: float = 1.0
    sample_steps: int = 256
    sample_count: int = 32
    sample_every_epochs: int = 1
    display_samples_in_notebook: bool = True
    channels_per_level: str = "96,192,384"
    num_res_blocks: int = 3
    dropout: float = 0.0
    precision: str = "bf16"
    compile_model: bool = False
    limit_train: int = 0
    limit_val: int = 0
    save_last_every_epochs: int = 10
    download_best_in_colab: bool = False


def prepare_score_dataset(config: ScoreTrainConfig) -> LoadedDataset:
    print("Preparing score-based dataset")
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
    print("Coordinate channels are added at training time and are never noised.")
    return dataset


def train_score_model(config: ScoreTrainConfig, dataset: LoadedDataset | None = None) -> Path:
    _set_reproducibility(config.seed)
    _configure_torch_for_a100()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset is None:
        dataset = prepare_score_dataset(config)

    run_dir = _make_run_dir(config, dataset)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", asdict(config))
    _write_json(run_dir / "data_stats.json", dataset.stats.to_dict())
    _write_json(run_dir / "dataset_files.json", {"files": dataset.files})

    coords = _make_coord_grid(dataset.stats.height, dataset.stats.width, device)
    data_channels = dataset.stats.channels
    model = ScoreUNet(
        in_channels=data_channels + 2,
        out_channels=data_channels,
        channels_per_level=_parse_int_tuple(config.channels_per_level),
        num_res_blocks=config.num_res_blocks,
        image_size=dataset.stats.height,
        dropout=config.dropout,
    ).to(device)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    sde = VPCosineSDE().to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max(config.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16" and device.type == "cuda")
    train_loader = DataLoader(
        dataset.train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset.val,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        drop_last=False,
    )

    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_path = run_dir / "checkpoints" / f"best_{run_dir.name}.pt"
    global_step = 0
    print("Starting score-based VP SDE training")
    print(f"Run directory: {run_dir}")
    print(f"Model input channels: {data_channels} noisy data + 2 clean coordinate channels")

    for epoch in range(1, config.epochs + 1):
        train_loss, global_step = _train_epoch(
            model=model,
            sde=sde,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            coords=coords,
            device=device,
            config=config,
            epoch=epoch,
            global_step=global_step,
        )
        val_loss = _validate(model, sde, val_loader, coords, device, config, epoch)
        scheduler.step()
        improved = val_loss < best_val
        if improved:
            best_val = val_loss

        if config.sample_every_epochs > 0 and epoch % config.sample_every_epochs == 0:
            sample_path = run_dir / "samples" / f"epoch_{epoch:04d}_val_{val_loss:.6f}.png"
            _save_samples(model, sde, dataset, coords, device, config, sample_path)
            print(f"Saved score samples: {sample_path}")
            if config.display_samples_in_notebook:
                _display_image_in_notebook(sample_path)

        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "lr": float(scheduler.get_last_lr()[0]),
            "improved": int(improved),
        }
        history.append(record)
        _write_json(run_dir / "history.json", {"history": history})

        if config.save_last_every_epochs > 0 and epoch % config.save_last_every_epochs == 0:
            _save_checkpoint(run_dir / "checkpoints" / f"last_{run_dir.name}_epoch_{epoch:04d}.pt", model, optimizer, config, dataset, epoch, global_step, best_val, history)

        if improved:
            epoch_best_path = run_dir / "checkpoints" / f"best_{run_dir.name}_epoch_{epoch:04d}_val_{val_loss:.6f}.pt"
            _save_checkpoint(epoch_best_path, model, optimizer, config, dataset, epoch, global_step, best_val, history)
            shutil.copy2(epoch_best_path, best_path)
            if config.download_best_in_colab:
                _download_in_colab(best_path)

        print(
            f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best={best_val:.6f} lr={scheduler.get_last_lr()[0]:.3e} improved={improved}"
        )

    return best_path


def _train_epoch(
    model: nn.Module,
    sde: VPCosineSDE,
    loader: DataLoader[torch.Tensor],
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    coords: torch.Tensor,
    device: torch.device,
    config: ScoreTrainConfig,
    epoch: int,
    global_step: int,
) -> tuple[float, int]:
    model.train()
    losses: list[float] = []
    progress = tqdm(loader, desc=f"score train epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(progress, start=1):
        if config.batches_per_epoch > 0 and batch_idx > config.batches_per_epoch:
            break
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, config.precision):
            loss = sde.training_loss(model, batch, coords)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        value = float(loss.detach().item())
        losses.append(value)
        progress.set_postfix(loss=f"{value:.4f}", step=global_step)
    return float(np.mean(losses)), global_step


@torch.no_grad()
def _validate(
    model: nn.Module,
    sde: VPCosineSDE,
    loader: DataLoader[torch.Tensor],
    coords: torch.Tensor,
    device: torch.device,
    config: ScoreTrainConfig,
    epoch: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"score val epoch {epoch}", leave=False), start=1):
        if config.val_batches > 0 and batch_idx > config.val_batches:
            break
        batch = batch.to(device, non_blocking=True)
        with _autocast_context(device, config.precision):
            loss = sde.training_loss(model, batch, coords)
        losses.append(float(loss.detach().item()))
    return float(np.mean(losses))


@torch.no_grad()
def _save_samples(
    model: nn.Module,
    sde: VPCosineSDE,
    dataset: LoadedDataset,
    coords: torch.Tensor,
    device: torch.device,
    config: ScoreTrainConfig,
    path: Path,
) -> None:
    shape = (config.sample_count, dataset.stats.channels, dataset.stats.height, dataset.stats.width)
    with _autocast_context(device, config.precision):
        samples = sde.sample(model, shape, coords, steps=config.sample_steps, device=device)
    samples_np = samples.float().cpu().numpy()
    mean = np.asarray(dataset.stats.mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.asarray(dataset.stats.std, dtype=np.float32).reshape(1, -1, 1, 1)
    _save_preview_grid(samples_np * std + mean, path)


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
    ux = images[:, 0]
    uy = images[:, 1]
    return 0.5 * (np.roll(uy, -1, axis=-1) - np.roll(uy, 1, axis=-1)) - 0.5 * (
        np.roll(ux, -1, axis=-2) - np.roll(ux, 1, axis=-2)
    )


def _make_coord_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=0).unsqueeze(0)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    config: ScoreTrainConfig,
    dataset: LoadedDataset,
    epoch: int,
    global_step: int,
    best_val: float,
    history: list[dict[str, float | int]],
) -> None:
    torch.save(
        {
            "model": _unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "data_stats": dataset.stats.to_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val,
            "history": history,
            "coordinate_channels": ["x", "y"],
            "coordinate_range": [-1.0, 1.0],
        },
        path,
    )


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _display_image_in_notebook(path: Path) -> None:
    try:
        from IPython.display import display  # type: ignore
        from PIL import Image

        display(Image.open(path))
    except Exception as exc:  # pragma: no cover
        print(f"Could not display sample image {path}: {exc}")


def _download_in_colab(path: Path) -> None:
    try:
        from google.colab import files  # type: ignore

        files.download(str(path))
    except Exception as exc:  # pragma: no cover
        print(f"Could not trigger Colab download for {path}: {exc}")


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
