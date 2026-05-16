from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


ParamMode = Literal["fixed", "mixed"]
Channels = Literal["v", "uv"]


@dataclass(frozen=True)
class GrayScottConfig:
    output_dir: str = "data/grayscott_64"
    total_images: int = 10_000
    grid_size: int = 64
    num_trajectories: int = 500
    max_trajectories: int = 2_000
    snapshots_per_trajectory: int = 20
    burn_in_steps: int = 2_500
    save_interval: int = 30
    chunk_size: int = 1_000
    sim_batch_size: int = 500
    dt: float = 1.0
    du: float = 0.16
    dv: float = 0.08
    param_mode: ParamMode = "mixed"
    fixed_f: float = 0.035
    fixed_k: float = 0.060
    channels: Channels = "v"
    dtype: str = "float16"
    compress: bool = False
    seed: int = 42
    device: str = "auto"
    num_threads: int = 0
    save_previews: bool = True
    preview_every_chunks: int = 1
    preview_count: int = 32
    min_image_std: float = 0.025
    min_image_range: float = 0.15


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _laplacian_periodic(x: torch.Tensor) -> torch.Tensor:
    return (
        -4.0 * x
        + torch.roll(x, shifts=1, dims=-1)
        + torch.roll(x, shifts=-1, dims=-1)
        + torch.roll(x, shifts=1, dims=-2)
        + torch.roll(x, shifts=-1, dims=-2)
    )


def _sample_parameters(
    batch: int,
    config: GrayScottConfig,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    if config.param_mode == "fixed":
        f = torch.full((batch, 1, 1), config.fixed_f, device=device)
        k = torch.full((batch, 1, 1), config.fixed_k, device=device)
        regime_id = np.zeros(batch, dtype=np.int16)
        return f, k, regime_id

    # Curated Gray-Scott regimes that tend to produce spots, worms and mixed
    # long-time patterns at 64x64 with dt=1.0.
    regimes = torch.tensor(
        [
            [0.022, 0.051],
            [0.026, 0.055],
            [0.030, 0.057],
            [0.034, 0.060],
            [0.039, 0.058],
            [0.046, 0.059],
            [0.054, 0.062],
        ],
        device=device,
        dtype=torch.float32,
    )
    ids = torch.randint(
        low=0,
        high=regimes.shape[0],
        size=(batch,),
        device=device,
        generator=generator,
    )
    base = regimes[ids]
    jitter = torch.randn((batch, 2), device=device, generator=generator) * torch.tensor(
        [0.0015, 0.0010],
        device=device,
    )
    params = base + jitter
    f = params[:, 0].clamp(0.018, 0.060).view(batch, 1, 1)
    k = params[:, 1].clamp(0.048, 0.066).view(batch, 1, 1)
    return f, k, ids.detach().cpu().numpy().astype(np.int16)


def _initial_state(
    batch: int,
    grid_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    h = w = grid_size
    u = torch.ones((batch, h, w), device=device)
    v = torch.zeros((batch, h, w), device=device)

    yy = torch.arange(h, device=device, dtype=torch.float32).view(1, h, 1)
    xx = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, w)

    max_blobs = 4
    blob_count = torch.randint(
        1,
        max_blobs + 1,
        size=(batch,),
        device=device,
        generator=generator,
    )

    for j in range(max_blobs):
        active = (blob_count > j).float().view(batch, 1, 1)
        cx = torch.rand((batch, 1, 1), device=device, generator=generator) * w
        cy = torch.rand((batch, 1, 1), device=device, generator=generator) * h
        sigma = (2.5 + 5.5 * torch.rand((batch, 1, 1), device=device, generator=generator))
        amp = 0.45 + 0.35 * torch.rand((batch, 1, 1), device=device, generator=generator)

        dx = torch.minimum((xx - cx).abs(), w - (xx - cx).abs())
        dy = torch.minimum((yy - cy).abs(), h - (yy - cy).abs())
        blob = torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)) * active

        v = v + amp * blob
        u = u - 0.55 * amp * blob

    u = u + 0.015 * torch.randn((batch, h, w), device=device, generator=generator)
    v = v + 0.015 * torch.randn((batch, h, w), device=device, generator=generator)
    return u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)


def _make_split_by_trajectory(
    num_trajectories: int,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = np.arange(num_trajectories)
    rng.shuffle(ids)
    split = np.empty(num_trajectories, dtype="<U5")
    n_train = int(round(train_frac * num_trajectories))
    n_val = int(round(val_frac * num_trajectories))
    split[ids[:n_train]] = "train"
    split[ids[n_train : n_train + n_val]] = "val"
    split[ids[n_train + n_val :]] = "test"
    return split


def _quality_mask(
    images: np.ndarray,
    min_std: float,
    min_range: float,
) -> np.ndarray:
    if min_std <= 0.0 and min_range <= 0.0:
        return np.ones(images.shape[0], dtype=bool)

    # Use the v channel as the visual/physical field of interest. For one-channel
    # datasets this is channel 0; for two-channel uv datasets it is channel 1.
    channel_index = 1 if images.shape[1] > 1 else 0
    flat = images[:, channel_index].astype(np.float32).reshape(images.shape[0], -1)
    std = flat.std(axis=1)
    value_range = np.ptp(flat, axis=1)
    return (std >= min_std) & (value_range >= min_range)


def _to_numpy_dtype(dtype: str) -> np.dtype:
    if dtype == "float16":
        return np.float16
    if dtype == "float32":
        return np.float32
    raise ValueError(f"Unsupported dtype: {dtype}. Use float16 or float32.")


def _save_chunk(
    output_dir: Path,
    chunk_id: int,
    images: list[np.ndarray],
    trajectory_id: list[np.ndarray],
    snapshot_index: list[np.ndarray],
    step: list[np.ndarray],
    f_values: list[np.ndarray],
    k_values: list[np.ndarray],
    regime_id: list[np.ndarray],
    split: list[np.ndarray],
    compress: bool,
) -> Path:
    path = output_dir / f"grayscott_chunk_{chunk_id:03d}.npz"
    payload = {
        "images": np.concatenate(images, axis=0),
        "trajectory_id": np.concatenate(trajectory_id, axis=0),
        "snapshot_index": np.concatenate(snapshot_index, axis=0),
        "step": np.concatenate(step, axis=0),
        "F": np.concatenate(f_values, axis=0),
        "k": np.concatenate(k_values, axis=0),
        "regime_id": np.concatenate(regime_id, axis=0),
        "split": np.concatenate(split, axis=0),
    }
    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    return path


def _save_preview(
    output_dir: Path,
    chunk_id: int,
    images: np.ndarray,
    max_images: int,
) -> Path:
    count = min(max_images, images.shape[0])
    if count <= 0:
        raise ValueError("Cannot save preview for an empty image array.")

    cols = min(8, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.55 * cols, 1.55 * rows))
    axes_np = np.atleast_1d(axes).ravel()

    for ax in axes_np:
        ax.axis("off")

    for ax, image in zip(axes_np, images[:count]):
        if image.shape[0] == 1:
            ax.imshow(image[0], cmap="magma", vmin=0.0, vmax=1.0)
        else:
            # For two-channel data, preview the v concentration.
            ax.imshow(image[1], cmap="magma", vmin=0.0, vmax=1.0)

    fig.suptitle(f"Gray-Scott samples, chunk {chunk_id:03d}", fontsize=12)
    fig.tight_layout()
    path = output_dir / f"preview_chunk_{chunk_id:03d}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_dataset(
    config: GrayScottConfig,
    on_chunk_saved: Callable[[Path], None] | None = None,
) -> list[Path]:
    """Generate Gray-Scott long-time snapshots and save them as NPZ chunks.

    The saved arrays contain raw concentrations in [0, 1]. For diffusion-model
    training, a common preprocessing step is `x = 2 * images - 1`.
    """

    if config.max_trajectories < config.num_trajectories:
        raise ValueError("max_trajectories must be >= num_trajectories")

    if config.max_trajectories * config.snapshots_per_trajectory < config.total_images:
        raise ValueError(
            "max_trajectories * snapshots_per_trajectory must be >= total_images"
        )

    device = _resolve_device(config.device)
    if config.num_threads > 0:
        torch.set_num_threads(config.num_threads)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(config.seed)

    np_dtype = _to_numpy_dtype(config.dtype)
    trajectory_split = _make_split_by_trajectory(config.max_trajectories, config.seed)

    saved_paths: list[Path] = []
    preview_paths: list[Path] = []
    chunk_id = 0
    emitted = 0
    rejected = 0
    simulated_trajectories = 0
    start_time = time.time()

    buffers: dict[str, list[np.ndarray]] = {
        "images": [],
        "trajectory_id": [],
        "snapshot_index": [],
        "step": [],
        "F": [],
        "k": [],
        "regime_id": [],
        "split": [],
    }
    buffer_count = 0

    def append_records(
        images_np: np.ndarray,
        traj_np: np.ndarray,
        snap_np: np.ndarray,
        step_np: np.ndarray,
        f_np: np.ndarray,
        k_np: np.ndarray,
        regime_np: np.ndarray,
        split_np: np.ndarray,
    ) -> None:
        nonlocal buffer_count, chunk_id, emitted
        remaining = images_np.shape[0]
        offset = 0
        while remaining > 0 and emitted < config.total_images:
            room = config.chunk_size - buffer_count
            take = min(room, remaining, config.total_images - emitted)

            buffers["images"].append(images_np[offset : offset + take])
            buffers["trajectory_id"].append(traj_np[offset : offset + take])
            buffers["snapshot_index"].append(snap_np[offset : offset + take])
            buffers["step"].append(step_np[offset : offset + take])
            buffers["F"].append(f_np[offset : offset + take])
            buffers["k"].append(k_np[offset : offset + take])
            buffers["regime_id"].append(regime_np[offset : offset + take])
            buffers["split"].append(split_np[offset : offset + take])

            buffer_count += take
            emitted += take
            offset += take
            remaining -= take

            if buffer_count == config.chunk_size:
                path = _save_chunk(
                    output_dir=output_dir,
                    chunk_id=chunk_id,
                    images=buffers["images"],
                    trajectory_id=buffers["trajectory_id"],
                    snapshot_index=buffers["snapshot_index"],
                    step=buffers["step"],
                    f_values=buffers["F"],
                    k_values=buffers["k"],
                    regime_id=buffers["regime_id"],
                    split=buffers["split"],
                    compress=config.compress,
                )
                saved_paths.append(path)
                if (
                    config.save_previews
                    and config.preview_every_chunks > 0
                    and chunk_id % config.preview_every_chunks == 0
                ):
                    chunk_images = np.concatenate(buffers["images"], axis=0)
                    preview_path = _save_preview(
                        output_dir=output_dir,
                        chunk_id=chunk_id,
                        images=chunk_images,
                        max_images=config.preview_count,
                    )
                    preview_paths.append(preview_path)
                if on_chunk_saved is not None:
                    on_chunk_saved(path)
                for value in buffers.values():
                    value.clear()
                buffer_count = 0
                chunk_id += 1
                progress.set_postfix(
                    chunks=len(saved_paths),
                    previews=len(preview_paths),
                    refresh=False,
                )

    total_batches = math.ceil(config.max_trajectories / config.sim_batch_size)
    progress = tqdm(total=config.total_images, desc="Gray-Scott snapshots", unit="img")

    with torch.no_grad():
        for batch_idx in range(total_batches):
            if emitted >= config.total_images:
                break

            start_id = batch_idx * config.sim_batch_size
            end_id = min(start_id + config.sim_batch_size, config.max_trajectories)
            batch = end_id - start_id
            simulated_trajectories += batch
            traj_ids = np.arange(start_id, end_id, dtype=np.int32)

            u, v = _initial_state(batch, config.grid_size, device, torch_generator)
            f, k, regime = _sample_parameters(batch, config, device, torch_generator)

            f_np = f.view(-1).detach().cpu().numpy().astype(np.float32)
            k_np = k.view(-1).detach().cpu().numpy().astype(np.float32)
            split_np = trajectory_split[traj_ids]

            total_steps = (
                config.burn_in_steps
                + config.snapshots_per_trajectory * config.save_interval
            )
            snapshot_idx = 0

            for step_idx in range(1, total_steps + 1):
                uvv = u * v * v
                u = u + config.dt * (
                    config.du * _laplacian_periodic(u) - uvv + f * (1.0 - u)
                )
                v = v + config.dt * (
                    config.dv * _laplacian_periodic(v) + uvv - (f + k) * v
                )
                u = u.clamp(0.0, 1.0)
                v = v.clamp(0.0, 1.0)

                if (
                    step_idx > config.burn_in_steps
                    and (step_idx - config.burn_in_steps) % config.save_interval == 0
                ):
                    if config.channels == "v":
                        images = v[:, None, :, :]
                    else:
                        images = torch.stack((u, v), dim=1)

                    images_np = images.detach().cpu().numpy().astype(np_dtype)
                    count = images_np.shape[0]
                    keep = _quality_mask(
                        images=images_np,
                        min_std=config.min_image_std,
                        min_range=config.min_image_range,
                    )
                    rejected += int((~keep).sum())
                    if not keep.any():
                        snapshot_idx += 1
                        progress.set_postfix(
                            chunks=len(saved_paths),
                            rejected=rejected,
                            simulated=simulated_trajectories,
                            refresh=False,
                        )
                        continue

                    append_records(
                        images_np=images_np[keep],
                        traj_np=traj_ids[keep].copy(),
                        snap_np=np.full(count, snapshot_idx, dtype=np.int16)[keep],
                        step_np=np.full(count, step_idx, dtype=np.int32)[keep],
                        f_np=f_np[keep].copy(),
                        k_np=k_np[keep].copy(),
                        regime_np=regime[keep].copy(),
                        split_np=split_np[keep].copy(),
                    )
                    snapshot_idx += 1
                    progress.n = emitted
                    progress.refresh()

                    if emitted >= config.total_images:
                        break

    if emitted < config.total_images:
        raise RuntimeError(
            f"Only accepted {emitted} images after simulating "
            f"{simulated_trajectories} trajectories. Increase max_trajectories, "
            "lower min_image_std/min_image_range, or use a more active parameter regime."
        )

    if buffer_count > 0:
        path = _save_chunk(
            output_dir=output_dir,
            chunk_id=chunk_id,
            images=buffers["images"],
            trajectory_id=buffers["trajectory_id"],
            snapshot_index=buffers["snapshot_index"],
            step=buffers["step"],
            f_values=buffers["F"],
            k_values=buffers["k"],
            regime_id=buffers["regime_id"],
            split=buffers["split"],
            compress=config.compress,
        )
        saved_paths.append(path)
        if config.save_previews and config.preview_every_chunks > 0:
            chunk_images = np.concatenate(buffers["images"], axis=0)
            preview_path = _save_preview(
                output_dir=output_dir,
                chunk_id=chunk_id,
                images=chunk_images,
                max_images=config.preview_count,
            )
            preview_paths.append(preview_path)
        if on_chunk_saved is not None:
            on_chunk_saved(path)

    progress.n = emitted
    progress.close()

    manifest = {
        "config": asdict(config),
        "device": str(device),
        "num_chunks": len(saved_paths),
        "total_images": emitted,
        "shape": [config.channels == "uv" and 2 or 1, config.grid_size, config.grid_size],
        "value_range": [0.0, 1.0],
        "elapsed_seconds": time.time() - start_time,
        "simulated_trajectories": simulated_trajectories,
        "rejected_images": rejected,
        "accepted_images": emitted,
        "files": [path.name for path in saved_paths],
        "previews": [path.name for path in preview_paths],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return saved_paths
