# Gray-Scott Snapshot Dataset

Мини-репозиторий для генерации датасета из `10_000` Gray-Scott reaction-diffusion snapshots.
Идея: симулировать динамическую PDE, пропустить начальный transient через `burn_in_steps`, а затем
сохранять отдельные состояния как картинки из long-time regime.

Это не полноценная trajectory data assimilation. Это датасет для более простой задачи:

```text
single PDE state reconstruction from sparse/noisy observations
```

## Доступные датасеты

- `grayscott_dataset/`: быстрый reaction-diffusion датасет с attractor-like patterns.
- `kolmogorov_dataset/`: более близкий к статье fluid-like датасет, forced 2D incompressible
  Navier-Stokes in vorticity form with Kolmogorov-style forcing. Текущий Colab preset хранит
  raw velocity snapshots `(u_x, u_y)` в `float32`, без scaling/clipping, а preview строит по raw vorticity через обычный `imshow`.
  Симуляция идет на `256x256`, затем average-pooling downscale до `64x64`. Для более контрастных
  vortex-like states используется stronger forcing / lower damping.

Готовые Colab variants:

```text
notebooks/grayscott_128_10k/generate_grayscott_128_10k_colab.ipynb
notebooks/kolmogorov_64_10k/generate_kolmogorov_64_10k_colab.ipynb
notebooks/kolmogorov_256_to_64_100k/generate_kolmogorov_256_to_64_100k_colab.ipynb
```

## Обучение diffusion prior

Код обучения находится в:

```text
diffusion_training/
scripts/train_diffusion.py
notebooks/train_diffusion_a100_bf16_colab.ipynb
```

Пайплайн обучает unconditional DDPM/VP diffusion prior на сохраненных PDE snapshots. Перед
обучением датасет целиком загружается в RAM, по train split считаются channel-wise `mean` и `std`,
после чего train/val нормализуются по этим статистикам. Каждую эпоху считается validation loss,
сохраняются unconditional samples для наблюдения за прогрессом и checkpoint с `model`, `ema_model`,
optimizer state, config, dataset stats и history. Если validation loss улучшился, сохраняется
`best_*.pt`; в Colab notebook этот файл дополнительно скачивается через `files.download`.

DDPM baseline и score-based baseline используют одинаковую модель и batch size:
`diffusers.UNet2DModel` с channels `(96, 192, 384)`, `3` residual blocks per level,
attention на нижних разрешениях, `attention_head_dim=32`, SiLU, batch size `128`. В обоих
вариантах к входу модели добавляются clean coordinate channels `(x, y)` в диапазоне `[-1, 1]`;
они никогда не шумятся и не входят в loss.

`--data-source` может быть:

- публичной ссылкой Yandex Disk на `.zip`, `.npz` или папку;
- локальным `.zip`/`.npz`;
- локальной папкой с `.npz` chunks или split shards.

Для публичной папки Yandex Disk загрузчик скачивает `.npz` и `.json` файлы по отдельности в
`cache_dir` и пропускает уже скачанные файлы с совпадающим размером. Эта ссылка работает напрямую:

```text
https://disk.yandex.ru/d/rrjDGzzX5cfFnA
```

В этой папке сейчас есть `train_000..015.npz`, `val_000..001.npz` и `test_000.npz`. Для обучения
используются только train/val, поэтому отсутствующий `test_001.npz` не блокирует training pipeline.

Статистики нормализации сохраняются в JSON. Если указать `--stats-cache-path`, то при следующем
запуске `mean/std` будут загружены из этого файла, а не пересчитаны:

Пример локального запуска:

```bash
python3 scripts/train_diffusion.py \
  --data-source https://disk.yandex.ru/d/rrjDGzzX5cfFnA \
  --stats-cache-path data/download_cache/kolmogorov_velocity_256_to_64_train_stats.json \
  --dataset-tag kolmogorov_velocity_256_to_64 \
  --epochs 100 \
  --batch-size 128 \
  --val-batch-size 128 \
  --precision bf16 \
  --channels-per-level 96,192,384 \
  --num-res-blocks 3 \
  --attention-head-dim 32 \
  --sample-every-epochs 1 \
  --sample-steps 250
```

Для Colab/A100 используйте:

```text
notebooks/train_diffusion_a100_bf16_colab.ipynb
```

В notebook нужно указать `DATA_SOURCE` как публичную ссылку Yandex Disk или путь к архиву/папке.
Preset оптимизирован под `64x64` Kolmogorov velocity snapshots `[N, 2, 64, 64]`, A100 40GB и bf16.
Если памяти не хватает, уменьшите `batch_size` до `64`.

В Colab notebook процесс разделен на отдельные ячейки:

- `Configure training`: создает `TrainConfig`, ничего не скачивает и не обучает.
- `Download and load dataset`: скачивает или проверяет cache, грузит train/val в RAM, считает или
  читает `mean/std`, нормализует данные.
- `Start training`: запускает обучение модели на уже подготовленном `dataset`.

Отдельный score-based VP SDE вариант с clean coordinate channels:

```text
score_training/
scripts/train_score_vp_coords.py
notebooks/train_score_vp_coords_a100_bf16_colab.ipynb
```

В этом варианте физические velocity channels шумятся по VP SDE из статьи:

```text
mu(t) = cos(arccos(1e-3) * t)
sigma(t) = sqrt(1 - mu(t)^2)
```

К входу `diffusers.UNet2DModel` на каждом timestep добавляются два координатных канала `(x, y)` в диапазоне
`[-1, 1]`. Они подаются всегда без шума и не входят в loss; модель предсказывает epsilon только
для физических каналов. Архитектура совпадает с DDPM вариантом: channels `(96, 192, 384)`,
`3` residual blocks per level, attention на нижних разрешениях, AdamW, weight decay `1e-3`,
linear LR decay, `256` sampling steps. Непрерывное VP-SDE время `t in [0, 1]` передается в
diffusers как `t * 999`, то есть в том же масштабе timestep, что DDPM `0..999`. Эпоха в этом
варианте теперь означает полный проход по train loader; `batches_per_epoch=0` и `val_batches=0`.

CLI:

```bash
python3 scripts/train_score_vp_coords.py \
  --data-source https://disk.yandex.ru/d/rrjDGzzX5cfFnA \
  --stats-cache-path data/download_cache/kolmogorov_velocity_256_to_64_train_stats.json \
  --dataset-tag kolmogorov_velocity_256_to_64_coords \
  --epochs 1024 \
  --batches-per-epoch 0 \
  --batch-size 128 \
  --precision bf16 \
  --time-embedding-scale 999
```

## Что генерируется

Gray-Scott system:

```text
u_t = D_u Laplacian(u) - u v^2 + F(1 - u)
v_t = D_v Laplacian(v) + u v^2 - (F + k) v
```

Быстрый preset по умолчанию:

- grid: `64x64`
- channels: один канал `v`
- trajectories: `500`
- max trajectories if quality filtering rejects samples: `2000`
- snapshots per trajectory: `20`
- total images: `10_000`
- burn-in: `2500` explicit Euler steps
- save interval: `30` steps
- solver substeps: `1`
- chunk size: `1000` images
- preview: PNG-сетка samples после каждого chunk
- sequence preview: подряд идущие snapshots одной trajectory
- quality filter: отбрасывает почти однородные snapshots
- boundary condition: periodic
- saved values: raw concentrations in `[0, 1]`

Для diffusion training обычно использовать:

```python
x = 2.0 * images - 1.0
```

## Быстрый запуск

```bash
pip install -r requirements.txt
python scripts/generate_grayscott.py \
  --output-dir data/grayscott_64 \
  --total-images 10000 \
  --num-trajectories 500 \
  --max-trajectories 2000 \
  --snapshots-per-trajectory 20 \
  --grid-size 64 \
  --burn-in-steps 2500 \
  --save-interval 30 \
  --chunk-size 1000 \
  --sim-batch-size 500 \
  --solver-substeps 1 \
  --save-previews \
  --preview-every-chunks 1 \
  --save-sequence-previews \
  --sequence-preview-count 16 \
  --min-image-std 0.025 \
  --min-image-range 0.15 \
  --num-threads 12 \
  --device auto
```

Файлы будут сохранены как:

```text
data/grayscott_64/grayscott_chunk_000.npz
data/grayscott_64/grayscott_chunk_001.npz
...
data/grayscott_64/preview_chunk_000.png
data/grayscott_64/preview_chunk_001.png
...
data/grayscott_64/sequence_preview_batch_000.png
data/grayscott_64/sequence_preview_chunk_000.png
...
data/grayscott_64/manifest.json
```

Каждый chunk содержит:

- `images`: shape `[N, C, H, W]`
- `trajectory_id`
- `snapshot_index`
- `step`
- `F`
- `k`
- `regime_id`
- `split`: `train`, `val`, `test`

Split делается по `trajectory_id`, а не по отдельным кадрам. Это важно, чтобы соседние snapshots
одной траектории не попадали одновременно в train и test.

## Higher-quality preset

Для лучшего разрешения и более аккуратного explicit Euler решения используйте `128x128` и
`solver_substeps=2`. Это не меняет сохраненные timestep labels, но каждый solver step считается
двумя внутренними шагами размера `dt / 2`.

```bash
python scripts/generate_grayscott.py \
  --output-dir data/grayscott_128 \
  --total-images 10000 \
  --num-trajectories 500 \
  --max-trajectories 2000 \
  --snapshots-per-trajectory 20 \
  --grid-size 128 \
  --burn-in-steps 3000 \
  --save-interval 25 \
  --chunk-size 1000 \
  --sim-batch-size 250 \
  --solver-substeps 2 \
  --save-previews \
  --save-sequence-previews \
  --sequence-preview-count 16 \
  --min-image-std 0.025 \
  --min-image-range 0.15 \
  --device auto
```

Colab notebook сейчас настроен именно на этот higher-quality preset.

## Colab

Откройте notebook:

```text
notebooks/generate_grayscott_colab.ipynb
```

Notebook уже указывает на этот repo:

```python
REPO_URL = "https://github.com/SadreevAmir/DL_final_project.git"
```

Если repository private, Colab не сможет клонировать его обычной HTTPS-командой и выдаст ошибку
`could not read Username for 'https://github.com'`. В этом случае:

1. Создайте GitHub personal access token с доступом к repo.
2. В Colab откройте Secrets.
3. Добавьте secret с именем `GITHUB_TOKEN`.
4. Перезапустите clone cell.

Если сделать repository public, token не нужен. Notebook клонирует репозиторий, запускает генерацию
и скачивает каждый `.npz` chunk после сохранения.

## Оценка времени

Грубая оценка для `10_000` хороших картинок:

- Colab A100, `sim_batch_size=500`: обычно `5-20` минут.
- Colab T4/L4: обычно `15-45` минут.
- CPU 12 cores с `--num-threads 12`: может быть `1-3` часа, потому что код векторизован, но PyTorch CPU здесь заметно медленнее GPU.

Для higher-quality `128x128`, `solver_substeps=2`, `sim_batch_size=250`:

- Colab A100: примерно `25-70` минут.
- T4/L4: примерно `1-3` часа.
- CPU не рекомендуется.

Если preview показывает пустые/однородные квадраты, это trajectories, которые пришли к почти
homogeneous fixed point. По умолчанию они отбрасываются фильтром:

```text
min_image_std = 0.025
min_image_range = 0.15
```

Если фильтр слишком строгий и генерация не добирает `total_images`, увеличьте `--max-trajectories`
или ослабьте thresholds.

## Качество и корректность

Получающиеся структурные паттерны являются корректными численными состояниями Gray-Scott model
для выбранной дискретизации, параметров `F, k` и periodic boundary conditions. Почти однородные
поля тоже являются физически допустимым fixed-point outcome, но для diffusion-prior датасета они
обычно вредны, поэтому фильтруются.

Этот генератор не является high-precision PDE benchmark. Это controlled synthetic dataset для ML:
finite-difference Laplacian, explicit Euler, bounded concentrations. Для строгой numerical analysis
нужно делать convergence study по grid size и timestep. Для проекта по reconstruction/diffusion
этого уровня достаточно, особенно с `128x128` и `solver_substeps=2`.

Если нужно быстрее для первого smoke test:

```bash
python scripts/generate_grayscott.py \
  --output-dir data/test \
  --total-images 1000 \
  --num-trajectories 50 \
  --snapshots-per-trajectory 20 \
  --burn-in-steps 1000 \
  --sim-batch-size 50
```

## Насколько это "аттрактор"

Практически это snapshots из long-time regime: начальные случайные blobs не сохраняются, система
сначала прогоняется `burn_in_steps`, и только затем кадры попадают в датасет.

Строго математически мы не доказываем, что samples идеально распределены по инвариантной мере
аттрактора. Для учебного проекта это корректная и честная формулировка:

```text
we sample Gray-Scott states after burn-in as an approximation of the system's long-time attractor distribution
```

Если хотите более однородный датасет, используйте `--param-mode fixed`. Если хотите больше визуального
разнообразия, оставьте default `--param-mode mixed`.
