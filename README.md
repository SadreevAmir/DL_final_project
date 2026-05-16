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
  Navier-Stokes in vorticity form with Kolmogorov-style forcing.

Готовые Colab variants:

```text
notebooks/grayscott_128_10k/generate_grayscott_128_10k_colab.ipynb
notebooks/kolmogorov_64_10k/generate_kolmogorov_64_10k_colab.ipynb
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
