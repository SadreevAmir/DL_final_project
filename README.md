# Gray-Scott Snapshot Dataset

Мини-репозиторий для генерации датасета из `10_000` Gray-Scott reaction-diffusion snapshots.
Идея: симулировать динамическую PDE, пропустить начальный transient через `burn_in_steps`, а затем
сохранять отдельные состояния как картинки из long-time regime.

Это не полноценная trajectory data assimilation. Это датасет для более простой задачи:

```text
single PDE state reconstruction from sparse/noisy observations
```

## Что генерируется

Gray-Scott system:

```text
u_t = D_u Laplacian(u) - u v^2 + F(1 - u)
v_t = D_v Laplacian(v) + u v^2 - (F + k) v
```

По умолчанию:

- grid: `64x64`
- channels: один канал `v`
- trajectories: `500`
- snapshots per trajectory: `20`
- total images: `10_000`
- burn-in: `2500` explicit Euler steps
- save interval: `30` steps
- chunk size: `1000` images
- preview: PNG-сетка samples после каждого chunk
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
  --snapshots-per-trajectory 20 \
  --grid-size 64 \
  --burn-in-steps 2500 \
  --save-interval 30 \
  --chunk-size 1000 \
  --sim-batch-size 500 \
  --save-previews \
  --preview-every-chunks 1 \
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

## Colab

Откройте notebook:

```text
notebooks/generate_grayscott_colab.ipynb
```

Перед запуском замените:

```python
REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"
```

на URL вашего GitHub-репозитория после push. Notebook клонирует репозиторий, запускает генерацию
и скачивает каждый `.npz` chunk после сохранения.

## Оценка времени

Грубая оценка для `10_000` картинок `64x64`, `500` trajectories, `2500` burn-in steps:

- Colab A100, `sim_batch_size=500`: обычно `5-20` минут.
- Colab T4/L4: обычно `15-45` минут.
- CPU 12 cores с `--num-threads 12`: может быть `1-3` часа, потому что код векторизован, но PyTorch CPU здесь заметно медленнее GPU.

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
