# Kolmogorov Flow 64x64 10k Colab Variant

Отдельный вариант для генерации fluid-like датасета ближе к статье:

```text
forced 2D incompressible Navier-Stokes in vorticity form
periodic domain
Kolmogorov-style forcing
10_000 normalized vorticity snapshots
64x64 resolution
```

Notebook:

```text
generate_kolmogorov_64_10k_colab.ipynb
```

Это не полная реплика NeurIPS SDA paper. В статье использовался Kolmogorov flow / Navier-Stokes
velocity field, а здесь используется более простой pseudo-spectral vorticity solver, чтобы быстро
получить похожие vortex-like states для ML-проекта.

Формат:

```python
images  # [N, 1, 64, 64], float16, normalized vorticity in [-1, 1]
```

Приблизительно:

```python
omega = images * vorticity_scale
```
