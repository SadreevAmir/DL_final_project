# Kolmogorov Flow 64x64 10k Colab Variant

Отдельный вариант для генерации fluid-like датасета ближе к статье:

```text
forced 2D incompressible Navier-Stokes in vorticity form
periodic domain
Kolmogorov-style forcing
10_000 normalized vorticity snapshots
64x64 resolution
FP32 storage
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
images  # [N, 1, 64, 64], float32, normalized vorticity in [-1, 1]
```

Приблизительно:

```python
omega = images * vorticity_scale
```

Current preset is tuned for more paper-like, higher-contrast states than the first quick version:

```text
dt = 0.01
viscosity = 3e-4
drag = 0.025
forcing_amp = 0.55
burn_in_steps = 5000
dtype = float32
```

Preview images use robust adaptive contrast based on the 99th percentile. The saved arrays are still
fixed-scale normalized vorticity values in `[-1, 1]`.
