# Kolmogorov Flow 64x64 10k Colab Variant

Отдельный вариант для генерации fluid-like датасета ближе к статье:

```text
forced 2D incompressible Navier-Stokes in vorticity form
periodic domain
Kolmogorov-style forcing
10_000 normalized velocity-field snapshots
64x64 resolution
FP32 storage
```

Notebook:

```text
generate_kolmogorov_64_10k_colab.ipynb
```

Это не полная реплика NeurIPS SDA paper. В статье использовался Kolmogorov flow / Navier-Stokes
velocity field. Здесь динамика тоже интегрируется в vorticity form, но в `.npz` сохраняется
восстановленное velocity field `(u_x, u_y)`, а previews строятся по vorticity.

Формат:

```python
images  # [N, 2, 64, 64], float32, normalized velocity (u_x, u_y) in [-1, 1]
```

Приблизительно:

```python
velocity = images * velocity_scale
```

Current preset is tuned for more paper-like, higher-contrast states than the first quick version:

```text
dt = 0.01
viscosity = 3e-4
drag = 0.025
forcing_amp = 0.55
burn_in_steps = 5000
dtype = float32
output_field = velocity
```

Preview images use robust adaptive contrast based on the 99th percentile and show derived vorticity,
matching the visualization convention used in the paper. The saved arrays are fixed-scale normalized
velocity values in `[-1, 1]`.
