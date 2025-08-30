# Degradations

This directory provides **image degradation operators** that are **dual‑backend** (NumPy / PyTorch) and **ND‑aware** where possible.  
They are typically used to synthesize inputs for restoration, inpainting, or robustness experiments.

## Files

- `noise.py`  
  Additive **Gaussian noise** with local RNG.
  - API: `apply_noise(image, sigma=0.2, framework='numpy'|'torch', return_noise=False, seed=None, value_range=None)`
  - **Sigma** can be a scalar or **broadcastable** array/tensor (per‑voxel/per‑channel noise).
  - **Device/dtype policy**:
    - Torch: input stays on its device; integers are promoted to `float32`.
    - NumPy: floating dtype preserved, else promoted to `float32`.
  - Optional outputs: return both `(noisy, noise)` with `return_noise=True`.

- `blur.py`  
  Spatial **Gaussian blur**.
  - API: `apply_spatial_blur(image, sigma=..., channel_axis=None, framework='numpy'|'torch', return_kernel=False)`
  - **NumPy path**: fully **ND‑ready**; if `channel_axis` is provided, the blur **does not** diffuse across channels (sigma=0 on that axis).
  - **Torch path**: relies on `torchvision.transforms.functional.gaussian_blur` → supports only 2D spatial tensors `(H,W)`, `(C,H,W)` or `(B,C,H,W)`.
  - `generate_gaussian_kernel_nd(size, sigma, framework)` builds normalized ND Gaussian kernels (odd sizes enforced).

- `inpaint.py`  
  Masked **inpainting by noise injection** (mean=0), with three modes.
  - API: `apply_inpaint(image, mask=None, sigma=0.05, framework='torch'|'numpy', threshold=0.4, mode='replace'|'masked_noised'|'grid_noised', seed=None)`
  - **Mask semantics**: `True` = keep original pixel, `False` = candidate for inpainting.
  - If `mask is None`, a random mask is generated with **holes rate** ≈ `threshold`.
  - **Modes**:
    - `replace`: `image*mask + noise*(~mask)` (standard inpaint‑by‑noise).
    - `masked_noised`: apply noise **only** where `mask==True` (keep outside unchanged).
    - `grid_noised`: auto grid mask over the last two spatial dims `(H,W)` then same policy as `replace`.
  - Returns `(inpainted, mask)` with the **same backend** as the input.

## Layouts & dimensions

These operators are **layout‑agnostic**: they operate on raw arrays/tensors.  
Conventions used internally:
- For **inpaint** and grid building, the **last two dimensions** are treated as spatial `(H, W)`.
- For **NumPy blur**, use `channel_axis` to avoid blurring across channels (e.g., `HWC` → `channel_axis=2`).  
- For **Torch blur**, only 2D spatial inputs are supported by `torchvision` (with optional `C` and `B` prefixes), e.g., `(B,C,H,W)`.

## Quick Start

### 1) Gaussian noise (NumPy)
```python
import numpy as np
from degradations.noise import apply_noise

img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)   # HWC uint8
noisy = apply_noise(img, sigma=20.0, framework="numpy", value_range=(0, 255))
print(noisy.dtype, noisy.shape)  # float32, (128,128,3)
```

### 2) Per‑channel noise with broadcasting (Torch)
```python
import torch
from degradations.noise import apply_noise

x = torch.rand(4, 3, 256, 256, device="cpu")  # (B,C,H,W) in [0,1]
sigma = torch.tensor([0.05, 0.10, 0.20], device=x.device).view(1, 3, 1, 1)
noisy, eps = apply_noise(x, sigma=sigma, framework="torch", return_noise=True, seed=123)
```

### 3) Gaussian blur (NumPy, ND‑ready, channel‑aware)
```python
import numpy as np
from degradations.blur import apply_spatial_blur

x = np.random.rand(64, 64, 3).astype(np.float32)  # HWC
y = apply_spatial_blur(x, sigma=(1.2, 2.0, 1.0), channel_axis=2, framework="numpy")
```

### 4) Gaussian blur (Torch 2D)
```python
import torch
from degradations.blur import apply_spatial_blur

x = torch.rand(1, 3, 128, 128)  # (B,C,H,W)
y, k = apply_spatial_blur(x, sigma=(1.5, 1.5), framework="torch", return_kernel=True)
```

### 5) Inpainting by noise (mask generated)
```python
import numpy as np
from degradations.inpaint import apply_inpaint

img = np.random.rand(1, 128, 128).astype(np.float32)  # e.g., (C,H,W) grayscale
inp, m = apply_inpaint(img, framework="numpy", threshold=0.3, mode="replace", sigma=0.1, seed=42)
```

### 6) Inpainting with custom mask (Torch) and "masked_noised" mode
```python
import torch
from degradations.inpaint import apply_inpaint

x = torch.rand(1, 1, 256, 256)             # (B,C,H,W)
mask = torch.zeros_like(x, dtype=torch.bool)
mask[..., 64:192, 64:192] = True           # only center region gets noise
y, used_mask = apply_inpaint(x, mask=mask, sigma=0.05, framework="torch", mode="masked_noised", seed=7)
```

## Tips & gotchas

- **Reproducibility**: pass `seed` to keep noise/inpaint deterministic without touching the global RNG.  
- **Clipping**: for `apply_noise`, use `value_range=(min,max)` to clamp the output (e.g., `(0,1)` or `(0,255)`).  
- **Performance**:
  - Torch noise uses generator‑based `normal_`/`randn_like` on the **current device**.
  - Torch blur delegates to `torchvision` (2D only); for ND kernels use the NumPy path or your own conv.  
- **Channel handling (NumPy blur)**: set `channel_axis` for color images (`HWC` → `channel_axis=2`) to avoid blending across channels.
