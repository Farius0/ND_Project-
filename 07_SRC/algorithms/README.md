# Algorithms

This directory contains **image processing algorithms** built on top of the project’s operator core and filters.  
They implement classical diffusion models with ND-support and dual backend (NumPy / PyTorch).

## Files

- `perona_malik.py`  
  Implementation of the **Perona–Malik anisotropic diffusion** (classic and enhanced variants).
  - Built on `OperatorCore` with full tag propagation.
  - Dual-backend (NumPy, Torch); ND-ready (2D and 3D volumes).
  - Variants:
    - **Classic PM**: conductivity function based on `|∇u|`.
    - **Enhanced PM**: conductivity function based on `|∇(Gσ * u)|`.
  - Integrated components:
    - `DiffOperator` (gradients/divergence),
    - `EdgeAwareFilter` (conductivity weighting),
    - `NDConvolver` (Gaussian smoothing),
    - `ImageProcessor` (execution strategy).
  - Convenience wrapper: `pm(img, ...)` with reasonable defaults.

## Quick Start

```python
import numpy as np
from algorithms.perona_malik import pm

# Create a noisy test image
img = np.random.rand(128, 128).astype(np.float32)

# Apply classic Perona–Malik denoising (NumPy backend)
denoised = pm(img, algorithm="pm", framework="numpy", layout_name ="HW", steps=10)

# Apply enhanced PM with Torch backend
denoised_torch = pm(img, algorithm="enhanced", framework="torch", layout_name ="HW", steps=15)

print(denoised.shape, denoised_torch.shape)
