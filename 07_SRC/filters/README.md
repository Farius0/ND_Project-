# Filters

This directory provides **filtering operators** that enhance or clean images, often used in preprocessing before segmentation or denoising.  
They are **dual-backend (NumPy / PyTorch)** and support **ND inputs** when relevant.

## Files

- `edge_aware_filter.py`  
  Edge-aware conductivity function used in anisotropic diffusion.  
  - Modes:  
    - `'pm'`: g(x) = 1 / sqrt(1 + (x/alpha)^2)  
    - `'exp'`: g(x) = exp(-(x/alpha)^2)  
  - Dual backend, ND-ready via `ImageProcessor`.  
  - Typically integrated into Perona–Malik diffusion as the edge-stopping function.

- `perona_enhancing.py`  
  Lightweight façade around `PeronaMalikDenoiser`.  
  - Fixed algorithm: classic Perona–Malik (no enhanced variant).  
  - Parameters: `alpha`, `dt`, `steps`, `clip`, etc.  
  - Uses sigma=1.0 for Gaussian convolution inside operators.  
  - Provides a simple callable interface for quick denoising.

- `artifact_cleaning.py`  
  ND-compatible artifact cleaner for structured artifacts (e.g., horizontal stripes).  
  - Based on feature extraction (Sobel gradients, edge maps, median filter).  
  - Detects stripe bands and inpaints them by vertical interpolation.  
  - Works on 2D slices or 3D volumes (`layout_name="DHW"`).  
  - Uses `ImageProcessor` for slice-by-slice cleaning.

## Layouts & dimensions

- `EdgeAwareFilter`: layout passed via `LayoutConfig`; works per-channel.  
- `PeronaEnhancer`: by default `layout_name="DHW"` for volumes, but accepts any `LayoutConfig`.  
- `ArtifactCleanerND`: requires `HW` (2D images) or `DHW` (3D volumes).  

## Quick Start

### 1) Edge-aware filter
```python
import torch
from filters.edge_aware_filter import EdgeAwareFilter
from core.config import FilterConfig, LayoutConfig, GlobalConfig, ImageProcessorConfig

flt = EdgeAwareFilter(
    filter_cfg=FilterConfig(mode="pm", alpha=0.1),
    layout_cfg=LayoutConfig(layout_name="HW", layout_framework="torch"),
    global_cfg=GlobalConfig(framework="torch", output_format="torch"),
    img_process_cfg=ImageProcessorConfig(processor_strategy="torch"),
)

x = torch.rand(1, 128, 128)
y = flt(x)
```

### 2) Perona Enhancer
```python
import numpy as np
from filters.perona_enhancing import PeronaEnhancer

img = np.random.rand(128, 128).astype(np.float32)
enhancer = PeronaEnhancer(framework="numpy", layout_name="HW", layout_ensured_name="HW")
out = enhancer(img)
```

### 3) Artifact Cleaner (3D volume)
```python
import numpy as np
from filters.artifact_cleaning import ArtifactCleanerND

volume = np.random.rand(32, 128, 128).astype(np.float32)  # DHW
cleaner = ArtifactCleanerND(framework="numpy", layout_name="DHW")
cleaned = cleaner(volume, axis=1)
print(cleaned.shape)
```
