# Core Modules

This directory contains **foundational**, framework‑agnostic components used across the project.
They define **layout semantics**, **tagging & tracing**, and **conversion policies** for NumPy/PyTorch.

## Files

- `base_converter.py`  
  Unified, layout‑aware converter with traceable tagging (NumPy ↔ PyTorch).
  - ND‑ready; preserves dtype/device/layout when feasible.
  - Axis‑safe moves; optional integer→float normalization.
  - Tagging with UID, history trace, and operator metadata.

- `operator_core.py`  
  Base class for higher‑level operators (filters, segmenters, etc.) built on `BaseConverter`.
  - One‑shot conversions (`convert_once`, `to_output`).
  - Layout enforcement (`ensure_format`) via tracked axis moves.
  - Utilities: tag copy/reset, safe copy, summaries.

- `tag_registry.py`  
  Central lightweight registry (in‑memory) to attach/read/delete tags from arrays/tensors.
  - Backends: `numpy`, `torch`.
  - Debug log (recent tags), UID helpers, summaries.

- `layout_axes.py`  
  Declarative layout catalog and tools.
  - Format dictionaries (e.g. `HWC`, `NCHW`, `GNCDHW`).
  - `LayoutResolver` to build/parse layout strings and guess from shapes.
  - Helpers to map/validate axes and derive names.

- `config.py`  
  Dataclasses defining operator configs (global, layout, preprocess, diff, resize, etc.).
  - Traceable `TransformerConfig` to build invertible torchvision pipelines.
  - Uniform `update_config(**kwargs)` across all configs.

## Quick Start

```python
import numpy as np
import torch
from core.base_converter import BaseConverter
from core.config import LayoutConfig, GlobalConfig

img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)  # HWC
converter = BaseConverter(layout_cfg=LayoutConfig(layout_name="HWC", layout_framework="numpy"),
                          global_cfg=GlobalConfig(framework="torch", output_format="torch"))

# Convert NumPy → Torch with tagging & preserved layout
tensor = converter.convert_once(img, framework="torch", tag_as="input")

# Back to NumPy (optional squeeze/add dims controlled by config)
back = converter.convert_once(tensor, framework="numpy", tag_as="output")

# Inspect the tag
tag = converter.get_tag_summary(tensor, "torch")
print(tag)
