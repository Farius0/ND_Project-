# Utils

This directory contains **stateless utility modules**: decorators, logging, label tools, 
N‑D math, visualization, emojis, and segmentation support functions.  
They are lightweight, independent, and do not import from project internals (except tags/layout if needed).

## Files

- `decorators.py`  
  Timer and debugging decorators (`TimerManager`, `timer`, `safe_timer_and_debug`).

- `emojis.py`  
  Emoji utilities (`enable_emojis`, `emojize`, `deemojize`, `emojis_dict`) for log/UI enrichment.

- `labels_tools.py`  
  Label post‑processing utilities:  
  - `scribble_labels`, `scribble_labels_2`: enforce vertical consistency and remove small blobs.  
  - `return_region_cropped`: crop segmentation with heuristics.  
  - `generate_minimal_scribble`: create sparse scribbles from dense masks.

- `logger.py`  
  Logging utilities with rotating file handlers:  
  - `get_logger`, `get_error_logger`, `get_debug_logger`.  
  - Default log directory: `logs/`.

- `nd_math.py`  
  N‑D finite differences and operators: `diff_forward_roll`, `diff_backward_pad`, `gradient_nd`, `divergence_nd`, `laplacian_nd`.

- `nd_tools.py`  
  Visualization and plotting helpers:  
  - `colormap_picker`, `show_plane`, `plot_hist`, `plot_images_and_hists`, `display_features`, `save_features_grid`.

- `segmentations_tools.py`  
  Segmentation utilities:  
  - Slice extraction (`slice_sample`),  
  - Metrics (`evaluate_segmentation_nd`),  
  - Contour tools (`return_thick_contours`),  
  - Robust masks (`mask_quantile`, `mask_iqr`, `mask_zscore`, `mask_mad`, `mask_minmax`),  
  - Thickness estimation (`estimate_thickness_from_all_points`, `aggregate_stack_summaries`, `visualize_thickness_profiles`).

## Quick Start

### 1) Timer decorator
```python
from utils.decorators import timer

@timer(return_result=True)
def slow_fn(n=10**6):
    return sum(range(n))

print(slow_fn(1000))
```

### 2) Scribble labels cleaning
```python
import numpy as np
from utils.labels_tools import scribble_labels

raw = np.random.randint(0, 4, (128,128))
clean = scribble_labels(raw, num_layers=3)
print(np.unique(clean))
```

### 3) Segmentation evaluation
```python
import numpy as np
from utils.segmentations_tools import evaluate_segmentation_nd

ref = np.array([[0,1,1],[0,1,2]])
pred = np.array([[0,1,2],[0,1,2]])
metrics = evaluate_segmentation_nd(ref, pred)
print(metrics["macro"]["iou"], metrics["macro"]["dice"])
```
