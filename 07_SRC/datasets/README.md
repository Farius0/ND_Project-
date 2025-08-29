# Datasets

This directory provides **ND-ready datasets** to generate `input/truth` pairs (degradations), 
load **segmentation labels**, or organize **folder-based classification** — 
with full **NumPy/Torch compatibility**, **axis tagging**, and **synchronized transforms**.

## Files

- `operator_dataset.py`  
  Generic datasets built on `OperatorCore` and `PreprocessorND`.  
  - **Classes**:
    - `BaseOperatorDataset`: common skeleton (preprocessing, resize, synchronized transforms, tagging).  
    - `OperatorDataset`: classical degradations via `Operator` (`noise`, `blur`, `inpaint`).  
    - `DeepOperatorDataset`: degradations via `DeepOperator` (“deep inverse” operators).  
  - **Operators available**: `"noise"`, `"blur"`, `"paint"` (inpainting), `"identity"`, `"segmentation"`, `"classification"`.
  - **Transforms**: built via `TransformerConfig` (with separate image and label configs for segmentation); **synchronized seeds** ensure identical transforms on image and label.  
  - **Resize**: bilinear for images, nearest for labels (if `size` is not None).  
  - **Layouts & backend**:
    - Controlled via `LayoutConfig` (`layout_name`, `layout_framework`, `layout_ensured_name`).  
    - Typical usage: `layout_name="HWC"` and `layout_ensured_name="NCHW"` when training with Torch.  
  - **Return modes**: `to_return ∈ {"untransformed", "transformed", "both"}`; optional `return_param` (transform parameters are preserved with `safe_collate`).  
  - **Classification**: if `operator="classification"`, subfolders in `images_dir` define the classes; no `images_files` expected.  
  - **Utilities**:
    - `build_dataset(...)`: builds a coherent `OperatorDataset` with transforms (and `transform_2` for labels in segmentation).  
    - `safe_collate(batch)`: preserves parameter dicts (`params`, `label_params`, `*_params`) during collate.

## Quick Start

### 1) Degradations (noise/blur/inpaint) with PyTorch (layout HWC → ensured NCHW)
```python
from pathlib import Path
from torch.utils.data import DataLoader
from datasets.operator_dataset import build_dataset, safe_collate

root = Path("path/to/images")
images = ["a.png", "b.png", "c.png"]

dataset = build_dataset(
    dir_path=root,
    images_names=images,
    operator="blur",                 # "noise" | "blur" | "paint" | "identity"
    blur_level=5.0,
    layout_framework="numpy",
    layout_name="HWC",               # input HWC
    layout_ensured_name="NCHW",      # ensured Torch layout for training
    add_batch_dim=True,
    use_transforms=True,
    to_return="both",
    return_param=True,
    size=(256, 256),                 # bilinear resize (images)
    horizontal_flip=0.5, rotation=10
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=safe_collate)
batch = next(iter(loader))
# Example batch keys: "input", "truth", "kernel", "t_input", "t_truth", "params", ...
```

### 2) Segmentation (image + label) with synchronized transforms
```python
from datasets.operator_dataset import OperatorDataset, DatasetConfig

dataset = OperatorDataset(
    images_dir="path/to/images",
    images_files=["a.png", "b.png"],
    labels_dir="path/to/labels",
    labels_files=["a.png", "b.png"],
    dataset_cfg=DatasetConfig(
        operator="segmentation",
        to_return="both",
        return_param=True
    ),
)

sample = dataset[0]
# Expected keys: {"input", "truth", "t_input", "t_truth", "params", "label_params"}
```

### 3) Folder-based Classification
```python
from datasets.operator_dataset import OperatorDataset, DatasetConfig

cls_ds = OperatorDataset(
    images_dir="path/to/dataset_root",    # subfolders = class names
    dataset_cfg=DatasetConfig(operator="classification"),
)
x = cls_ds[0]
# {"input": image_tensor, "truth": class_index}
```

## Practical Notes

- **Recommended layouts**  
  - Input images as NumPy: `layout_name="HWC"`; training with Torch: `layout_ensured_name="NCHW"`.  
  - ND dimensions and axes are handled by `LayoutConfig` / `OperatorCore`; axis tags are propagated during conversion and moves.  
- **Transforms**  
  - Built with `TransformerConfig`; for segmentation, one pipeline for images and one for labels (nearest resize for labels).  
- **Parameter safety**  
  - Always use `safe_collate` to preserve transform parameters (`params`, `label_params`) in batch mode (lists not collated), useful for debugging and traceability.
