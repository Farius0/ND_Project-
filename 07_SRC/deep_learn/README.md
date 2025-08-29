# Deep Learning

This directory provides **deep-learning components** for segmentation and related tasks.  
It includes custom architectures, advanced loss functions, and training utilities.

## Files

- `FEUNet_v2.py`  
  Implementation of the **FEUNet_v2** architecture (Flexible Enhanced U-Net).  
  - **Encoder**: dynamic UNet-style with configurable feature maps and dropout.  
  - **GLCA blocks** (optional): Global & Local Context-Aware attention.  
  - **Fusion blocks** (optional): feature fusion with channel attention.  
  - **Decoder**: flexible UpBlocks with bilinear or transposed upsampling.  
  - **Output heads**: segmentation output (`seg_out`) and optional regularization (`reg_out`).  
  - **Utilities**: weight initialization, forward hooks for feature debugging, padding/cropping to handle arbitrary input sizes.

- `losses.py`  
  Comprehensive loss library for segmentation and weakly-supervised learning.  
  - **Supervised losses**: `CrossEntropyLoss`, `DiceLoss`, `IoULoss`, `LovaszSoftmaxLoss`.  
  - **Partially-supervised losses**: `pCELoss`, `pDLoss`.  
  - **Combined losses**: `ComboLoss` (weighted CE + Dice), `FocalLoss`.  
  - **Advanced losses**: `TverskyLoss`, `FocalTverskyLoss`, `ConsistencyLoss`, `DistillationLoss`.  
  - **Selector**: `LossSelector(name=..., n_classes=..., ...)` automatically builds the requested loss function with proper parameters.

- `utils.py`  
  Utilities for model training and evaluation.  
  - **EarlyStopping**: monitors validation loss, saves checkpoints, and halts training when no improvement occurs.  
  - **plot_training**: plots training/validation curves for loss and accuracy.

## Quick Start

### FEUNet_v2
```python
import torch
from deep_learn.FEUNet_v2 import FEUNet_v2, default_config

# Create model from default YAML config
model = FEUNet_v2(default_config)

# Forward pass with dummy input
x = torch.randn(2, 1, 256, 256)  # (B, C, H, W)
out = model(x)

print(out["seg_out"].shape)   # segmentation output
print(out.get("reg_out", None))
```

### Loss Selector
```python
from deep_learn.losses import LossSelector
import torch

# Example: combined Dice + CrossEntropy with ignore_index=255
loss_fn = LossSelector(name="combo", n_classes=4, ignore_index=255)

logits = torch.randn(2, 4, 128, 128)  # predictions
targets = torch.randint(0, 4, (2, 128, 128))  # labels

loss = loss_fn(logits, targets)
print("Loss:", loss.item())
```

### Early Stopping
```python
from deep_learn.utils import EarlyStopping

early_stopping = EarlyStopping(save_path="best_model.pth", patience=5)

for epoch in range(50):
    val_loss = 0.01 * (50 - epoch)  # fake decreasing loss
    stop = early_stopping(val_loss, model=None, epoch=epoch)
    if stop:
        break
```
