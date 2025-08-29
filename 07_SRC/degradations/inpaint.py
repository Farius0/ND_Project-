# ==================================================
# ==============  MODULE: inpaint  =================
# ==================================================
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch

from degradations.noise import apply_noise

__all__ = ["apply_inpaint"]

ArrayNP = np.ndarray
ArrayTorch = torch.Tensor
ArrayLike = Union[ArrayNP, ArrayTorch]
Framework = Literal["numpy", "torch"]
Mode = Literal["replace", "masked_noised", "grid_noised"]


def _ensure_bool_mask(mask: ArrayLike, framework: Framework) -> ArrayLike:
    """Ensure a boolean mask with the same backend."""
    if framework == "torch":
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Torch backend expects a torch.Tensor mask.")
        return mask.to(dtype=torch.bool)
    else:
        if not isinstance(mask, np.ndarray):
            raise TypeError("NumPy backend expects a np.ndarray mask.")
        return mask.astype(bool, copy=False)
    

# ====[ Generate Binary Mask (Random) ]====    
def _generate_mask(
    image: ArrayLike,
    threshold: float,
    framework: Framework,
    seed: Optional[int] = None,
) -> ArrayLike:
    """
    Generate a random boolean mask with P(False) ~= threshold (i.e., 'holes' rate ~ threshold).
    True => keep original pixel; False => candidate for inpainting.
    """
    if not (0.0 <= threshold < 1.0):
        raise ValueError("threshold must be in [0, 1).")

    if framework == "torch":
        if not isinstance(image, torch.Tensor):
            raise TypeError("Torch backend expects a torch.Tensor image.")
        if seed is not None:
            torch.manual_seed(int(seed))
            if image.is_cuda:
                torch.cuda.manual_seed_all(int(seed))
        return torch.rand_like(image, dtype=torch.float32).gt(threshold)
    else:
        if not isinstance(image, np.ndarray):
            raise TypeError("NumPy backend expects a np.ndarray image.")
        rng = np.random.default_rng(seed)
        return rng.random(size=image.shape) > float(threshold)
    
# ====[ Replace Noise on Masked Region ]====
def _mode_replace(
    image: ArrayLike,
    mask: ArrayLike,
    sigma: float,
    framework: Framework,
    seed: Optional[int] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Replace only the masked-out region (mask == False) by noisy samples, keep mask==True unchanged.
    """
    mask = _ensure_bool_mask(mask, framework)
    noisy = apply_noise(image, sigma=sigma, framework=framework, seed=seed)
    if framework == "torch":
        with torch.no_grad():
            out = image * mask + noisy * (~mask)
    else:
        out = image * mask + noisy * np.logical_not(mask)
    return out, mask


# ====[ Apply Noise Only to Masked Region ]====
def _mode_masked_noised(
    image: ArrayLike,
    mask: ArrayLike,
    sigma: float,
    framework: Framework,
    seed: Optional[int] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Apply noise only within the masked-in region (mask == True); outside remains unchanged.
    """
    mask = _ensure_bool_mask(mask, framework)
    if framework == "torch":
        with torch.no_grad():
            base = image.clone()
            noisy_region = apply_noise(image, sigma=sigma, framework=framework, seed=seed)
            out = torch.where(mask, noisy_region, base)
    else:
        base = image.copy()
        noisy_region = apply_noise(image, sigma=sigma, framework=framework, seed=seed)
        out = np.where(mask, noisy_region, base)
    return out, mask

# ====[ Grid Masking + Noise ]====

def _build_grid_mask_like(
    image: ArrayLike,
    framework: Framework,
    step_h: int,
    step_w: int,
) -> ArrayLike:
    """
    Build a boolean grid mask on the last two spatial dims (H, W).
    True => keep; False => drop on grid lines.
    """
    if step_h < 1 or step_w < 1:
        raise ValueError("Grid steps must be >= 1.")

    if framework == "torch":
        if not isinstance(image, torch.Tensor):
            raise TypeError("Torch backend expects a torch.Tensor image.")
        mask = torch.ones_like(image, dtype=torch.bool)
        # Set False on grid rows/cols (broadcast on prefix)
        mask[..., 0::step_h, :] = False
        mask[..., :, 0::step_w] = False
        return mask
    else:
        if not isinstance(image, np.ndarray):
            raise TypeError("NumPy backend expects a np.ndarray image.")
        mask = np.ones_like(image, dtype=bool)
        # Always target the last two dims as (H, W)
        slicer_rows = (slice(None),) * (image.ndim - 2) + (slice(0, None, step_h), slice(None))
        slicer_cols = (slice(None),) * (image.ndim - 2) + (slice(None), slice(0, None, step_w))
        mask[slicer_rows] = False
        mask[slicer_cols] = False
        return mask
    
def _mode_grid_noised(
    image: ArrayLike,
    sigma: float,
    framework: Framework,
    seed: Optional[int] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Build a grid mask on (H,W) then apply noise to the retained region (mask==True) or, by policy here,
    keep pixels on the False grid lines as 'holes' to be replaced by noise (like replace).
    """
    # Steps proportional to sqrt of spatial dims, but never zero
    if framework == "torch":
        if not isinstance(image, torch.Tensor):
            raise TypeError("Torch backend expects a torch.Tensor image.")
        H = int(image.shape[-2])
        W = int(image.shape[-1])
    else:
        if not isinstance(image, np.ndarray):
            raise TypeError("NumPy backend expects a np.ndarray image.")
        H = int(image.shape[-2])
        W = int(image.shape[-1])

    step_h = max(1, int(np.sqrt(max(1, H))))
    step_w = max(1, int(np.sqrt(max(1, W))))

    mask = _build_grid_mask_like(image, framework, step_h=step_h, step_w=step_w)
    # Same policy as "replace": keep True, replace False with noise
    return _mode_replace(image, mask, sigma, framework, seed=seed)
    


def apply_inpaint(
    image: ArrayLike,
    mask: Optional[ArrayLike] = None,
    sigma: float = 0.05,
    framework: Framework = "torch",
    threshold: float = 0.4,
    mode: Mode = "replace",
    seed: Optional[int] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Inpaint masked regions by injecting Gaussian noise (mean=0), RNG-local and device-safe.

    Parameters
    ----------
    image : np.ndarray or torch.Tensor
        Input image. Any ND shape; the last two dims are treated as spatial (H,W) when needed.
    mask : same backend as image, optional
        Boolean mask. True = keep original pixel; False = replace/noise depending on mode.
        If None, a random mask is generated with P(False) ~= threshold.
    sigma : float, default 0.05
        Noise standard deviation (scalar). For per-channel control, applique d'abord un sigma broadcastable via apply_noise.
    framework : {'torch','numpy'}, default 'torch'
        Must match the input type.
    threshold : float, default 0.4
        If mask is None, controls the 'holes' rate. Must be in [0,1).
    mode : {'replace','masked_noised','grid_noised'}, default 'replace'
        - 'replace': image*mask + noise*(~mask)
        - 'masked_noised': noisy(image) only where mask==True
        - 'grid_noised': auto grid mask on (H,W) then same policy as 'replace'
    seed : int, optional
        Seed used in local RNGs (no global RNG side-effects).

    Returns
    -------
    inpainted : same backend as image
        Degraded image.
    mask : same backend as image
        Boolean mask used.
    """
    if framework not in ("torch", "numpy"):
        raise ValueError("framework must be 'torch' or 'numpy'.")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")

    if framework == "torch":
        if not isinstance(image, torch.Tensor):
            raise TypeError("Torch backend expects a torch.Tensor.")
    else:
        if not isinstance(image, np.ndarray):
            raise TypeError("NumPy backend expects a np.ndarray.")

    # Generate or validate mask
    if mask is None:
        mask = _generate_mask(image, threshold=threshold, framework=framework, seed=seed)
    else:
        mask = _ensure_bool_mask(mask, framework)

    if mode == "replace":
        return _mode_replace(image, mask, sigma, framework, seed=seed)
    if mode == "masked_noised":
        return _mode_masked_noised(image, mask, sigma, framework, seed=seed)
    if mode == "grid_noised":
        return _mode_grid_noised(image, sigma, framework, seed=seed)

    raise ValueError("Unsupported mode. Choose from 'replace', 'masked_noised', 'grid_noised'.")
