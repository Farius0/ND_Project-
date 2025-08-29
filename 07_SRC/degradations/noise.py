# ==================================================
# ===============  MODULE: noise  ==================
# ==================================================
from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Union, overload

import numpy as np
import torch

__all__ = ["apply_noise"]

ArrayNP = np.ndarray
ArrayTorch = torch.Tensor
ArrayLike = Union[ArrayNP, ArrayTorch]
Framework = Literal["numpy", "torch"]

# ---  Utils ---

def _is_float_dtype_np(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.floating)


def _ensure_float_numpy(arr: ArrayNP) -> ArrayNP:
    """Return a floating array (preserve float dtype, else cast to float32)."""
    return arr if _is_float_dtype_np(arr.dtype) else arr.astype(np.float32, copy=False)


def _ensure_float_torch(t: ArrayTorch) -> ArrayTorch:
    """Return a floating tensor (preserve float dtype, else cast to float32)."""
    return t if t.is_floating_point() else t.to(dtype=torch.float32)

# def _np_broadcastable(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> bool:
#     try:
#         # NumPy >= 1.20
#         _ = np.broadcast_shapes(shape_a, shape_b)
#         return True
#     except Exception:
#         # Fallback manuel si besoin
#         a_rev, b_rev = shape_a[::-1], shape_b[::-1]
#         for i in range(max(len(a_rev), len(b_rev))):
#             da = a_rev[i] if i < len(a_rev) else 1
#             db = b_rev[i] if i < len(b_rev) else 1
#             if not (da == db or da == 1 or db == 1):
#                 return False
#         return True

# def _torch_broadcastable(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> bool:
#     try:
#         # PyTorch >= 2.0
#         _ = torch.broadcast_shapes(shape_a, shape_b)
#         return True
#     except Exception:
#         # Fallback manuel
#         a_rev, b_rev = shape_a[::-1], shape_b[::-1]
#         for i in range(max(len(a_rev), len(b_rev))):
#             da = a_rev[i] if i < len(a_rev) else 1
#             db = b_rev[i] if i < len(b_rev) else 1
#             if not (da == db or da == 1 or db == 1):
#                 return False
#         return True

@overload
def apply_noise(
    image: ArrayNP,
    sigma: Union[float, ArrayNP],
    framework: Literal["numpy"],
    return_noise: Literal[False] = ...,
    seed: Optional[int] = ...,
    value_range: Optional[Tuple[float, float]] = ...,
) -> ArrayNP: ...
@overload
def apply_noise(
    image: ArrayNP,
    sigma: Union[float, ArrayNP],
    framework: Literal["numpy"],
    return_noise: Literal[True],
    seed: Optional[int] = ...,
    value_range: Optional[Tuple[float, float]] = ...,
) -> Tuple[ArrayNP, ArrayNP]: ...
@overload
def apply_noise(
    image: ArrayTorch,
    sigma: Union[float, ArrayTorch],
    framework: Literal["torch"],
    return_noise: Literal[False] = ...,
    seed: Optional[int] = ...,
    value_range: Optional[Tuple[float, float]] = ...,
) -> ArrayTorch: ...
@overload
def apply_noise(
    image: ArrayTorch,
    sigma: Union[float, ArrayTorch],
    framework: Literal["torch"],
    return_noise: Literal[True],
    seed: Optional[int] = ...,
    value_range: Optional[Tuple[float, float]] = ...,
) -> Tuple[ArrayTorch, ArrayTorch]: ...


def apply_noise(
    image: ArrayLike,
    sigma: Union[float, ArrayLike] = 0.2,
    framework: Framework = "numpy",
    return_noise: bool = False,
    seed: Optional[int] = None,
    value_range: Optional[Tuple[float, float]] = None,
) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
    """
    Apply additive Gaussian noise (mean=0) with local RNG.

    Parameters
    ----------
    image : ndarray or Tensor
        Clean input image. Any ND shape. Integer inputs are promoted to float32.
    sigma : float or array/tensor broadcastable to image.shape, default 0.2
        Standard deviation of the Gaussian noise.
    framework : {'numpy','torch'}, default 'numpy'
        Must match the input type.
    return_noise : bool, default False
        If True, also return the generated noise (same backend & shape).
    seed : int, optional
        Seed for reproducibility.
    value_range : (float, float), optional
        If provided, clip the noisy output into [min, max].

    Returns
    -------
    noisy : same backend as `image`
        The noisy image.
    noise : same backend as `image` (optional)
        The generated Gaussian noise.

    Notes
    -----
    - Device/dtype policy:
      * Torch: image stays on its current device and dtype (promoted to float32 if not floating).
      * NumPy: image dtype preserved if floating, else promoted to float32.
    - Sigma can be scalar or broadcastable (per-channel/per-voxel).
    """
    if framework not in ("numpy", "torch"):
        raise ValueError("framework must be 'torch' or 'numpy'.")

    # ===== Torch backend =====
    if framework == "torch":
        if not isinstance(image, torch.Tensor):
            raise TypeError("Torch backend expects a torch.Tensor.")
        # Ensure floating dtype (avoid integer overflow/saturation)
        img_f = _ensure_float_torch(image)

        # Prepare sigma as tensor broadcastable to image
        if isinstance(sigma, (int, float)):
            sigma_t = torch.as_tensor(float(sigma), dtype=img_f.dtype, device=img_f.device)
            # Use normal_ with generator for efficient scalar-std
            if seed is not None:
                torch.manual_seed(int(seed))
                if image.is_cuda:
                    torch.cuda.manual_seed_all(int(seed))
            noise = torch.empty_like(img_f).normal_(mean=0.0, std=float(sigma_t.item()))
        else:
            if not isinstance(sigma, torch.Tensor):
                raise TypeError("For Torch backend, sigma must be float or torch.Tensor.")
            sigma_t = sigma.to(dtype=img_f.dtype, device=img_f.device)
            # Use randn_like then scale by broadcasted sigma
            gen = torch.Generator(device=img_f.device)
            if seed is not None:
                gen.manual_seed(int(seed))
            noise = torch.randn_like(img_f) * sigma_t

        noisy = img_f + noise
        if value_range is not None:
            lo, hi = float(value_range[0]), float(value_range[1])
            noisy = torch.clamp(noisy, min=lo, max=hi)

        return (noisy, noise) if return_noise else noisy

    # ===== NumPy backend =====
    if not isinstance(image, np.ndarray):
        raise TypeError("NumPy backend expects a np.ndarray.")

    img_f = _ensure_float_numpy(image)

    # Local RNG (no global seeding)
    rng = np.random.default_rng(seed)

    if isinstance(sigma, (int, float)):
        sigma_arr = float(sigma)
        noise_np = rng.normal(loc=0.0, scale=sigma_arr, size=img_f.shape).astype(img_f.dtype, copy=False)
    else:
        if isinstance(sigma, np.ndarray):
            sigma_arr = sigma.astype(img_f.dtype, copy=False)
        else:
            raise TypeError("For NumPy backend, sigma must be float or np.ndarray.")
        # Broadcast via multiplication
        noise_np = rng.normal(loc=0.0, scale=1.0, size=img_f.shape).astype(img_f.dtype, copy=False) * sigma_arr

    noisy_np = img_f + noise_np
    if value_range is not None:
        lo, hi = float(value_range[0]), float(value_range[1])
        np.clip(noisy_np, lo, hi, out=noisy_np)

    return (noisy_np, noise_np) if return_noise else noisy_np
