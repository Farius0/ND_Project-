# ==================================================
# ================  MODULE: blur  ==================
# ==================================================
from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Union, overload

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torchvision.transforms.functional import gaussian_blur as torch_gaussian_blur

# Public API
__all__ = ["apply_spatial_blur", "generate_gaussian_kernel_nd"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Framework = Literal["numpy", "torch"]

def _ensure_odd(k: int) -> int:
    """
    Ensure that an integer is odd.

    Parameters
    ----------
    k : int
        Input integer.

    Returns
    -------
    int
        The same value if k is already odd; otherwise, k + 1.
    """
    return k if (k % 2 == 1) else (k + 1)


def _as_sequence(x: Union[int, float, Sequence[Union[int, float]]], ndim: int) -> Tuple[float, ...]:
    """
    Broadcast a scalar or sequence to a float-valued sequence of length `ndim`.

    Parameters
    ----------
    x : int, float, or sequence of numbers
        Scalar or list/tuple to broadcast.
    ndim : int
        Target length of the output sequence.

    Returns
    -------
    Tuple[float, ...]
        Tuple of floats with length equal to `ndim`.
    """
    if isinstance(x, (int, float)):
        return tuple(float(x) for _ in range(ndim))
    if isinstance(x, Sequence):
        if len(x) == 0:
            raise ValueError("Empty sequence is not allowed for sigma/size.")
        if len(x) == ndim:
            return tuple(float(v) for v in x)
        if len(x) == 1:
            return tuple(float(x[0]) for _ in range(ndim))
        raise ValueError(f"Length mismatch: expected {ndim}, got {len(x)}.")
    raise TypeError("Value must be a number or a sequence of numbers.")


def generate_gaussian_kernel_nd(
    size: Union[int, Sequence[int]],
    sigma: Union[float, Sequence[float]],
    framework: Framework = "numpy",
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[Union[np.dtype, torch.dtype]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generate a separable ND Gaussian kernel.

    Parameters
    ----------
    size : int or sequence of int
        Kernel size per dimension. Each entry will be forced to be odd.
    sigma : float or sequence of float
        Standard deviation per dimension (same length as `size`, or scalar).
    framework : {'numpy','torch'}, default 'numpy'
        Format of the returned kernel.
    device : torch device or str, default 'cpu'
        Only used if framework='torch'.
    dtype : numpy/torch dtype, optional
        If None: np.float32 for NumPy, torch.float32 for Torch.

    Returns
    -------
    kernel : np.ndarray or torch.Tensor
        ND Gaussian kernel normalized to sum=1.
    """
    # Normalize size and sigma as sequences of the same length
    if isinstance(size, int):
        size_seq = (size,)
    else:
        size_seq = tuple(int(s) for s in size)

    size_seq = tuple(_ensure_odd(int(abs(s))) for s in size_seq)  # ensure positive odd sizes

    sigma_seq = _as_sequence(sigma, len(size_seq))
    if any(s <= 0 for s in sigma_seq):
        raise ValueError("All sigma values must be positive.")

    # Build ND coordinate grid in NumPy
    axes = [np.linspace(-(s // 2), s // 2, s, dtype=np.float64) for s in size_seq]
    mesh = np.meshgrid(*axes, indexing="ij")

    kernel_np = np.ones_like(mesh[0], dtype=np.float64)
    for g, s in zip(mesh, sigma_seq):
        kernel_np *= np.exp(-(g ** 2) / (2.0 * (s ** 2)))

    # Normalize
    kernel_np /= kernel_np.sum(dtype=np.float64)

    if framework == "numpy":
        out_dtype = np.float32 if dtype is None else dtype
        return kernel_np.astype(out_dtype, copy=False)

    # Torch
    out_dtype_t = torch.float32 if dtype is None else dtype
    dev = torch.device(device)
    return torch.as_tensor(kernel_np, dtype=out_dtype_t, device=dev)

@overload
def apply_spatial_blur(
    image: np.ndarray,
    sigma: Union[float, Sequence[float]],
    channel_axis: Optional[int] = ...,
    framework: Literal["numpy"] = ...,
    return_kernel: Literal[False] = ...,
) -> np.ndarray: ...
@overload
def apply_spatial_blur(
    image: np.ndarray,
    sigma: Union[float, Sequence[float]],
    channel_axis: Optional[int] = ...,
    framework: Literal["numpy"] = ...,
    return_kernel: Literal[True] = ...,
) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def apply_spatial_blur(
    image: torch.Tensor,
    sigma: Union[float, Sequence[float]],
    channel_axis: Optional[int] = ...,
    framework: Literal["torch"] = ...,
    return_kernel: Literal[False] = ...,
) -> torch.Tensor: ...
@overload
def apply_spatial_blur(
    image: torch.Tensor,
    sigma: Union[float, Sequence[float]],
    channel_axis: Optional[int] = ...,
    framework: Literal["torch"] = ...,
    return_kernel: Literal[True] = ...,
) -> Tuple[torch.Tensor, torch.Tensor]: ...


def apply_spatial_blur(
    image: ArrayLike,
    sigma: Union[float, Sequence[float]]=1.0,
    channel_axis: Optional[int] = None,
    framework: Framework = "numpy",
    return_kernel: bool = False,
) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
    """
    Apply Gaussian blur with NumPy (ND) or Torch (2D via torchvision).

    Parameters
    ----------
    image : ndarray or Tensor
        Input image. NumPy: any ND shape. Torch: (H,W), (C,H,W), or (B,C,H,W).
    sigma : float or sequence of float
        Standard deviation(s) for the Gaussian. Must be > 0.
    channel_axis : int or None, optional
        For NumPy only: axis index for channels; if provided, no blur along it.
    framework : {'numpy','torch'}, default 'numpy'
        Backend selector. Must match the input type.
    return_kernel : bool, default False
        If True, also return the ND kernel (NumPy) or the 2D kernel (Torch).

    Returns
    -------
    blurred : same type as `image`
        Blurred image.
    kernel : same backend as `image` (optional)
        If `return_kernel=True`: ND kernel (NumPy) or 2D kernel (Torch).

    Notes
    -----
    - NumPy path is fully ND-ready and respects channel_axis by setting sigma=0 on that axis.
    - Torch path relies on torchvision 2D gaussian_blur and supports only 2D images with optional (C) and (B) axes.
    """
    # --- Validate framework & types ---
    if framework not in ("numpy", "torch"):
        raise ValueError("framework must be 'torch' or 'numpy'.")

    if framework == "torch":
        if not isinstance(image, torch.Tensor):
            raise TypeError("Torch backend expects a torch.Tensor.")

        if image.ndim not in (2, 3, 4):
            raise ValueError(
                "Torch gaussian_blur supports only 2D spatial tensors with shapes "
                "(H,W), (C,H,W) or (B,C,H,W)."
            )
    else:
        if not isinstance(image, np.ndarray):
            raise TypeError("NumPy backend expects a np.ndarray.")

    # --- Validate sigma ---
    if isinstance(sigma, (int, float)):
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")
    else:
        if any(float(s) <= 0 for s in sigma):
            raise ValueError("All sigma values must be > 0.")

    # === Torch backend (2D only) ===
    if framework == "torch":
        # Prepare kernel_size (height,width)
        if isinstance(sigma, (int, float)):
            k = _ensure_odd(int(np.floor(2.0 * float(sigma) + 1.0)))
            kernel_size = [k, k]
            sigma_used: Union[float, Tuple[float, float]] = float(sigma)
        else:
            sig2 = _as_sequence(sigma, 2)  # (sy, sx) expected by torchvision
            ky = _ensure_odd(int(np.floor(2.0 * sig2[0] + 1.0)))
            kx = _ensure_odd(int(np.floor(2.0 * sig2[1] + 1.0)))
            kernel_size = [ky, kx]
            sigma_used = (sig2[0], sig2[1])

        # Apply blur
        with torch.no_grad():
            blurred = torch_gaussian_blur(image, kernel_size=kernel_size, sigma=sigma_used)

        if not return_kernel:
            return blurred

        # Build a 2D kernel (NumPy → Torch) for information/inspection
        ker_2d = generate_gaussian_kernel_nd(
            size=kernel_size,
            sigma=sigma_used if isinstance(sigma_used, Sequence) else (float(sigma_used), float(sigma_used)),
            framework="torch",
            device=image.device,
            dtype=blurred.dtype,
        )
        return blurred, ker_2d

    # === NumPy backend (fully ND) ===
    if channel_axis is not None:
        if channel_axis < 0:
            channel_axis = image.ndim + channel_axis
        if not (0 <= channel_axis < image.ndim):
            raise ValueError("Invalid channel_axis for the given image shape.")

        sigma_full: Tuple[float, ...] = []
        # Broadcast user sigma to image.ndim
        sigma_bcast = _as_sequence(sigma, image.ndim)
        for ax in range(image.ndim):
            if ax == channel_axis:
                sigma_full.append(0.0)  # do not blur across channels
            else:
                sigma_full.append(sigma_bcast[ax])
        sigma_full = tuple(sigma_full)
    else:
        sigma_full = _as_sequence(sigma, image.ndim)

    blurred_np = gaussian_filter(image, sigma=sigma_full)

    if not return_kernel:
        return blurred_np

    # Build an ND kernel aligned to image.ndim:
    sizes: Tuple[int, ...] = []
    for ax, s in enumerate(sigma_full):
        if channel_axis is not None and ax == channel_axis:
            sizes.append(1)
        else:
            sizes.append(_ensure_odd(int(np.floor(2.0 * s + 1.0))))
    sizes_t = tuple(int(v) for v in sizes)

    kernel_nd = generate_gaussian_kernel_nd(
        size=sizes_t,
        sigma=sigma_full,
        framework="numpy",
        dtype=blurred_np.dtype,
    )
    return blurred_np, kernel_nd
