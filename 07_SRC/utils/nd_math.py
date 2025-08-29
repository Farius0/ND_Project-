# ==================================================
# ===============  MODULE: nd_math  ================
# ==================================================
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

ArrayLikeNP = np.ndarray
TensorT = torch.Tensor
ArrayLike = Union[ArrayLikeNP, TensorT]
SpacingLike = Union[float, Sequence[float]]

# =========================================================
# Helpers
# =========================================================

# only for debug
# def _as_numpy(a: ArrayLike) -> ArrayLikeNP:
#     """
#     Convert input to a NumPy array if it is not already one.

#     Parameters
#     ----------
#     a : ArrayLike
#         Input data (e.g., list, NumPy array, Torch tensor).

#     Returns
#     -------
#     ArrayLikeNP
#         NumPy array representation of the input.
#     """
#     return a.detach().cpu().numpy() if torch.is_tensor(a) else a

# def _make_slices(ndim: int, axis: int, idx: Union[int, slice]) -> Tuple[slice, ...]:
#     """
#     Create a tuple of slice objects to index along a specific axis in an N-dimensional array.

#     Parameters
#     ----------
#     ndim : int
#         Number of dimensions of the target array.
#     axis : int
#         Axis along which to apply the index or slice.
#     idx : int or slice
#         Index or slice to apply on the specified axis.

#     Returns
#     -------
#     Tuple[slice, ...]
#         Tuple of slice objects to be used for array indexing.
#     """
#     sl: List[slice] = [slice(None)] * ndim
#     sl[axis] = idx if isinstance(idx, slice) else slice(idx, idx + 1) if isinstance(idx, int) else idx
#     return tuple(sl)

def _normalize_axes_and_spacing(
    u_ndim: int,
    spacing: Optional[SpacingLike],
    channel_axis: Optional[int],
) -> Tuple[List[int], List[float]]:
    """
    Normalize axes and spacing information for ND image processing.

    Parameters
    ----------
    u_ndim : int
        Total number of dimensions in the input array (including channel if present).
    spacing : SpacingLike, optional
        Spacing between pixels/voxels along each axis. Can be a single float,
        a list of floats, or None (default to 1.0 per axis).
    channel_axis : int or None
        Index of the channel axis, if present. This axis will be excluded from computation.

    Returns
    -------
    axes : List[int]
        List of axis indices to compute over (excluding channel axis if specified).
    spacing_per_axis : List[float]
        List of spacing values for each axis in `axes`, with proper broadcasting.
    """
    axes = [i for i in range(u_ndim) if i != channel_axis] if channel_axis is not None else list(range(u_ndim))
    if spacing is None:
        sp = [1.0] * len(axes)
    elif isinstance(spacing, (int, float)):
        sp = [float(spacing)] * len(axes)
    else:
        spacing_seq = list(map(float, spacing))
        if len(spacing_seq) == u_ndim:
            # Map spacing given for all dims → on retire celle du channel si présent
            if channel_axis is not None:
                sp = [spacing_seq[i] for i in range(u_ndim) if i != channel_axis]
            else:
                sp = spacing_seq
        elif len(spacing_seq) == len(axes):
            sp = spacing_seq
        else:
            raise ValueError(
                f"Invalid spacing length: got {len(spacing_seq)}, expected {len(axes)} "
                f"(axes without channel) or {u_ndim} (all dims)."
            )
    return axes, sp

# ============================================================
# Finite differences (forward / backward) with backend support
# ============================================================

# --- Forward difference using roll (vectorized) ---
def diff_forward_roll(
    arr: ArrayLike,
    axis: int,
    spacing: float = 1.0,
) -> ArrayLike:
    """
    Compute the forward finite difference along a specified axis using array roll.

    The last element along the axis is filled using the previous value to preserve
    array shape and avoid wrap-around artifacts. Compatible with NumPy and PyTorch.

    Parameters
    ----------
    arr : ArrayLike
        Input array (NumPy or Torch) of arbitrary dimension.
    axis : int
        Axis along which to compute the forward difference.
    spacing : float, optional
        Grid spacing between elements along the specified axis. Default is 1.0.

    Returns
    -------
    ArrayLike
        Array of the same shape as input, containing the forward differences.
    """

    if torch.is_tensor(arr):
        rolled = torch.roll(arr, shifts=-1, dims=axis)
        diff = (rolled - arr) / spacing
        # last index along axis gets value from -2 slice
        # build slices
        idx_last = [slice(None)] * arr.ndim
        idx_prev = [slice(None)] * arr.ndim
        idx_last[axis] = -1
        idx_prev[axis] = -2
        diff[tuple(idx_last)] = diff[tuple(idx_prev)]
        return diff
    else:
        rolled = np.roll(arr, -1, axis=axis)
        diff = (rolled - arr) / spacing
        idx_last = [slice(None)] * arr.ndim
        idx_prev = [slice(None)] * arr.ndim
        idx_last[axis] = -1
        idx_prev[axis] = -2
        diff[tuple(idx_last)] = diff[tuple(idx_prev)]
        return diff

# --- Backward difference with edge padding ---
def diff_backward_pad(
    arr: ArrayLike,
    axis: int,
    spacing: float = 1.0,
) -> ArrayLike:
    """
    Compute the backward finite difference along a specified axis with edge padding.

    The first element along the axis is filled using the second value to maintain
    the array shape and avoid boundary artifacts. Compatible with NumPy and PyTorch.

    Parameters
    ----------
    arr : ArrayLike
        Input array (NumPy or Torch) of any dimension.
    axis : int
        Axis along which to compute the backward difference.
    spacing : float, optional
        Grid spacing between elements along the specified axis. Default is 1.0.

    Returns
    -------
    ArrayLike
        Array of the same shape as input, containing the backward differences.
    """

    if torch.is_tensor(arr):
        diff = torch.zeros_like(arr)
        idx_curr = [slice(None)] * arr.ndim
        idx_prev = [slice(None)] * arr.ndim
        idx_curr[axis] = slice(1, None)
        idx_prev[axis] = slice(0, -1)
        diff[tuple(idx_curr)] = (arr[tuple(idx_curr)] - arr[tuple(idx_prev)]) / spacing

        # pad first slice from second
        idx_first = [slice(None)] * arr.ndim
        idx_first[axis] = slice(0, 1)
        # take first valid gradient slice (position 1 along axis)
        idx_ref = [slice(None)] * arr.ndim
        idx_ref[axis] = slice(1, 2)
        diff[tuple(idx_first)] = diff[tuple(idx_ref)]
        return diff
    else:
        diff = np.zeros_like(arr)
        idx_curr = [slice(None)] * arr.ndim
        idx_prev = [slice(None)] * arr.ndim
        idx_curr[axis] = slice(1, None)
        idx_prev[axis] = slice(0, -1)
        diff[tuple(idx_curr)] = (arr[tuple(idx_curr)] - arr[tuple(idx_prev)]) / spacing

        idx_first = [slice(None)] * arr.ndim
        idx_first[axis] = slice(0, 1)
        idx_ref = [slice(None)] * arr.ndim
        idx_ref[axis] = slice(1, 2)
        diff[tuple(idx_first)] = diff[tuple(idx_ref)]
        return diff
    
# --- Adjust axis for divergence with channel ---
def axis_adjustment(i: int, channel_axis: Optional[int] = None) -> int:
    """
    Adjust axis index if a channel axis is present before the given index.

    Parameters
    ----------
    i : int
        Original axis index (without accounting for channel axis).
    channel_axis : int or None, optional
        Index of the channel axis, if present.

    Returns
    -------
    int
        Adjusted axis index, incremented if channel axis is before it.
    """

    return i if (channel_axis is None or i < channel_axis) else i + 1

# ==============================================================
# ================ N-Dimensional Differential Tools ============
# ==============================================================

# =========================================================
# Gradient / Divergence / Laplacian (N-D, NumPy/Torch)
# =========================================================
def gradient_nd(
    u: ArrayLike,
    spacing: Optional[SpacingLike] = None,
    channel_axis: Optional[int] = None,
    backend: str = "auto",
) -> ArrayLike:
    """
    Compute forward finite differences (N-D gradient) along all spatial axes.

    The gradient is computed on all axes except the `channel_axis`, if specified.
    Supports both NumPy and PyTorch arrays, with optional backend selection.

    Parameters
    ----------
    u : ArrayLike
        Input N-dimensional array (NumPy or Torch).
    spacing : SpacingLike, optional
        Grid spacing along each axis. Can be a scalar or list of floats.
        If None, defaults to 1.0 for all axes.
    channel_axis : int or None, optional
        Index of the channel axis to exclude from gradient computation.
    backend : str, optional
        Backend to use: "auto" (default), "numpy", or "torch".

    Returns
    -------
    ArrayLike
        Array of shape (N_axes, ...) where each slice contains the gradient
        along one spatial axis.
    """

    is_tensor = torch.is_tensor(u)
    if backend == "auto":
        backend = "torch" if is_tensor else "numpy"

    axes, sp = _normalize_axes_and_spacing(u.ndim, spacing, channel_axis)

    grads: List[ArrayLike] = []
    for i, ax in enumerate(axes):
        if backend == "torch":
            rolled = torch.roll(u, shifts=-1, dims=ax)
            g = (rolled - u) / sp[i]
            # fix last slice
            idx_last = [slice(None)] * u.ndim
            idx_prev = [slice(None)] * u.ndim
            idx_last[ax] = -1
            idx_prev[ax] = -2
            g[tuple(idx_last)] = g[tuple(idx_prev)]
            grads.append(g)
        else:
            rolled = np.roll(u, -1, axis=ax)
            g = (rolled - u) / sp[i]
            idx_last = [slice(None)] * u.ndim
            idx_prev = [slice(None)] * u.ndim
            idx_last[ax] = -1
            idx_prev[ax] = -2
            g[tuple(idx_last)] = g[tuple(idx_prev)]
            grads.append(g)

    return torch.stack([g for g in grads], dim=0) if backend == "torch" else np.stack([g for g in grads], axis=0)


def divergence_nd(
    v: ArrayLike,
    spacing: Optional[SpacingLike] = None,
    channel_axis: Optional[int] = None,
    backend: str = "auto",
) -> ArrayLike:
    """
    Compute the divergence of an N-dimensional vector field.

    Assumes the input has shape (N_axes, ...) where each slice corresponds
    to a vector component along one spatial axis. The divergence is computed
    using backward differences with padding on the first slice to preserve shape.

    Parameters
    ----------
    v : ArrayLike
        Input vector field of shape (N_axes, ...), where N_axes is the number of spatial dimensions.
    spacing : SpacingLike, optional
        Grid spacing along each axis. Can be a scalar or list of floats.
        If None, defaults to 1.0 for all axes.
    channel_axis : int or None, optional
        Index of the channel axis (if any), excluded from computation.
    backend : str, optional
        Backend to use: "auto" (default), "numpy", or "torch".

    Returns
    -------
    ArrayLike
        Scalar field (same shape as a single component of `v`) representing the divergence.
    """

    is_tensor = torch.is_tensor(v)
    if backend == "auto":
        backend = "torch" if is_tensor else "numpy"

    if (is_tensor and v.dim() < 2) or (not is_tensor and np.ndim(v) < 2):
        raise ValueError("`v` must have shape (N_axes, ...).")

    n_axes = v.shape[0]
    field_ndim = v[0].ndim
    axes, sp = _normalize_axes_and_spacing(field_ndim, spacing, channel_axis)

    if len(axes) != n_axes:
        if len(sp) != n_axes:
            raise ValueError(f"Mismatch between v.shape[0]={n_axes} and spacing/axes length={len(sp)}.")

    if backend == "torch":
        out = torch.zeros_like(v[0])
        for i in range(n_axes):
            ax = axes[i]  # axis in scalar field
            rolled = torch.roll(v[i], shifts=1, dims=ax)
            d = v[i] - rolled
            # fix first slice from second
            idx_first = [slice(None)] * v[i].ndim
            idx_ref = [slice(None)] * v[i].ndim
            idx_first[ax] = 0
            idx_ref[ax] = 1
            d[tuple(idx_first)] = d[tuple(idx_ref)]
            out = out + d / sp[i]
        return out
    else:
        out = np.zeros_like(v[0])
        for i in range(n_axes):
            ax = axes[i]
            rolled = np.roll(v[i], 1, axis=ax)
            d = v[i] - rolled
            idx_first = [slice(None)] * v[i].ndim
            idx_ref = [slice(None)] * v[i].ndim
            idx_first[ax] = 0
            idx_ref[ax] = 1
            d[tuple(idx_first)] = d[tuple(idx_ref)]
            out = out + d / sp[i]
        return out


def laplacian_nd(
    u: ArrayLike,
    spacing: Optional[SpacingLike] = None,
    channel_axis: Optional[int] = None,
    backend: str = "auto",
) -> ArrayLike:
    """
    Compute the N-dimensional Laplacian of a scalar field.

    The Laplacian is computed as the divergence of the gradient: div(grad(u)).
    The output preserves the data type and device (NumPy or Torch) of the input.

    Parameters
    ----------
    u : ArrayLike
        Input scalar field (N-dimensional NumPy or Torch array).
    spacing : SpacingLike, optional
        Grid spacing along each axis. Can be a scalar or list of floats.
        If None, defaults to 1.0 for all axes.
    channel_axis : int or None, optional
        Index of the channel axis (if any), excluded from computation.
    backend : str, optional
        Backend to use: "auto" (default), "numpy", or "torch".

    Returns
    -------
    ArrayLike
        Laplacian of the input field, with same shape and type as `u`.
    """

    g = gradient_nd(u, spacing=spacing, channel_axis=channel_axis, backend=backend)
    return divergence_nd(g, spacing=spacing, channel_axis=channel_axis, backend=backend)