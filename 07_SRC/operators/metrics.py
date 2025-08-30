# ==================================================
# ===============  MODULE: metrics  ================
# ==================================================
from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union
from collections import OrderedDict

from skimage.metrics import structural_similarity as sk_ssim
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import torch

from operators.image_processor import ImageProcessor
from core.base_converter import BaseConverter
from core.config import LayoutConfig, GlobalConfig, ImageProcessorConfig

ArrayNP = np.ndarray
ArrayTorch = torch.Tensor
ArrayLike = Union[ArrayNP, ArrayTorch]
Framework = Literal["numpy", "torch", "auto"]

__all__ = [
    "scalar_product",
    "norm",
    "cosine_similarity",
    "l2_distance",
    "MSE",
    "PSNR",
    "SSIM",
    "MS_SSIM",
    "LPIPS",
    "MAE",
    "RMSE",
    "NRMSE",
    "MetricEvaluator",
    "safe_mse",
]

# ----------------------------- #
# ------- Basic utilities ----- #
# ----------------------------- #

def _ensure_same_backend(u: ArrayLike, v: ArrayLike, framework: Framework) -> Literal["numpy", "torch"]:
    """
    Ensure that two input arrays use the same computational backend (NumPy or Torch).

    Parameters
    ----------
    u : ArrayLike
        First input array.
    v : ArrayLike
        Second input array.
    framework : Framework
        Detected or specified framework (e.g., "numpy" or "torch").

    Returns
    -------
    Literal["numpy", "torch"]
        The common backend to be used for further operations.

    Raises
    ------
    ValueError
        If the two arrays belong to different frameworks.
    """
    if framework == "auto":
        if isinstance(u, torch.Tensor) and isinstance(v, torch.Tensor):
            return "torch"
        if isinstance(u, np.ndarray) and isinstance(v, np.ndarray):
            return "numpy"
        raise TypeError("When framework='auto', u and v must be of the same backend.")
    if framework not in ("numpy", "torch"):
        raise ValueError("framework must be 'numpy', 'torch' or 'auto'.")
    # Soft check: caller’s responsibility to pass consistent types
    return framework  # type: ignore[return-value]

def _as_numpy(x: ArrayLike) -> ArrayNP:
    """
    Convert the input to a NumPy array, preserving values and shape.

    Supports conversion from common array types such as PyTorch tensors.

    Parameters
    ----------
    x : ArrayLike
        Input array, either NumPy or Torch.

    Returns
    -------
    ArrayNP
        NumPy array version of the input.
    """

    return x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()  # torch -> numpy

# def _as_float(x: Union[float, np.floating]) -> float:
#     """
#     Ensure the input is returned as a Python float.

#     Parameters
#     ----------
#     x : float or np.floating
#         Input numeric value.

#     Returns
#     -------
#     float
#         Standard Python float representation of the input.
#     """

#     return float(x)

# ----------------------------- #
# ----- Vector ops (ND) ------- #
# ----------------------------- #

# --- Scalar product (dot product) ---
def scalar_product(
    u: ArrayLike,
    v: ArrayLike,
    axis: Optional[int | Sequence[int]] = None,
    framework: Framework = "auto"
) -> ArrayLike:
    """
    Compute the dot (scalar) product of two arrays along the specified axis or axes.

    Supports both NumPy and Torch backends, and returns a result with the same framework
    as the inputs. Broadcasting rules are applied if needed.

    Parameters
    ----------
    u : ArrayLike
        First input array (NumPy or Torch).
    v : ArrayLike
        Second input array (must be broadcast-compatible with `u`).
    axis : int or Sequence[int], optional
        Axis or axes along which to perform the dot product.
        If None, uses all axes where broadcasting is possible.
    framework : {"auto", "numpy", "torch"}, optional
        Backend to use. If "auto", detected from inputs.

    Returns
    -------
    ArrayLike
        Scalar product computed along the given axis, with same backend as inputs.
    """

    fw = _ensure_same_backend(u, v, framework)
    if fw == "torch":
        with torch.no_grad():
            return torch.sum(u * v, dim=axis)
    return np.sum(u * v, axis=axis)

# --- Euclidean norm ---
def norm(
    u: ArrayLike,
    axis: Optional[int | Sequence[int]] = None,
    framework: Framework = "auto"
) -> ArrayLike:
    """
    Compute the L2 norm (Euclidean norm) of an array along the specified axis or axes.

    Supports both NumPy and Torch backends. Returns a result with the same framework
    and dtype as the input.

    Parameters
    ----------
    u : ArrayLike
        Input array (NumPy or Torch).
    axis : int or Sequence[int], optional
        Axis or axes along which to compute the norm. If None, computes the global norm.
    framework : {"auto", "numpy", "torch"}, optional
        Backend to use. If "auto", inferred from the input.

    Returns
    -------
    ArrayLike
        Array of norms computed along the specified axis, with same backend as input.
    """

    if framework == "auto":
        fw = "torch" if isinstance(u, torch.Tensor) else "numpy"
    else:
        fw = framework
    if fw == "torch":
        with torch.no_grad():
            return torch.sqrt(scalar_product(u, u, axis, "torch"))
    return np.sqrt(scalar_product(u, u, axis, "numpy"))

# --- Cosine similarity ---
def cosine_similarity(
    u: ArrayLike,
    v: ArrayLike,
    axis: Optional[int | Sequence[int]] = None,
    framework: Framework = "auto"
) -> ArrayLike:
    """
    Compute the cosine similarity between two arrays along the specified axis or axes.

    Cosine similarity is defined as the dot product of `u` and `v` divided by the product
    of their L2 norms. Supports NumPy and Torch backends.

    Parameters
    ----------
    u : ArrayLike
        First input array (NumPy or Torch).
    v : ArrayLike
        Second input array (must be broadcast-compatible with `u`).
    axis : int or Sequence[int], optional
        Axis or axes along which to compute the similarity.
        If None, uses all common dimensions.
    framework : {"auto", "numpy", "torch"}, optional
        Backend to use. If "auto", inferred from the inputs.

    Returns
    -------
    ArrayLike
        Cosine similarity values, with same shape and backend as inputs (excluding `axis`).
    """

    fw = _ensure_same_backend(u, v, framework)
    dot = scalar_product(u, v, axis, fw)
    nu = norm(u, axis, fw)
    nv = norm(v, axis, fw)
    eps = 1e-8
    if fw == "torch":
        with torch.no_grad():
            return dot / (nu * nv + eps)
    return dot / (nu * nv + eps)

# --- L2 distance ---
def l2_distance(
    u: ArrayLike,
    v: ArrayLike,
    axis: Optional[int | Sequence[int]] = None,
    framework: Framework = "auto"
) -> ArrayLike:
    """
    Compute the Euclidean (L2) distance between two arrays along the specified axis or axes.

    Equivalent to the L2 norm of (u - v). Supports both NumPy and Torch backends.

    Parameters
    ----------
    u : ArrayLike
        First input array (NumPy or Torch).
    v : ArrayLike
        Second input array (must be broadcast-compatible with `u`).
    axis : int or Sequence[int], optional
        Axis or axes along which to compute the distance.
        If None, computes global distance.
    framework : {"auto", "numpy", "torch"}, optional
        Backend to use. If "auto", inferred from the inputs.

    Returns
    -------
    ArrayLike
        L2 distance between `u` and `v`, with the same backend as the inputs.
    """

    fw = _ensure_same_backend(u, v, framework)
    return norm(u - v, axis, fw)


# ------------------------------------ #
# -------- Scalar image metrics ------ #
# ------------------------------------ #

# ====[MSE and PSNR]====

def MSE(u_truth: ArrayLike, u_estim: ArrayLike, clip_val: float = 1e5) -> float:
    """
    Compute the Mean Squared Error (MSE) between two arrays, with optional clipping.

    Clipping is applied to avoid exploding values in case of unstable predictions.

    Parameters
    ----------
    u_truth : ArrayLike
        Ground truth array (reference values).
    u_estim : ArrayLike
        Estimated or predicted array.
    clip_val : float, optional
        Maximum value to clip the squared error. Default is 1e5.

    Returns
    -------
    float
        Mean squared error between `u_truth` and `u_estim`.
    """

    if u_truth.shape != u_estim.shape:
        raise ValueError("Input arrays must have the same shape.")
    if isinstance(u_truth, torch.Tensor) and isinstance(u_estim, torch.Tensor):
        with torch.no_grad():
            diff = torch.clamp(u_truth - u_estim, min=-clip_val, max=clip_val)
            return float(torch.mean(diff ** 2).item())
    diff_np = np.clip(_as_numpy(u_truth) - _as_numpy(u_estim), -clip_val, clip_val)
    return float(np.mean(diff_np ** 2))

def PSNR(
    u_truth: ArrayLike,
    u_estim: ArrayLike,
    max_intensity: float = 1.0,
    clip_val: float = 1e5
) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) in decibels (dB) between two arrays.

    PSNR is calculated based on the Mean Squared Error (MSE), with optional clipping
    for numerical stability. If MSE is zero, PSNR returns +∞ (perfect reconstruction).

    Parameters
    ----------
    u_truth : ArrayLike
        Ground truth array (reference values).
    u_estim : ArrayLike
        Estimated or predicted array.
    max_intensity : float, optional
        Maximum possible intensity value in the input images. Default is 1.0.
    clip_val : float, optional
        Maximum value to clip the squared error before computing MSE. Default is 1e5.

    Returns
    -------
    float
        PSNR value in decibels (dB).
    """

    if u_truth.shape != u_estim.shape:
        raise ValueError("Input images must have the same shape.")
    if isinstance(u_truth, torch.Tensor) and isinstance(u_estim, torch.Tensor):
        with torch.no_grad():
            diff = torch.clamp(u_truth - u_estim, min=-clip_val, max=clip_val)
            mse_val = float(torch.mean(diff ** 2).item())
    else:
        diff = np.clip(_as_numpy(u_truth) - _as_numpy(u_estim), -clip_val, clip_val)
        mse_val = float(np.mean(diff ** 2))
    if mse_val == 0.0:
        return float("inf")
    return 20.0 * np.log10(max_intensity) - 10.0 * np.log10(mse_val)

# -------------------------- #
# ---- SSIM & MS-SSIM ------ #
# -------------------------- #

# --- Structural Similarity Index (SSIM) ---
def SSIM(
    u_truth: ArrayLike,
    u_estim: ArrayLike,
    return_map: bool = False,
    framework: Literal["numpy", "torch"] = "numpy",
    output_format: Literal["numpy", "torch"] = "numpy",
    layout_framework: Literal["numpy", "torch"] = "numpy",
    layout_name: str = "HWC",
    processor_strategy: Optional[str] = "classic",
    backend: str = "sequential",
) -> Union[float, ArrayLike]:
    """
    Compute the Structural Similarity Index (SSIM) between two arrays.

    SSIM is a perceptual metric that evaluates image similarity in terms of
    luminance, contrast, and structure. This implementation supports N-dimensional
    data and uses the `ImageProcessor` to handle slicing or vectorized computation.

    Parameters
    ----------
    u_truth : ArrayLike
        Ground truth array (reference image or volume).
    u_estim : ArrayLike
        Estimated or predicted image/volume.
    return_map : bool, optional
        If True, return the full SSIM similarity map instead of a global score.
    framework : {"numpy", "torch"}, optional
        Framework used for internal computation. Default is "numpy".
    output_format : {"numpy", "torch"}, optional
        Format of the output. Default is "numpy".
    layout_framework : {"numpy", "torch"}, optional
        Framework used to interpret layout strings (e.g., "HWC", "CHW").
    layout_name : str, optional
        Layout string specifying axis order (e.g., "HWC", "NCHW"). Default is "HWC".
    processor_strategy : str or None, optional
        Strategy for image processing: "classic", "vectorized", or "torch".
        If None, auto-selects: "vectorized" for NumPy, "torch" for Torch.
    backend : str, optional
        Backend strategy for slice processing (e.g., "sequential", "parallel").

    Returns
    -------
    float or ArrayLike
        Global SSIM score (float) or full SSIM map (array), depending on `return_map`.

    Notes
    -----
    - If a slice is constant, a minimum `data_range` of 1e-8 is enforced to prevent numerical errors.
    - Uses scikit-image's SSIM implementation internally via ImageProcessor abstraction.
    """

    def ssim_slice(x: ArrayLike, y: ArrayLike) -> Union[float, Tuple[float, ArrayNP]]:
        dr = float(x.max() - x.min())
        dr = dr if dr > 1e-8 else 1.0
        return sk_ssim(x, y, data_range=dr, full=return_map)

    chosen_strategy = processor_strategy if processor_strategy is not None else ("vectorized" if framework == "numpy" else "torch")

    proc_params = {"processor_strategy": chosen_strategy}
    layout_params = {"layout_name": layout_name, "layout_framework": layout_framework}
    global_params = {"framework": framework, "output_format": output_format, "backend": backend}

    processor = ImageProcessor(
        img_process_cfg=ImageProcessorConfig(function=ssim_slice, **proc_params),
        layout_cfg=LayoutConfig(**layout_params),
        global_cfg=GlobalConfig(**global_params),
    )
    result = processor(u_truth, u_estim)
    if not return_map and isinstance(result, (np.ndarray, torch.Tensor)):
        return float(result.mean().item() if isinstance(result, torch.Tensor) else float(result.mean()))
    return result

# ====[ Multi-Scale SSIM (MS-SSIM) – Torch Only ]====
def MS_SSIM(
    u_truth: ArrayLike,
    u_estim: ArrayLike,
    framework: Literal["torch", "numpy"] = "torch",
    output_format: Literal["torch", "numpy"] = "torch",
    layout_framework: Literal["numpy", "torch"] = "numpy",
    layout_name: str = "HWC",
    processor_strategy: Optional[str] = "torch",
    backend: str = "sequential",
    normalize: bool = True,
    return_float: bool = True,
) -> Union[float, ArrayLike]:
    """
    Compute the Multi-Scale Structural Similarity Index (MS-SSIM) between two arrays.

    This metric evaluates perceptual similarity across multiple scales. Internally,
    it relies on the `pytorch_msssim` library and is only compatible with PyTorch tensors.

    Parameters
    ----------
    u_truth : ArrayLike
        Ground truth array (reference image or volume).
    u_estim : ArrayLike
        Estimated or predicted array.
    framework : {"torch", "numpy"}, optional
        Specifies the accepted input format. Only "torch" is supported for computation.
    output_format : {"torch", "numpy"}, optional
        Desired format of the output. Default is "torch".
    layout_framework : {"numpy", "torch"}, optional
        Framework used to interpret the layout string.
    layout_name : str, optional
        Layout string defining axis order (e.g., "HWC", "NCHW").
    processor_strategy : str or None, optional
        Strategy for processing data before feeding into MS-SSIM. Default is "torch".
    backend : str, optional
        Backend strategy for slice processing (e.g., "sequential", "parallel").
    normalize : bool, optional
        If True, normalize input tensors to [0, 1] before computing MS-SSIM.
    return_float : bool, optional
        If True, return the score as a float. If False, return as tensor/array.

    Returns
    -------
    float or ArrayLike
        Global MS-SSIM score as float (if `return_float=True`) or in original backend format.

    Notes
    -----
    - Only works with PyTorch backend. Inputs will be converted if necessary.
    - Uses the `pytorch_msssim` package for MS-SSIM computation.
    """

    try:
        from pytorch_msssim import ms_ssim
    except Exception as e:
        raise ImportError("pytorch_msssim not available. Please install it to use MS_SSIM.") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def _ms_ssim(x: ArrayLike, y: ArrayLike) -> float:
        xt = x.to(dtype=torch.float32, device=device)
        yt = y.to(dtype=torch.float32, device=device)
        # shapes: (H,W) → (1,1,H,W), (C,H,W) → (1,C,H,W), (B,C,H,W) → pass-through
        if xt.dim() == 2:
            xt = xt.unsqueeze(0).unsqueeze(0)
            yt = yt.unsqueeze(0).unsqueeze(0)
        elif xt.dim() == 3 and xt.shape[0] <= 4:  # assume (C,H,W) if C<=4
            xt = xt.unsqueeze(0)
            yt = yt.unsqueeze(0)
        # assume data_range=1.0 when normalize=True
        return float(ms_ssim(xt, yt, data_range=1.0 if normalize else float(xt.max() - xt.min()), size_average=True).item())

    chosen_strategy = processor_strategy if processor_strategy is not None else ("vectorized" if framework == "numpy" else "torch")

    proc_params = {"processor_strategy": chosen_strategy, "fallback": True}
    layout_params = {"layout_name": layout_name, "layout_framework": layout_framework}
    global_params = {
        "framework": framework,
        "output_format": output_format,
        "backend": backend,
        "normalize": normalize,
        "add_batch_dim": True,
        "device": str(device),
    }

    processor = ImageProcessor(
        img_process_cfg=ImageProcessorConfig(function=_ms_ssim, **proc_params),
        layout_cfg=LayoutConfig(**layout_params),
        global_cfg=GlobalConfig(**global_params),
    )
    result = processor(u_truth, u_estim)
    return float(result.mean().item()) if return_float else result

# -------------------------- #
# ---- Perceptual metric --- #
# -------------------------- #

# ====[ Compute Learned Perceptual Image Patch Similarity (LPIPS) ]====
_lpips_cache = {}

def LPIPS(
    u_truth: ArrayLike,
    u_estim: ArrayLike,
    net_type: Literal["vgg", "alex", "squeeze"] = "vgg",
    framework: Literal["torch", "numpy"] = "torch",
    output_format: Literal["numpy", "torch"] = "numpy",
    layout_framework: Literal["torch", "numpy"] = "torch",
    layout_name: str = "NCHW",
    add_batch_dim: bool = True,
    normalize: bool = True,
    return_float: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[float, ArrayLike]:
    """
    Compute the LPIPS (Learned Perceptual Image Patch Similarity) metric between two images.

    LPIPS is a deep-learning-based perceptual similarity measure that uses a pre-trained
    network (VGG, AlexNet, or SqueezeNet) to compare features extracted from image patches.

    Parameters
    ----------
    u_truth : ArrayLike
        Ground truth image or batch of images.
    u_estim : ArrayLike
        Estimated or predicted image or batch.
    net_type : {"vgg", "alex", "squeeze"}, optional
        Backbone network to use for LPIPS computation. Default is "vgg".
    framework : {"torch", "numpy"}, optional
        Input data format. Must be convertible to PyTorch for processing.
    output_format : {"torch", "numpy"}, optional
        Format of the returned LPIPS score.
    layout_framework : {"torch", "numpy"}, optional
        Framework used to interpret the layout string (e.g., "NCHW").
    layout_name : str, optional
        Layout string specifying axis order. Default is "NCHW".
    add_batch_dim : bool, optional
        If True, adds a batch dimension if input is a single image.
    normalize : bool, optional
        If True, normalize input images to the LPIPS expected range.
    return_float : bool, optional
        If True, return a float scalar. If False, return a tensor or array.
    device : str or torch.device, optional
        Device on which to run the LPIPS model. If None, use default device.

    Returns
    -------
    float or ArrayLike
        LPIPS distance between `u_truth` and `u_estim`, as float or backend array.

    Notes
    -----
    - Requires the `lpips` PyTorch package (learned perceptual similarity).
    - Only compatible with Torch backend internally.
    - Input images should be in range [0, 1] or normalized appropriately.
    """

    try:
        from lpips import LPIPS as LPIPS_Model
    except Exception as e:
        raise ImportError("lpips not available. Please install it to use LPIPS.") from e

    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    layout_params = {"layout_name": layout_name, "layout_framework": layout_framework}
    global_params = {
        "framework": framework,
        "output_format": output_format,
        "add_batch_dim": add_batch_dim,
        "normalize": normalize,
        "device": str(dev),
    }

    # ====[ Image Conversion with tagging ]====
    converter = BaseConverter(
                            layout_cfg=LayoutConfig(**layout_params),
                            global_cfg=GlobalConfig(**global_params),
                            )

    img1 = converter.convert_once(u_truth, framework=framework, track=True, trace_limit=10, normalize_override=None)
    img2 = converter.convert_once(u_estim, framework=framework, track=True, trace_limit=10, normalize_override=None)


    # ====[ LPIPS Model – with cache ]====
    if net_type not in _lpips_cache:
        _lpips_cache[net_type] = LPIPS_Model(net=net_type).to(dev).eval()

    model = _lpips_cache[net_type]

    with torch.no_grad():
        val = model(img1.to(dev), img2.to(dev))

    if return_float:
        return float(val.item())

    if output_format == "numpy":
        return val.detach().cpu().numpy()
    return val

# ------------------------------------ #
# ---------- Other metrics ----------- #
# ------------------------------------ #

# ====[ Mean Absolute Error ]====
def MAE(
    u_truth: ArrayLike,
    u_estim: ArrayLike,
    framework: Literal["numpy", "torch"] = "numpy",
    output_format: Literal["numpy", "torch"] = "numpy",
    layout_framework: Literal["numpy", "torch"] = "numpy",
    layout_name: str = "HWC",
    processor_strategy: Optional[str] = "classic",
    backend: str = "sequential",
) -> float:
    """
    Compute the Mean Absolute Error (MAE) between two arrays.

    Supports N-dimensional data and both NumPy and Torch backends via the
    ImageProcessor abstraction.

    Parameters
    ----------
    u_truth : ArrayLike
        Ground truth array (reference image or volume).
    u_estim : ArrayLike
        Estimated or predicted image/volume.
    framework : {"numpy", "torch"}, optional
        Backend used for computation. Default is "numpy".
    output_format : {"numpy", "torch"}, optional
        Format of the returned result. Default is "numpy".
    layout_framework : {"numpy", "torch"}, optional
        Framework used to interpret the layout string.
    layout_name : str, optional
        Layout string defining axis order (e.g., "HWC", "NCHW"). Default is "HWC".
    processor_strategy : str or None, optional
        Strategy for image processing ("classic", "vectorized", "torch", etc.).
    backend : str, optional
        Backend processing mode ("sequential", "parallel", etc.).

    Returns
    -------
    float
        Mean Absolute Error between `u_truth` and `u_estim`.
    """

    def mae_slice(x: ArrayLike, y: ArrayLike) -> ArrayLike:
        return np.abs(x - y) if isinstance(x, np.ndarray) else torch.abs(x - y)

    chosen_strategy = processor_strategy if processor_strategy is not None else ("vectorized" if framework == "numpy" else "torch")
    proc_params = {"processor_strategy": chosen_strategy}
    layout_params = {"layout_name": layout_name, "layout_framework": layout_framework}
    global_params = {"framework": framework, "output_format": output_format, "backend": backend}

    processor = ImageProcessor(
        img_process_cfg=ImageProcessorConfig(function=mae_slice, **proc_params),
        layout_cfg=LayoutConfig(**layout_params),
        global_cfg=GlobalConfig(**global_params),
    )
    result = processor(u_truth, u_estim)
    if isinstance(result, torch.Tensor):
        return float(result.mean().item())
    return float(np.mean(result))

# ====[ Root Mean Squared Error ]====
def RMSE(
    u_truth: ArrayLike,
    u_estim: ArrayLike,
    framework: Literal["numpy", "torch"] = "numpy",
    output_format: Literal["numpy", "torch"] = "numpy",
    layout_framework: Literal["numpy", "torch"] = "numpy",
    layout_name: str = "HWC",
    processor_strategy: Optional[str] = "classic",
    backend: str = "sequential",
) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) between two arrays.

    RMSE is the square root of the Mean Squared Error (MSE), providing
    an interpretable error metric in the same unit as the original data.
    Supports ND inputs with NumPy or Torch backends via ImageProcessor.

    Parameters
    ----------
    u_truth : ArrayLike
        Ground truth array (reference image or volume).
    u_estim : ArrayLike
        Estimated or predicted image/volume.
    framework : {"numpy", "torch"}, optional
        Backend used for computation. Default is "numpy".
    output_format : {"numpy", "torch"}, optional
        Format of the returned result. Default is "numpy".
    layout_framework : {"numpy", "torch"}, optional
        Framework used to interpret the layout string.
    layout_name : str, optional
        Layout string defining axis order (e.g., "HWC", "NCHW"). Default is "HWC".
    processor_strategy : str or None, optional
        Strategy for image processing ("classic", "vectorized", "torch", etc.).
    backend : str, optional
        Backend processing mode ("sequential", "parallel", etc.).

    Returns
    -------
    float
        Root Mean Squared Error between `u_truth` and `u_estim`.
    """

    def rmse_slice(x: ArrayLike, y: ArrayLike) -> float:
        return float(np.sqrt(((x - y) ** 2).mean()))

    chosen_strategy = processor_strategy if processor_strategy is not None else ("vectorized" if framework == "numpy" else "torch")
    proc_params = {"processor_strategy": chosen_strategy}
    layout_params = {"layout_name": layout_name, "layout_framework": layout_framework}
    global_params = {"framework": framework, "output_format": output_format, "backend": backend}

    processor = ImageProcessor(
        img_process_cfg=ImageProcessorConfig(function=rmse_slice, **proc_params),
        layout_cfg=LayoutConfig(**layout_params),
        global_cfg=GlobalConfig(**global_params),
    )
    result = processor(u_truth, u_estim)
    if isinstance(result, torch.Tensor):
        return float(result.mean().item())
    return float(np.mean(result))

# ====[ Normalized RMSE ]====
def NRMSE(
    u_truth: ArrayLike,
    u_estim: ArrayLike,
    framework: Literal["numpy", "torch"] = "numpy",
    output_format: Literal["numpy", "torch"] = "numpy",
    layout_framework: Literal["numpy", "torch"] = "numpy",
    layout_name: str = "HWC",
    processor_strategy: Optional[str] = "classic",
    backend: str = "sequential",
) -> float:
    """
    Compute the Normalized Root Mean Squared Error (NRMSE) between two arrays.

    NRMSE is the RMSE divided by the dynamic range (max - min) of the ground truth.
    This normalization makes the metric scale-invariant and easier to interpret
    across datasets with different value ranges.

    Parameters
    ----------
    u_truth : ArrayLike
        Ground truth array (reference image or volume).
    u_estim : ArrayLike
        Estimated or predicted image/volume.
    framework : {"numpy", "torch"}, optional
        Backend used for computation. Default is "numpy".
    output_format : {"numpy", "torch"}, optional
        Format of the returned result. Default is "numpy".
    layout_framework : {"numpy", "torch"}, optional
        Framework used to interpret the layout string.
    layout_name : str, optional
        Layout string defining axis order (e.g., "HWC", "NCHW"). Default is "HWC".
    processor_strategy : str or None, optional
        Strategy for image processing ("classic", "vectorized", "torch", etc.).
    backend : str, optional
        Backend processing mode ("sequential", "parallel", etc.).

    Returns
    -------
    float
        Normalized Root Mean Squared Error between `u_truth` and `u_estim`.
    """

    def nrmse_slice(x: ArrayLike, y: ArrayLike) -> float:
        rmse = float(np.sqrt(((x - y) ** 2).mean()))
        rang = float(x.max() - x.min())
        return rmse / (rang if rang > 1e-12 else 1.0)

    chosen_strategy = processor_strategy if processor_strategy is not None else ("vectorized" if framework == "numpy" else "torch")
    proc_params = {"processor_strategy": chosen_strategy}
    layout_params = {"layout_name": layout_name, "layout_framework": layout_framework}
    global_params = {"framework": framework, "output_format": output_format, "backend": backend}

    processor = ImageProcessor(
        img_process_cfg=ImageProcessorConfig(function=nrmse_slice, **proc_params),
        layout_cfg=LayoutConfig(**layout_params),
        global_cfg=GlobalConfig(**global_params),
    )
    result = processor(u_truth, u_estim)
    if isinstance(result, torch.Tensor):
        return float(result.mean().item())
    return float(np.mean(result))
# ==================================================
# ============ Metric Evaluator Class ==============
# ==================================================

class MetricEvaluator:
    """
    Centralized metric evaluator to compute multiple image metrics
    in one pass, with individual parameter handling.
    """

    def __init__(self, metrics=None, return_dict=True):
        """
        Parameters
        ----------
        metrics : list of str or None
            Metrics to compute (e.g., ['mse', 'psnr', 'ssim', ...]).
            If None, compute all available metrics.
        return_dict : bool
            If True, return an ordered dictionary of results.
        """
        self.available_metrics: Dict[str, Tuple[Callable[..., Any], Dict[str, Any]]] = {
            "mse": (MSE, {}),
            "mae": (MAE, {}),
            "rmse": (RMSE, {}),
            "nrmse": (NRMSE, {}),
            "psnr": (PSNR, {}),
            "ssim": (SSIM, {"return_map": False}),
            "ms-ssim": (MS_SSIM, {"normalize": True}),
            "lpips": (LPIPS, {"normalize": True}),
        }
        self.metrics: Sequence[str] = list(self.available_metrics.keys()) if metrics is None else metrics
        self.return_dict: bool = return_dict

    def __call__(self, u_truth, u_estim, **global_kwargs):
        """
        Compute selected metrics between two images.

        Parameters
        ----------
        u_truth, u_estim : np.ndarray or torch.Tensor
            Ground truth and predicted images.
        global_kwargs : dict
            Global arguments passed to all metric functions.

        Returns
        -------
        dict or list
            Dictionary (or list) of metric results.
        """
        results: "OrderedDict[str, Any]" = OrderedDict()
        for name in self.metrics:
            try:
                func, specific_kwargs = self.available_metrics[name]
                kwargs = {**specific_kwargs, **global_kwargs}
                results[name] = func(u_truth, u_estim, **kwargs)
            except Exception as e:
                results[name] = f"Error: {e}"
        return results if self.return_dict else list(results.values())


# --- NumPy-only safe MSE helper ---
def safe_mse(u1: ArrayNP, u2: ArrayNP, clip_val: float = 1e5) -> float:
    """
    Compute the Mean Squared Error (MSE) between two NumPy arrays with optional clipping.

    This version clips extreme differences for numerical stability and does not support Torch tensors.

    Parameters
    ----------
    u1 : ArrayNP
        First input array (NumPy only).
    u2 : ArrayNP
        Second input array (NumPy only).
    clip_val : float, optional
        Maximum absolute value of the difference before squaring. Default is 1e5.

    Returns
    -------
    float
        Clipped mean squared error between `u1` and `u2`.
    """
    if not (isinstance(u1, np.ndarray) and isinstance(u2, np.ndarray)):
        raise ValueError("safe_mse only supports NumPy arrays.")

    diff = np.clip(u1 - u2, -clip_val, clip_val)
    return float(np.mean(diff ** 2))


# ---------------------------------------------- #
# -------- Grid search (backend-free) ---------- #
# ---------------------------------------------- #

def search_opt_general(    
    func: Callable[..., ArrayLike | Tuple[ArrayLike, ...]],
    u_truth: ArrayLike,
    param_grid: Mapping[str, Sequence[Any]],
    metric: Optional[Callable[[ArrayLike, ArrayLike], float | Mapping[str, float]]] = None,
    metric_name: Optional[str] = None,
    func_args: Optional[Mapping[str, Any]] = None,
    sub_params_key: Optional[str] = None,
    sub_param_grid: Optional[Mapping[str, Sequence[Any]]] = None,
    return_results: bool = False,
    verbose: bool = True,
) -> Union[
    Tuple[Dict[str, Any], float, pd.DataFrame],
    Tuple[Tuple[Dict[str, Any], Dict[str, Any]], float, pd.DataFrame],
    Tuple[Dict[str, Any], float, pd.DataFrame, ArrayLike],
    Tuple[Tuple[Dict[str, Any], Dict[str, Any]], float, pd.DataFrame, ArrayLike],
]:
    """
    Perform exhaustive parameter search to optimize a given function with optional sub-parameters.

    Parameters
    ----------
    func : callable
        Function to optimize. Can accept both *args and **kwargs.
    u_truth : np.ndarray
        Ground-truth data to evaluate against.
    param_grid : dict
        Main parameters to test (grid search). Keys are param names, values are lists of values.
    metric : callable
        Scoring function. Must accept (truth, prediction) and return a float.
    metric_name : str, optional
        Name of the metric to extract from dict (if metric returns dict).
    func_args : dict, optional
        Fixed arguments passed to func. Can include "args": [...] for positional args.
    sub_param_grid : dict, optional
        Secondary parameter space (e.g. for internal blocks like 'prox').
    sub_params_key : str, optional
        Key under which to inject sub-parameters into func kwargs.
    return_results : bool, optional
        If True, returns best result in addition to score and params.
    verbose : bool, optional
        If True, logs progress and errors.

    Returns
    -------
    best_params : dict or tuple
        Best combination of parameters (and sub-parameters if any).
    best_score : float
        Best score achieved.
    score_map : pd.DataFrame
        Score table of all combinations.
    best_result : np.ndarray, optional
        Only returned if return_results=True.
    """
    if func_args is None:
        func_args = {}
    if not param_grid:
        raise ValueError("param_grid cannot be empty.")
    
    if metric_name is not None and metric is None:
        metric = MetricEvaluator(metrics=[metric_name], return_dict=True)
    else:
        metric = MetricEvaluator(metrics=["psnr"], return_dict=True)
        metric_name = "psnr"
    
    # Build grid    
    main_keys = list(param_grid.keys())
    main_vals = list(param_grid.values())
    main_combos = list(itertools.product(*main_vals))

    use_subgrid = sub_param_grid is not None and bool(sub_param_grid)
    sub_keys = list(sub_param_grid.keys()) if use_subgrid else []
    sub_vals = list(sub_param_grid.values()) if use_subgrid else []
    sub_combos = list(itertools.product(*sub_vals)) if use_subgrid else [None]

    score_data: List[Any] = []
    best_score = -np.inf
    best_params: Any = None
    best_output: Optional[ArrayLike] = None

    total_iters = len(main_combos) * len(sub_combos)
    with tqdm(total=total_iters, desc="Grid search") as bar:
        for main in main_combos:
            current_main = dict(zip(main_keys, main))

            for sub in sub_combos:
                current_sub = dict(zip(sub_keys, sub)) if sub else {}

                all_params = func_args.copy()
                all_params.update(current_main)
                
                if use_subgrid:
                    all_params[sub_params_key] = current_sub

                try:
                    result = func(**all_params)
                    result = result[0] if isinstance(result, tuple) else result

                    if result.shape != u_truth.shape:
                        raise ValueError("Shape mismatch between result and u_truth.")

                    score_raw = metric(u_truth, result)

                    if isinstance(score_raw, dict):
                        if metric_name is None:
                            raise ValueError("metric_name must be specified when metric returns a dict.")
                        score = score_raw.get(metric_name, None)
                        if score is None:
                            raise ValueError(f"Metric '{metric_name}' not found in result.")
                    else:
                        score = float(score_raw)
                        
                    row = (current_main.copy(), current_sub.copy(), score) if use_subgrid else (current_main.copy(), score)
                    score_data.append(row)

                    if score > best_score:
                        best_score = score
                        best_params = (current_main.copy(), current_sub.copy()) if use_subgrid else current_main.copy()
                        best_output = result.copy()

                except Exception as e:
                    if verbose:
                        print(f"[!] Error with params={current_main}, prox={current_sub} \u2192 {e}")

                bar.update(1)

    # Create DataFrame
    if use_subgrid:
        df = pd.DataFrame([(p, s, sc) for p, s, sc in score_data], columns=["Params", "SubParams", "Score"])
    else:
        df = pd.DataFrame([(p, sc) for p, sc in score_data], columns=["Params", "Score"])

    df = df.sort_values(by="Score", ascending=False)

    if return_results:
        return best_params, best_score, df, best_output
    else:
        return best_params, best_score, df

