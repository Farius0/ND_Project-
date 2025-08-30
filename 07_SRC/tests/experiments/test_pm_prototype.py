import numpy as np, itertools, pandas as pd, torch
from tqdm import tqdm
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d
from skimage.filters import threshold_otsu
from pathlib import Path

import sys
sys.path.append(str(Path.cwd().parent))
from utils.decorators import TimerManager
from utils.image_io import ImageIO
from utils.nd_tools import show_plane

from core.config import (LayoutConfig, GlobalConfig,)

profiler = TimerManager()

def process_image_general(*images, function, channel_axis=None, return_tuple=False, **kwargs):
    """
    Apply a function to one or more images, possibly multi-channel or stacked.

    Parameters
    ----------
    *images : array-like
        One or more image arrays of identical shape.
    function : callable
        The function to apply to each image or slice.
    channel_axis : int or None, optional
        Axis to iterate over (e.g. channels or slices). If None, apply on full image(s).
    return_tuple : bool, optional
        If True, assumes the function returns a tuple (e.g. output + extra data).
        Outputs will be grouped into separate lists per returned element.
    **kwargs : dict
        Additional arguments passed to the function.

    Returns
    -------
    result : np.ndarray or tuple of np.ndarray
        If return_tuple is False, returns one array with the same shape as input.
        If return_tuple is True, returns tuple of stacked arrays per output.
    """
    if len(images) < 1:
        raise ValueError("At least one image must be provided.")

    base_shape = images[0].shape
    for img in images:
        if img.shape != base_shape:
            raise ValueError("All input images must have the same shape.")

    # No channel axis: apply directly
    if channel_axis is None:
        return function(*images, **kwargs)

    # Move channel_axis to first position
    images_swapped = [np.swapaxes(img, 0, channel_axis) for img in images]
    n_slices = images_swapped[0].shape[0]

    if return_tuple:
        # Each call returns a tuple — we group outputs by index
        results = [function(*(img[i] for img in images_swapped), **kwargs) for i in range(n_slices)]

        # Group by output type (zip the tuples)
        grouped = list(zip(*results))
        # Stack back per output
        outputs = tuple(np.swapaxes(np.stack(out), 0, channel_axis) for out in grouped)
        return outputs
    else:
        # Standard processing
        processed = [function(*(img[i] for img in images_swapped), **kwargs) for i in range(n_slices)]
        return np.swapaxes(np.stack(processed), 0, channel_axis)


def scalar_product(u, v):
    """
    Compute the scalar (dot) product between two arrays.

    Parameters
    ----------
    u : array_like
        First input array.
    v : array_like
        Second input array.

    Returns
    -------
    float
        Scalar product of u and v.

    Raises
    ------
    ValueError
        If input shapes do not match.
    """
    if u.shape != v.shape:
        raise ValueError("Input shapes must be the same.")
    return np.sum(u * v)

def norm(u, axis=None):
    """
    Compute the Euclidean norm of an array.

    Parameters
    ----------
    u : array_like
        Input array.
    axis : int or tuple of int, optional
        Axis or axes along which to compute the norm.
        Default is None (norm over the whole array).

    Returns
    -------
    float or np.ndarray
        Euclidean norm.

    Raises
    ------
    TypeError
        If input is not a numpy array.
    """
    if not isinstance(u, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    return np.sqrt(np.sum(u ** 2, axis=axis))

@profiler.decorator()
def gradient_nd(u, spacing=None, channel_axis=None, include_channel_gradient=False):
    """
    Compute the gradient of an n-dimensional array using forward differences.
    Allows to include or exclude channel axis from gradient computation.

    Parameters
    ----------
    u : np.ndarray
        Input array. Can be:
        - 2D: (H, W)
        - 3D: (H, W, C), (C, H, W), (Z, Y, X)
        - 4D: (Z, Y, X, C)

    spacing : tuple of float, optional
        Physical spacing between elements along spatial axes.

    channel_axis : int or None, optional
        Axis index for the channel (color or spectral). If given, that axis will
        be excluded from spatial gradient. Use `include_channel_gradient=True`
        to also compute gradient on that axis.

    include_channel_gradient : bool, optional
        Whether to compute gradient along the channel axis.
        Default is False.

    Returns
    -------
    grad : np.ndarray
        Array of shape (n_directions, ...) containing all gradients.
    """

    u = np.asarray(u)
    ndim = u.ndim

    # Determine which axes to compute spatial gradient on
    spatial_axes = [i for i in range(ndim) if i != channel_axis]

    # Default spacing is 1 for each spatial axis
    if spacing is None:
        spacing = [1.0] * len(spatial_axes)
    elif len(spacing) != len(spatial_axes):
        raise ValueError(f"Expected spacing of length {len(spatial_axes)}, got {len(spacing)}")

    grad_list = []

    # Spatial gradients
    for i, axis in enumerate(spatial_axes):
        slicer_curr, slicer_prev = [slice(None)] * ndim, [slice(None)] * ndim
        slicer_curr[axis], slicer_prev[axis] = slice(1, None), slice(0, -1)
        diff = (u[tuple(slicer_curr)] - u[tuple(slicer_prev)]) / spacing[i]

        # Pad the result to restore shape
        pad_config = [(0, 1) if j == axis else (0, 0) for j in range(ndim)]
        grad_list.append(np.pad(diff, pad_config, mode='edge'))

    # Optional gradient on channel axis
    if include_channel_gradient and channel_axis is not None:
        slicer_curr, slicer_prev = [slice(None)] * ndim, [slice(None)] * ndim
        slicer_curr[channel_axis], slicer_prev[channel_axis] = slice(1, None), slice(0, -1)
        diff = u[*slicer_curr] - u[*slicer_prev]

        # Pad the channel dimension
        pad_config = [(0, 1) if j == channel_axis else (0, 0) for j in range(ndim)]
        grad_list.append(np.pad(diff, pad_config, mode='edge'))

    return np.stack(grad_list, axis=0)

def divergence_nd(p, spacing=None):
    """
    Compute the divergence of a vector field using backward finite differences.

    Parameters
    ----------
    p : np.ndarray
        Vector field of shape (n_axes, ...), where n_axes is the number of spatial directions.

    spacing : tuple of float, optional
        Spacing for each spatial axis.

    Returns
    -------
    div : np.ndarray
        Scalar field with same shape as a single component of p.
    """
    if p.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions: (n_axes, ...).")

    n_axes = p.shape[0]
    ndim = p.ndim  # includes direction axis

    if spacing is None:
        spacing = [1.0] * n_axes
    elif len(spacing) != n_axes:
        raise ValueError(f"Expected spacing for {n_axes} axes, got {len(spacing)}")

    div = np.zeros_like(p[0])

    for i in range(n_axes):
        slicer_curr, slicer_prev  = [slice(None)] * (ndim - 1), [slice(None)] * (ndim - 1)
        slicer_curr[i], slicer_prev[i] = slice(1, None), slice(0, -1)

        diff = np.zeros_like(p[i])
        diff[tuple(slicer_curr)] = (p[i][tuple(slicer_curr)] - p[i][tuple(slicer_prev)]) / spacing[i]

        div += diff

    return div

@profiler.decorator()
def laplacian_nd(u, spacing=None, channel_axis=None, include_channel_gradient=False):
    """
    Compute the Laplacian of an n-dimensional array using divergence of its gradient.

    Parameters
    ----------
    u : np.ndarray
        Input array. Can be:
        - (H, W)
        - (Z, Y, X)
        - (H, W, C) or (Z, Y, X, C)

    spacing : tuple of float, optional
        Physical spacing along each spatial axis. If None, unit spacing is used.

    channel_axis : int or None, optional
        Axis representing channels to exclude from spatial gradient.
        Default is None (no axis excluded).

    include_channel_gradient : bool, optional
        Whether to compute the gradient along the channel axis. Default is False.

    Returns
    -------
    laplace_u : np.ndarray
        Laplacian of u, with same shape as u.
    """
    grad_u = gradient_nd(u, spacing=spacing, channel_axis=channel_axis, include_channel_gradient=include_channel_gradient)
    
    return divergence_nd(grad_u, spacing=spacing)

def mse(u_truth, u_estim):
    """
    Compute the Mean Squared Error (MSE) between two arrays.

    Parameters
    ----------
    u_truth : np.ndarray
        Ground truth reference array.
    u_estim : np.ndarray
        Estimated or reconstructed array.

    Returns
    -------
    float
        Mean Squared Error between the two arrays.

    Raises
    ------
    ValueError
        If the input arrays do not have the same shape.
    """
    u_truth = np.asarray(u_truth)
    u_estim = np.asarray(u_estim)

    if u_truth.shape != u_estim.shape:
        raise ValueError("Input arrays must have the same shape.")

    return np.mean((u_truth - u_estim) ** 2)

def psnr(u_truth, u_estim, max_intensity=1.0):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters
    ----------
    u_truth : np.ndarray
        Ground truth reference image. Shape: (H, W) or (H, W, C).
    u_estim : np.ndarray
        Estimated or reconstructed image. Same shape as u_truth.
    max_intensity : float, optional
        Maximum possible intensity value in the images.
        Use 1.0 for normalized images, 255.0 for 8-bit images.
        Default is 1.0.

    Returns
    -------
    float
        PSNR value in decibels (dB). Returns `inf` if images are identical.

    Raises
    ------
    ValueError
        If the input images do not have the same shape.
    """
    u_truth = np.asarray(u_truth, dtype=np.float64)
    u_estim = np.asarray(u_estim, dtype=np.float64)

    if u_truth.shape != u_estim.shape:
        raise ValueError("Input images must have the same shape.")

    mse_value = np.mean((u_truth - u_estim) ** 2)
    if mse_value == 0:
        return float("inf")

    return 20 * np.log10(max_intensity) - 10 * np.log10(mse_value)

def convolve_general(f, kernel, channel_axis=None, **kwargs):
    """
    Apply a 2D convolution to an image, slice by slice if multi-channel.

    Parameters
    ----------
    f : np.ndarray
        Input image, 2D (grayscale) or ND (multi-channel or multi-slice).
    kernel : np.ndarray
        2D convolution kernel.
    channel_axis : int or None, optional
        Axis representing channels or slices. If None, apply directly to entire image.
    **kwargs : dict
        Additional parameters passed to scipy.signal.convolve2d (e.g. mode, boundary).

    Returns
    -------
    np.ndarray
        Convolved image, same shape as input.

    Raises
    ------
    ValueError
        If input image or kernel have incompatible dimensions.
    """
    if kernel.ndim != 2:
        raise ValueError("The convolution kernel must be 2D.")
    if f.ndim < 2:
        raise ValueError("Input image must be at least 2D.")

    def convolve_single(channel):
        return convolve2d(channel, kernel, mode=kwargs.get("mode", "same"), boundary=kwargs.get("boundary", "symm"))

    return process_image_general(f, function=convolve_single, channel_axis=channel_axis)

def gaussian_kernel_2d(size=None, sigma=1.0, truncate=3.0, normalized=True, symmetry=True,
                       dtype=np.float32, visualize=False, angle=0):
    """
    Generate a 2D oriented Gaussian kernel with optional normalization and visualization.

    Parameters
    ----------
    size : int or None, optional
        Size of the kernel. If None, computed from truncate and sigma.
    sigma : float
        Standard deviation.
    truncate : float, optional
        Truncation threshold (default 3.0).
    normalized : bool, optional
        Normalize to sum=1 if True.
    symmetry : bool, optional
        Enforce symmetry of kernel.
    dtype : np.dtype
        Output type.
    visualize : bool
        Show kernel using matplotlib.
    angle : float, optional
        Rotation angle in degrees (default 0).

    Returns
    -------
    kernel : np.ndarray
        2D rotated Gaussian kernel.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be a positive number.")

    if size is None:
        size = int(2 * np.floor(truncate * sigma) + 1)

    if not isinstance(size, int) or size < 1 or size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd integer.")

    # Coordinate grid
    half = size // 2
    x = np.arange(-half, half + 1)
    X, Y = np.meshgrid(x, x)

    # Convert angle to radians
    theta = np.deg2rad(angle)

    # Rotate coordinates
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

    # Compute oriented Gaussian
    variance = sigma ** 2
    kernel = np.exp(-(X_rot**2 + Y_rot**2) / (2 * variance))
    kernel /= (2 * np.pi * variance)

    if normalized:
        kernel /= kernel.sum()

    if symmetry:
        kernel = (kernel + np.flipud(np.fliplr(kernel))) / 2

    kernel = kernel.astype(dtype)

    if visualize:
        plt.imshow(kernel)
        plt.title(f"Gaussian kernel (σ={sigma}, angle={angle}°)")
        plt.colorbar()
        plt.axis("off")
        plt.show()

    return kernel

def gaussian_kernel_nd(size, sigma):
    """
    Generate a normalized N-dimensional Gaussian kernel.

    Parameters
    ----------
    size : int or tuple of int
        Size of the kernel along each dimension.
        If int, the same size is used for all dimensions.
        Must be odd and positive.
    sigma : float or tuple of float
        Standard deviation along each axis.
        If float, the same sigma is used for all dimensions.

    Returns
    -------
    kernel : np.ndarray
        N-dimensional Gaussian kernel, normalized to sum to 1.

    Raises
    ------
    ValueError
        If size is not odd or if sigma is not positive.
    """
    # Normalize inputs
    if isinstance(size, int):
        size = (size,)
    if isinstance(sigma, (float, int)):
        sigma = (sigma,) * len(size)

    if len(size) != len(sigma):
        raise ValueError("size and sigma must have the same number of dimensions.")

    for s in size:
        if s <= 0 or s % 2 == 0:
            raise ValueError("Each dimension size must be a positive odd integer.")
    for s in sigma:
        if s <= 0:
            raise ValueError("Each sigma must be positive.")

    # Create meshgrid
    ranges = [np.arange(-(s // 2), s // 2 + 1) for s in size]
    grids = np.meshgrid(*ranges, indexing='ij')
    
    # Compute the Gaussian function
    kernel = np.ones_like(grids[0], dtype=np.float64)
    for g, s in zip(grids, sigma):
        kernel *= np.exp(-(g**2) / (2 * s**2))
    
    # Normalize
    kernel /= (np.sqrt(2 * np.pi) ** len(size)) * np.prod(sigma)
    kernel /= kernel.sum()

    return kernel

def gaussian_filter_separable_nd(image, size, sigma, channel_axis=None, **kwargs):
    """
    Apply a separable N-dimensional Gaussian filter using 1D convolutions.

    Parameters
    ----------
    image : np.ndarray
        Input image or volume (2D, 3D, ND with optional channels).
    size : int or tuple of int
        Kernel size for each spatial axis. Must be odd. If None, computed from sigma and truncate.
    sigma : float or tuple of float
        Standard deviation for each spatial axis.
    channel_axis : int or None, optional
        Axis to exclude from filtering (e.g., color channels).
    **kwargs : dict
        Additional parameters for convolve1d (e.g., mode, cval).

    Returns
    -------
    np.ndarray
        Filtered image with same shape as input.

    Raises
    ------
    ValueError
        If size/sigma values are invalid or mismatched.
    """
    import numpy as np
    from scipy.ndimage import convolve1d

    image = np.asarray(image, dtype=np.float64)
    ndim = image.ndim
    axes = list(range(ndim))

    if channel_axis is not None:
        if channel_axis < 0:
            channel_axis += ndim
        spatial_axes = [ax for ax in axes if ax != channel_axis]
    else:
        spatial_axes = axes

    # Handle flexible input
    if isinstance(sigma, (int, float)):
        sigma = [sigma] * len(spatial_axes)

    if size is None:
        size = [int(2 * np.floor(3 * s) + 1) for s in sigma]  # default truncate=3
    elif isinstance(size, int):
        size = [size] * len(spatial_axes)

    if len(size) != len(spatial_axes) or len(sigma) != len(spatial_axes):
        raise ValueError("Size and sigma must match number of spatial axes.")

    filtered = image.copy()

    for ax, sz, sig in zip(spatial_axes, size, sigma):
        if sz <= 0 or sz % 2 == 0:
            raise ValueError("Each kernel size must be a positive odd integer.")
        if sig <= 0:
            raise ValueError("Each sigma must be a positive value.")

        half = sz // 2
        x = np.arange(-half, half + 1)
        g = np.exp(-(x ** 2) / (2 * sig ** 2))
        g /= g.sum()

        filtered = convolve1d(filtered, g, axis=ax, **kwargs)

    return filtered


def grad_edge(u, eta=None, channel_axis=None, as_float=False, dtype="auto"):
    """
    Detect edges in an image or volume using gradient magnitude and a threshold.

    Parameters
    ----------
    u : np.ndarray
        Input image or volume (e.g. 2D grayscale, 3D RGB, or multi-channel volume).
    eta : float or str or None
        Threshold to detect edges:
        - float > 0: used directly
        - "otsu" or None: estimated automatically using Otsu's method
    channel_axis : int or None, optional
        Axis to treat as channels. If specified, applies detection per slice.
    as_float : bool, optional
        If True, converts output to float32 for display (default: False).
    dtype : str, optional
        Output dtype ("auto", "bool", "float32", "uint8", etc.).
        If "auto", uses float32 if as_float is True, otherwise bool.

    Returns
    -------
    np.ndarray
        Binary or scaled edge map with the same shape as input.

    Raises
    ------
    ValueError
        If input is invalid or threshold is not positive.
    """
    if not isinstance(u, np.ndarray) or u.ndim < 2:
        raise ValueError("Input must be a valid array with at least 2 dimensions.")

    def edge_detector_per_channel(slice_):
        grad = gradient_nd(slice_, include_channel_gradient=False)
        magnitude = np.sqrt(np.sum(grad ** 2, axis=0))
        threshold = threshold_otsu(magnitude) if eta is None or eta == "otsu" else eta
        if threshold <= 0:
            raise ValueError("Otsu estimated a non-positive threshold. Image might be empty.")
        return magnitude > threshold

    edges = process_image_general(u, function=edge_detector_per_channel, channel_axis=channel_axis)

    # Auto conversion logic
    if dtype == "auto":
        return edges.astype(np.float32) if as_float else edges
    else:
        return edges.astype(dtype)

def change_sign(u, channel_axis=None, return_magnitude=False, as_float=False, dtype="auto"):
    """
    Detect sign changes between adjacent pixels in all spatial directions.

    Parameters
    ----------
    u : np.ndarray
        Input array (2D, 3D, RGB, hyperspectral, etc.).
    channel_axis : int or None, optional
        Channel dimension to preserve. If given, function is applied per slice.
    return_magnitude : bool, optional
        If True, also returns the magnitude of change (abs difference).
    as_float : bool, optional
        If True, converts boolean result to float32 (0.0 or 1.0).
    dtype : str, optional
        Output dtype: "auto", "bool", "float32", "uint8", etc.

    Returns
    -------
    np.ndarray or tuple
        If return_magnitude is False: a binary or scaled sign change map.
        If return_magnitude is True: (sign_map, magnitude_map).
    """
    
    if not isinstance(u, np.ndarray) or u.ndim < 2:
        raise ValueError("Input must be at least a 2D array.")

    def detect_sign_change(channel):
        ndim = channel.ndim
        bool_map = np.zeros_like(channel, dtype=bool)
        mag_map = np.zeros_like(channel, dtype=np.float32)

        axes = list(range(ndim))

        for axis in axes:
            slicer_curr, slicer_prev = [slice(None)] * ndim, [slice(None)] * ndim
            slicer_curr[axis], slicer_prev[axis] = slice(1, None) , slice(0, -1)
            a, b = channel[tuple(slicer_curr)], channel[tuple(slicer_prev)]

            sign_change = (a * b) <= 0
            delta = np.abs(a) - np.abs(b)

            target = [slice(None)] * ndim
            target[axis] = slice(0, channel.shape[axis] - 1)

            bool_map[tuple(target)] |= sign_change & (delta >= 0)
            
            if return_magnitude:
                mag_map[tuple(target)] = np.abs(delta) * bool_map[tuple(target)]
         
        return (bool_map, mag_map) if return_magnitude else bool_map


    # Appel modulaire et propre via process_image_general
    result = process_image_general(u, function=detect_sign_change, channel_axis=channel_axis, return_tuple=return_magnitude,)

    if return_magnitude:
        sign_map, mag_map = result
        sign_map = sign_map.astype(np.float32) if (as_float and dtype == "auto") else sign_map.astype(dtype if dtype != "auto" else bool)
        return sign_map, mag_map
    else:
        result = result.astype(np.float32) if (as_float and dtype == "auto") else result.astype(dtype if dtype != "auto" else bool)
        return result

def lap_edge(u, spacing=None, channel_axis=None, as_float=False, dtype="auto"):
    """
    Detect edges in an image using the Laplacian and zero-crossings.

    Parameters
    ----------
    u : np.ndarray
        Input array (2D grayscale, RGB, multichannel or 3D volume).
    spacing : tuple of float, optional
        Physical spacing along each spatial axis. If None, spacing = 1.
    channel_axis : int or None, optional
        Axis representing channels (e.g. for RGB or hyperspectral).
    as_float : bool, optional
        If True, returns float32 for visualization.
    dtype : str, optional
        Output data type ("auto", "float32", "uint8", etc.).

    Returns
    -------
    np.ndarray
        Binary or float edge map.
    """
    if not isinstance(u, np.ndarray) or u.ndim < 2:
        raise ValueError("Input must be a valid 2D or higher-dimensional array.")

    def lap_edge_channel(channel):
        lap = laplacian_nd(channel, spacing=spacing, channel_axis=None, include_channel_gradient=False)
        return change_sign(lap, return_magnitude=False)

    result = process_image_general(u, function=lap_edge_channel, channel_axis=channel_axis, return_tuple=False)

    if dtype == "auto":
        return result.astype(np.float32) if as_float else result
    else:
        return result.astype(dtype)
    
def combined_edge(u, eta=None, spacing=None, channel_axis=None,mode="and", alpha=0.5, threshold="auto", as_float=False, dtype="auto"):
    """
    Detect contours by combining gradient and Laplacian-based edge detection.

    Parameters
    ----------
    u : np.ndarray
        Input image or volume (2D, RGB, hyperspectral, 3D, etc.).
    eta : float or str, optional
        Threshold for gradient magnitude. Can be float > 0 or "otsu" or None.
    spacing : tuple of float, optional
        Physical spacing between elements.
    channel_axis : int or None
        Axis for color/spectral channels.
    mode : str
        Fusion method: "and", "or", or "weighted".
    alpha : float
        Weight for gradient edge in "weighted" mode (0 ≤ alpha ≤ 1).
    threshold : float or str
        Threshold for final fusion map in "weighted" mode.
        If "auto", uses Otsu.
    as_float : bool
        If True, returns float32 (for display).
    dtype : str
        Output dtype: "auto", "float32", "bool", "uint8", etc.

    Returns
    -------
    np.ndarray
        Combined edge map.
    """
    if mode not in {"and", "or", "weighted"}:
        raise ValueError("Mode must be 'and', 'or', or 'weighted'.")

    if mode == "weighted" and not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1.")

    def fused_edge_channel(channel):
        grad = grad_edge(channel, eta=eta, channel_axis=None, as_float=True)
        lap = lap_edge(channel, spacing=spacing, channel_axis=None, as_float=True)

        if mode == "and":
            return (grad > 0) & (lap > 0)

        elif mode == "or":
            return (grad > 0) | (lap > 0)

        elif mode == "weighted":
            fusion = alpha * grad + (1 - alpha) * lap
            if threshold == "auto":
                thresh_val = threshold_otsu(fusion)
            else:
                thresh_val = threshold
            return fusion > thresh_val

    result = process_image_general(u, function=fused_edge_channel, channel_axis=channel_axis, return_tuple=False)

    if dtype == "auto":
        return result.astype(np.float32) if as_float else result
    else:
        return result.astype(dtype)
    
def marr_hildreth(u, eta="otsu", spacing=None, channel_axis=None, as_float=False, dtype="auto"):
    """
    Marr-Hildreth edge detection using zero-crossings and gradient magnitude.

    Parameters
    ----------
    u : np.ndarray
        Input image or volume (2D, 3D, RGB, hyperspectral).
    eta : float or "otsu"
        Threshold on gradient magnitude. If "otsu", computed automatically.
    spacing : tuple of float, optional
        Spacing between pixels/voxels along spatial axes.
    channel_axis : int or None, optional
        Axis of color or spectral channels.
    as_float : bool, optional
        If True, returns float32 (for display).
    dtype : str
        Output dtype. "auto" uses float32 if as_float, else bool.

    Returns
    -------
    np.ndarray
        Binary or float32 edge map.
    """
    if not (isinstance(eta, (int, float)) and eta > 0) and eta != "otsu":
        raise ValueError("eta must be a positive number or 'otsu'.")

    def channel_marr_hildreth(channel):
        lap = laplacian_nd(channel, spacing=spacing, channel_axis=None)
        grad = gradient_nd(channel, spacing=spacing, channel_axis=None)
        grad_norm = norm(grad, axis=0) 

        if eta == "otsu":
            local_eta = threshold_otsu(grad_norm)
        else:
            local_eta = eta

        bool_map = change_sign(lap) & (grad_norm > local_eta)
        return bool_map.astype(np.float32) if as_float else bool_map

    result = process_image_general(u, function=channel_marr_hildreth, channel_axis=channel_axis, return_tuple=False)

    if dtype == "auto":
        return result.astype(np.float32) if as_float else result
    else:
        return result.astype(dtype)

def g_exp(xi, alpha=1.0, channel_axis=None, as_float=False, dtype="auto"):
    """
    Apply exponential low-pass filtering to suppress high variations.

    Parameters
    ----------
    xi : np.ndarray
        Input image, volume, or multichannel data (1D, 2D, 3D or more).
    alpha : float, optional
        Smoothing parameter. Must be > 0. Higher = less smoothing. Default is 1.0.
    channel_axis : int or None, optional
        Axis corresponding to channels (color, spectrum...). If set, filtering is done per slice.
    as_float : bool, optional
        If True, output is returned as float32 in [0, 1].
    dtype : str, optional
        Output dtype. Use "auto" for float32 if as_float, otherwise keep original type.

    Returns
    -------
    np.ndarray
        Filtered output with same shape as input.

    Raises
    ------
    ValueError
        If alpha is not strictly positive.
    """
    if not isinstance(alpha, (float, int)) or alpha <= 0:
        raise ValueError("Parameter 'alpha' must be a positive number.")

    def exp_filter(channel):
        return np.exp(- (channel / alpha) ** 2)

    result = process_image_general(xi, function=exp_filter, channel_axis=channel_axis, return_tuple=False)

    if dtype == "auto":
        return result.astype(np.float32) if as_float else result
    else:
        return result.astype(dtype)
    
def g_PM(xi, alpha=1.0, channel_axis=None, as_float=False, dtype="auto"):
    """
    Apply Perona-Malik (high-pass) edge-preserving filtering.

    Parameters
    ----------
    xi : np.ndarray
        Input image, volume, or multichannel data (1D, 2D, 3D or more).
    alpha : float, optional
        Controls edge sensitivity. Higher alpha = less filtering. Must be > 0.
    channel_axis : int or None, optional
        Channel axis (color, spectral, etc.). If set, filtering is applied per channel slice.
    as_float : bool, optional
        If True, output is returned as float32 in [0, 1].
    dtype : str, optional
        Output dtype. Use "auto" for float32 if as_float, otherwise keep original type.

    Returns
    -------
    np.ndarray
        Filtered output with same shape as input.

    Raises
    ------
    ValueError
        If alpha is not strictly positive.
    """
    if not isinstance(alpha, (float, int)) or alpha <= 0:
        raise ValueError("Parameter 'alpha' must be a positive number.")

    def pm_filter(channel):
        return 1 / np.sqrt((channel / alpha) ** 2 + 1)

    result = process_image_general(xi, function=pm_filter, channel_axis=channel_axis, return_tuple=False)

    if dtype == "auto":
        return result.astype(np.float32) if as_float else result
    else:
        return result.astype(dtype)

def Perona_Malik(f, dt=0.1, K=20, alpha=1.0, g=g_PM, spacing=None, channel_axis=None, return_evolution=False, as_float=False, verbose=True):
    """
    Apply Perona-Malik anisotropic diffusion to smooth an image or volume.

    Parameters
    ----------
    f : np.ndarray
        Input image, volume or multichannel data (2D, 3D, ND).
    dt : float
        Time step for the diffusion. Must be > 0.
    K : int
        Number of diffusion iterations. Must be > 0.
    alpha : float
        Diffusion contrast parameter. Controls edge preservation.
    g : callable
        Edge-stopping function, e.g. g_PM or g_exp.
    spacing : tuple of float, optional
        Pixel/voxel spacing along each spatial axis. Default assumes unit spacing.
    channel_axis : int or None, optional
        Axis corresponding to color or spectral channels. If set, the diffusion is applied per channel.
    return_evolution : bool, optional
        If True, returns a list of intermediate states after each iteration.
    as_float : bool, optional
        If True, forces computation in float32 for stability and visualization.
    verbose : bool, optional
        If True, displays a progress bar during iterations using tqdm.

    Returns
    -------
    np.ndarray or list of np.ndarray
        Final denoised image, or list of intermediate results if return_evolution=True.

    Raises
    ------
    ValueError
        If any parameter is invalid.
    """


    if dt <= 0:
        raise ValueError("Parameter 'dt' must be positive.")
    if K <= 0:
        raise ValueError("Parameter 'K' must be a positive integer.")
    if alpha <= 0:
        raise ValueError("Parameter 'alpha' must be strictly positive.")
    if not isinstance(f, np.ndarray) or f.ndim < 2:
        raise ValueError("Input must be a 2D or higher array.")

    def pm_channel(channel):
        u = channel.astype(np.float32).copy() if as_float else channel.copy()
        evolution = [u.copy()] if return_evolution else None

        for _ in tqdm(range(K), desc="Perona-Malik Algorithm", disable=not verbose):
            grad_u = gradient_nd(u, spacing=spacing, channel_axis=None)
            norm_grad_u = norm(grad_u, axis=0)
            # print("Mean grad_norm_u", norm_grad_u.mean(), "Max grad_norm_u", norm_grad_u.max(), "Min grad_norm_u", norm_grad_u.min())
            weight = g(norm_grad_u, alpha=alpha)
            # print("Mean weight", weight.mean(), "Max weight", weight.max(), "Min weight", weight.min())
            u += dt * divergence_nd(weight * grad_u, spacing=spacing)
            if return_evolution:
                evolution.append(u.copy())

        return evolution if return_evolution else u

    return process_image_general(f, function=pm_channel, channel_axis=channel_axis, return_tuple=return_evolution)

def Perona_Malik_enhanced(f, dt=0.1, K=20, alpha=1.0, s=1.0, g=g_PM, spacing=None, channel_axis=None, return_evolution=False,
                          as_float=False, verbose=True):
    """
    Apply enhanced Perona-Malik anisotropic diffusion with Gaussian pre-smoothing.

    This variant first smooths the input using a Gaussian kernel before computing the gradient
    to improve robustness to noise. The evolution is driven by the smoothed gradient while
    diffusion is applied on the original image.

    Parameters
    ----------
    f : np.ndarray
        Input image, volume, or multichannel data (2D, 3D, ND).
    dt : float
        Time step for the diffusion. Must be > 0.
    K : int
        Number of diffusion iterations. Must be > 0.
    alpha : float
        Diffusion contrast parameter. Controls edge preservation.
    s : float
        Standard deviation for the Gaussian kernel (pre-smoothing). Must be > 0.
    g : callable
        Edge-stopping function, e.g. g_PM or g_exp.
    spacing : tuple of float, optional
        Physical spacing along each spatial axis. Default assumes unit spacing.
    channel_axis : int or None, optional
        Axis representing color or spectral channels. If set, the diffusion is applied per channel.
    return_evolution : bool, optional
        If True, returns a list of intermediate states after each iteration.
    as_float : bool, optional
        If True, computation is done in float32 (recommended for stability).
    verbose : bool, optional
        If True, shows a tqdm progress bar.

    Returns
    -------
    np.ndarray or list of np.ndarray
        Final diffused image or volume. If return_evolution=True, returns the list of all intermediate results.

    Raises
    ------
    ValueError
        If any parameter is invalid.
    """


    if dt <= 0:
        raise ValueError("Parameter 'dt' must be positive.")
    if K <= 0:
        raise ValueError("Parameter 'K' must be a positive integer.")
    if alpha <= 0:
        raise ValueError("Parameter 'alpha' must be strictly positive.")
    if s <= 0:
        raise ValueError("Parameter 's' must be strictly positive.")
    if not isinstance(f, np.ndarray) or f.ndim < 2:
        raise ValueError("Input must be a 2D or higher array.")

    # Precompute Gaussian kernel for all iterations
    gaussian = gaussian_kernel_2d(sigma=s)

    def pm_enhanced_channel(channel):
        u = channel.astype(np.float32).copy() if as_float else channel.copy()
        evolution = [u.copy()] if return_evolution else None

        for _ in tqdm(range(K), desc="Perona-Malik Enhanced Algorithm", disable=not verbose):
            grad_u = gradient_nd(u, spacing=spacing, channel_axis=None)
            # print("Mean grad_u", grad_u.mean(), "Max grad_u", grad_u.max(), "Min grad_u", grad_u.min())
            conv = convolve_general(u, kernel=gaussian, channel_axis=None)
            # print("Mean conv", conv.mean(), "Max conv", conv.max(), "Min conv", conv.min())
            grad_conv = gradient_nd(conv, spacing=spacing, channel_axis=None)
            # print("Mean grad_conv", grad_conv.mean(), "Max grad_conv", grad_conv.max(), "Min grad_conv", grad_conv.min())
            norm_grad_conv = norm(grad_conv, axis=0)
            # print("Mean grad_norm", norm_grad_conv.mean(), "Max grad_norm", norm_grad_conv.max(), "Min grad_norm", norm_grad_conv.min())
            weight = g(norm_grad_conv, alpha=alpha)
            # print("Mean weight", weight.mean(), "Max weight", weight.max(), "Min weight", weight.min())
            u += dt * divergence_nd(weight * grad_u, spacing=spacing)      
            # print("Mean u", u.mean(), "Max u", u.max(), "Min u", u.min(),"\n")   
            if return_evolution:
                evolution.append(u.copy())

        return evolution if return_evolution else u

    return process_image_general(f, function=pm_enhanced_channel, channel_axis=channel_axis, return_tuple=return_evolution)

def search_opt_general(func, u_truth, param_grid, metric, func_args=None, sub_params_key=None, sub_param_grid=None, return_results=False, verbose=True):
    """
    Perform exhaustive parameter search to optimize a given function with optional sub-parameters.

    Parameters
    ----------
    func : callable
        Function to optimize. Must return output matching the shape of u_truth.
    u_truth : np.ndarray
        Ground-truth data to evaluate against.
    param_grid : dict
        Main parameters to test (grid search). Keys are param names, values are lists of values.
    metric : callable
        Function used to score the result. Must accept (truth, prediction) and return a float.
    func_args : dict, optional
        Fixed arguments passed to func. Default is None.
    sub_param_grid : dict, optional
        Secondary parameter space (e.g. for internal blocks like 'prox'). Default is None.
    return_results : bool, optional
        If True, also returns the output associated with the best score. Default is False.
    verbose : bool, optional
        If True, print/log progress and errors.

    Returns
    -------
    best_params : dict or tuple
        Best combination of parameters (and sub-parameters if any).
    best_score : float
        Best score achieved.
    score_map : pd.DataFrame
        Score table of all combinations.
    best_result : np.ndarray, optional
        Returned only if return_results=True.
    """
    if func_args is None:
        func_args = {}
    if not param_grid:
        raise ValueError("param_grid cannot be empty.")

    main_keys = list(param_grid.keys())
    main_vals = list(param_grid.values())
    main_combos = list(itertools.product(*main_vals))

    use_subgrid = sub_param_grid is not None and bool(sub_param_grid)
    sub_keys = list(sub_param_grid.keys()) if use_subgrid else []
    sub_vals = list(sub_param_grid.values()) if use_subgrid else []
    sub_combos = list(itertools.product(*sub_vals)) if use_subgrid else [None]

    score_data = []
    best_score = -np.inf
    best_params = None
    best_output = None

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

                    score = metric(u_truth, result)

                    row = (current_main.copy(), current_sub.copy(), score) if use_subgrid else (current_main.copy(), score)
                    score_data.append(row)

                    if score > best_score:
                        best_score = score
                        best_params = (current_main.copy(), current_sub.copy()) if use_subgrid else current_main.copy()
                        best_output = result.copy()

                except Exception as e:
                    if verbose:
                        print(f"[!] Error with params={current_main}, prox={current_sub} → {e}")

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
    
def add_gaussian_noise(image, sigma, random=False, clip=False, clip_range=(0, 1), seed=None, return_noise=False, channel_axis=None):
    """
    Add Gaussian noise to an image, volume, or multichannel data.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D, 3D, ND).
    sigma : float or sequence of float
        Noise standard deviation. If `channel_axis` is not None and `sigma` is a sequence,
        each channel uses its own sigma.
    random : bool, optional
        If True, use random noise level in [0, sigma] per channel.
    clip : bool, optional
        If True, clip values after noise.
    clip_range : tuple, optional
        (min, max) values to clip to.
    seed : int or None, optional
        Fix the random seed for reproducibility.
    return_noise : bool, optional
        If True, also return the noise array.
    channel_axis : int or None, optional
        Axis of channels. If None, noise is applied uniformly on full image.

    Returns
    -------
    noisy_image : np.ndarray
        Image with Gaussian noise.
    noise : np.ndarray, optional
        The noise that was added (if return_noise=True).

    Raises
    ------
    ValueError
        If image shape or sigma values are inconsistent.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy array.")
    if image.ndim < 2:
        raise ValueError("Image must be at least 2D.")

    if seed is not None:
        np.random.seed(seed)

    shape = image.shape
    noise = np.zeros_like(image, dtype=np.float32)

    if channel_axis is not None:
        if channel_axis < 0:
            channel_axis += image.ndim
        num_channels = image.shape[channel_axis]

        if isinstance(sigma, (int, float)):
            sigma = [sigma] * num_channels
        elif len(sigma) != num_channels:
            raise ValueError("Length of sigma must match number of channels.")

        for c in range(num_channels):
            level = np.random.uniform(0, sigma[c]) if random else sigma[c]
            slicer = [slice(None)] * image.ndim
            slicer[channel_axis] = c
            noise[tuple(slicer)] = np.random.randn(*image[tuple(slicer)].shape) * level
    else:
        level = np.random.uniform(0, sigma) if random else sigma
        noise = np.random.randn(*shape) * level

    noisy_image = image + noise

    if clip:
        noisy_image = np.clip(noisy_image, *clip_range)

    return (noisy_image, noise) if return_noise else noisy_image



# ========================================================================================================================================================================================================================================================================================================================================================================
# Test
# ========================================================================================================================================================================================================================================================================================================================================================================
from skimage.exposure import rescale_intensity
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt

# cwd = Path.cwd().parent.parent.parent / "Bureau" /"STAGE FARIUS" /"Images" # B
cwd = Path.cwd().parent.parent / "Images" # H

images_path = []

for img_path in cwd.rglob("*.png"):
    images_path.append(str(img_path))

# rand = np.random.randint(0, len(images_path))

rand = 139

# rand = 1928
    
# image = Image.open(images_path[rand]).convert('RGB') 

layout_cfg = LayoutConfig(
    layout_name="HWC",
    layout_framework="numpy",
    layout_ensured_name="HWC",
)
global_cfg = GlobalConfig( 
    framework="numpy",
    output_format="numpy",   
)

# ====[ Init ImageIO ]====
io = ImageIO(
    layout_cfg=layout_cfg,
    global_cfg=global_cfg,
)

image_array = io.read_image(images_path[rand], framework="numpy") 

image_noise = add_gaussian_noise(image_array, sigma=0.1, random=False, channel_axis=2, clip=True)

image_noise = io.track(image_array).copy_to(image_noise).get()

# gradient = gradient_nd(image_array, spacing=(1, 1), channel_axis=None)

# laplacian = laplacian_nd(image_array, spacing=(1, 1), channel_axis=None)

# print(laplacian.shape)

# plt.imshow(gradient[*[1, slice(None), slice(None)]], cmap='gray')
# plt.show()

# plt.imshow(laplacian[*[slice(None), slice(None)]], cmap='seismic')
# plt.show()

# plt.imshow(rescale_intensity(laplacian[*[slice(None), slice(None)]], in_range='image', out_range=(-1, 1)), cmap='seismic')
# plt.show()

# grad_edge_image = grad_edge(image_array, eta=None, channel_axis=2, as_float=True)

# signs, magnitude = change_sign(image_array, channel_axis=2, as_float=True, return_magnitude=True)

# lap_edge_image = lap_edge(image_array, channel_axis=2, as_float=True)

# marr_hildreth_image = marr_hildreth(image_array, channel_axis=2, as_float=True)

# g_exp_image = g_exp(image_array, alpha=0.5, channel_axis=2, as_float=True)

# g_PM_image = g_PM(image_array, alpha=0.5, channel_axis=2, as_float=True)

# PM_image = Perona_Malik(image_noise, dt=2e-2, K=25, alpha=3e-2, g=g_PM, channel_axis=2, return_evolution=False) # (dt=2.5e-2, K=50, alpha=5e-2)

PM_image = Perona_Malik_enhanced(image_noise, dt=2e-2, K=25, alpha=4e-2, s=1.0, g=g_PM, channel_axis=2, return_evolution=False)

# PM_image = Perona_Malik_enhanced(image_noise, dt=4e-2, K=25, alpha=1e-1, s=1.0, g=g_PM, channel_axis=2, return_evolution=False)

# kernel = gaussian_kernel_2d(sigma=7.0, truncate=3.0, visualize=True, angle=0)

# plt.imshow(image_noise, cmap='gray')
# plt.show()



# edges_g_and_l = combined_edge(image_array,  channel_axis=2, mode="and", as_float=True)
# edges_g_or_l = combined_edge(image_array,  channel_axis=2, mode="or", as_float=True)
# edges_g_w_l = combined_edge(image_array,  channel_axis=2, mode="weighted", alpha=0.5, threshold="auto", as_float=True)

# show_plane(a, edges_g_and_l, title="Gradient & Laplacian Edge Image", cmap='gray')
# show_plane(b, edges_g_or_l, title="Gradient | Laplacian Edge Image", cmap='gray')        
# show_plane(c, edges_g_w_l, title="Gradient Weighted Edge Image", cmap='gray')

# Recherche des meilleurs paramètres pour la fonction Perona_Malik
# param_grid = {"alpha": [5e-2, 8e-2, 1.e-2], "dt": [2e-2, 3e-2, 4e-2], "K": [25, 50]} # Main parameters
# sub_grid = {} # No sub-parameters
# func_args = {"f": image_noise, "g": g_PM, "spacing": (1.0, 1.0), "channel_axis": 2, "return_evolution": False, "as_float": False} # Fixed arguments for the function

# best_params, best_score, table = search_opt_general(func=Perona_Malik, u_truth=image_array, param_grid=param_grid,
#     metric=psnr, func_args=func_args, sub_param_grid=sub_grid, return_results=False, verbose=True)


# Recherche des meilleurs paramètres pour la fonction Perona_Malik enhanced
# param_grid = {"alpha": [5e-2, 8e-2, 1.e-1], "dt": [2e-2, 3e-2, 4e-2]} # Main parameters
# sub_grid = {} # No sub-parameters
# func_args = {"f": image_noise, "g": g_PM, "spacing": (1.0, 1.0), "s": 1.0, "K":25,
#              "channel_axis": 2, "return_evolution": False, "as_float": False, "verbose": False} # Fixed arguments for the function

# best_params, best_score, table, PM_image = search_opt_general(func=Perona_Malik_enhanced, u_truth=image_array, param_grid=param_grid,
#     metric=psnr, func_args=func_args, sub_param_grid=sub_grid, return_results=True, verbose=True)

# print(best_params)
# print(table)

image_noise_metric = io.treat_image_and_add_metric(image_noise, image_array)

PM_image_metric = io.treat_image_and_add_metric(PM_image, image_array)

_, (a,b,c) = plt.subplots(1,3, figsize=(18,6))

show_plane(a, image_array, title="Ground Truth", cmap='gray')
show_plane(b, image_noise, title="Image_noised", cmap='gray')
show_plane(c, PM_image, title="Perona-Malik Image", cmap='gray')

print("PSNR AVANT PERONA MALIK : ", psnr(image_array, image_noise))
print("PSNR APRES PERONA MALIK : ", psnr(image_array, PM_image))

_, (a,b,c) = plt.subplots(1,3, figsize=(18,6))

show_plane(a, image_array, title="Ground Truth", cmap='gray')
show_plane(b, image_noise_metric, title="Image_noised", cmap='gray')
show_plane(c, PM_image_metric, title="Perona-Malik Image", cmap='gray')

# print(EMOJIS['success'] + " All tests passed!")

profiler.summary()