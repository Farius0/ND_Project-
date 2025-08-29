# ==================================================
# ===============  ND Tools & Plots  ===============
# ==================================================
from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple, Union
import math

import matplotlib.pyplot as plt
import numpy as np

import torch
from matplotlib.gridspec import GridSpec

ArrayNP = np.ndarray
ArrayTorch = torch.Tensor
ArrayLike = Union[ArrayNP, ArrayTorch]

#Public API
__all__ = [
    "colormap_picker",
    "show_plane",
    "plot_hist",
    "plot_images_and_hists",
    "plot_histogram_bins",
    "display_features",
    "save_features_grid",
    "search_opt_general",
]

# --------------------------- #
# ------- Colormaps --------- #
# --------------------------- #
def colormap_picker(feature_type: Optional[str]) -> str:
    """
    Select an appropriate colormap based on the type of image feature.

    Parameters
    ----------
    feature_type : str or None
        Name of the feature (e.g., "gradient", "entropy", "skewness").
        If None, a default colormap is returned.

    Returns
    -------
    str
        Name of the recommended colormap for visualizing the feature.
    """

    if not feature_type:
        return "viridis"
    ft = feature_type.lower()
    cmap_map = {
        # signed (diverging)
        "skewness": "PiYG",
        "gradient": "coolwarm",
        "laplacian": "coolwarm",
        "hessian": "coolwarm",
        "curvature": "RdBu",
        # positive (sequential)
        "entropy": "plasma",
        "spectral_entropy": "magma",
        "histogram": "viridis",
        "kurtosis": "cividis",
        "asm": "cividis",
        "homogeneity": "cividis",
        "glcm": "viridis",
        "fft": "plasma",
        # texture & filters
        "lbp": "viridis",
        "wavelet": "magma",
        "gabor": "magma",
        # pair textures
        "contrast": "inferno",
        "dissimilarity": "inferno",
        # basic stats
        "intensity": "gray",
        "mean": "Blues",
        "std": "Purples",
        "median": "Greens",
        "gaussian": "Greens",
        # morpho & ridge
        "ridge": "magma",
        "bandpass": "magma",
        # similarity & PCA
        "local_similarity": "inferno",
        "local_pca": "viridis",
        # edges
        "canny": "gray",
        "gradient_edge": "gray",
        "sobel_edge": "gray",
        "laplacian_edge": "gray",
        "marr_hildreth_edge": "gray",
        "sign_change_edge": "gray",
        "combined_edge": "gray",
    }
    if ft in cmap_map:
        return cmap_map[ft]
    if any(tok in ft for tok in ("edge", "canny")):
        return "gray"
    if any(tok in ft for tok in ("gradient", "laplacian", "hessian", "skew")):
        return "coolwarm"
    if any(tok in ft for tok in ("entropy", "histogram", "glcm", "fft", "pca")):
        return "viridis"
    return "viridis"

# --------------------------- #
# --------- Display --------- #
# --------------------------- #
def show_plane(
    ax,
    plane: ArrayLike,
    cmap: str = "auto",
    title: Optional[str] = None,
    colorbar: bool = False,
    norm: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    feature_type: Optional[str] = None,
    feature_index: Optional[int] = None,
    channel_index: Optional[int] = None,
    feature_position: str = "last",  # 'first' or 'last'
    channel_axis: Optional[int] = None,
    feature_axis: Optional[int] = None,
    as_rgb: Optional[bool] = None,
    postprocess: bool = True,
) -> None:
    """
    Display a 2D image or feature map on a Matplotlib axis.

    Supported input shapes
    -----------------------
    - (H, W)                 : grayscale 2D image
    - (H, W, F)              : multiple feature maps (select via feature_index)
    - (H, W, C)              : multichannel image (e.g., RGB if C==3 or as_rgb=True)
    - (H, W, C, F)           : multichannel with multiple features

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to display the image.
    plane : ArrayLike
        Input image or feature map (2D or ND with channels/features).
    cmap : str, optional
        Colormap to use. If "auto", selected based on `feature_type`.
    title : str, optional
        Title to display above the image.
    colorbar : bool, optional
        If True, display a colorbar alongside the image.
    norm : bool, optional
        If True, normalize the image to [0, 1] before display.
    vmin : float, optional
        Minimum display value (overrides automatic scaling).
    vmax : float, optional
        Maximum display value (overrides automatic scaling).
    feature_type : str, optional
        Type of feature (e.g., "gradient", "entropy") used to auto-select colormap.
    feature_index : int, optional
        Index of the feature channel to extract if features are present.
    channel_index : int, optional
        Index of the channel to extract if multiple channels are present.
    feature_position : str, optional
        Position of the feature axis: "last" (default) or "first".
    channel_axis : int, optional
        Axis index for channels if known (overrides shape-based inference).
    feature_axis : int, optional
        Axis index for features if known (overrides shape-based inference).
    as_rgb : bool, optional
        If True, treat input as RGB and display accordingly.
    postprocess : bool, optional
        If True, apply additional processing (e.g., type casting or scaling) before display.

    Returns
    -------
    None
        The image is rendered on the provided axis. Nothing is returned.
    """

    # Convert to NumPy (non-mutable)
    if isinstance(plane, torch.Tensor):
        plane = plane.detach().cpu().numpy()
    plane = np.asarray(plane)
    nd = plane.ndim

    # Feature-specific preprocessing
    def _feat_post(x: ArrayNP) -> ArrayNP:
        if not postprocess or not feature_type:
            return x
        ft = feature_type.lower()
        if ft in ("kurtosis", "skewness"):
            return np.clip(x, -20, 20)
        if ft in ("entropy", "spectral_entropy", "contrast", "dissimilarity"):
            return np.log1p(np.maximum(x, 0))
        if any(tok in ft for tok in ("gradient", "laplacian", "hessian")):
            return np.abs(x)
        return x

    # Helper: normalize a 2D array for display
    def _maybe_norm(x2d: ArrayNP) -> ArrayNP:
        if not norm:
            return x2d
        mn, mx = float(x2d.min()), float(x2d.max())
        if mx <= mn + 1e-12:
            return np.zeros_like(x2d)
        return (x2d - mn) / (mx - mn)

    # 4D: try to reorder to (H, W, C, F)
    if nd == 4:
        # Build permutation robustly
        order = [0, 1, 2, 3]  # current axes
        # Move channel to pos 2
        if channel_axis is not None:
            ca = channel_axis % 4
            order.remove(ca)
            order.insert(2, ca)
        # Move feature to pos 3
        if feature_axis is not None:
            fa = feature_axis % 4
            if fa in order:
                order.remove(fa)
            order.insert(3, fa)
        plane = np.transpose(plane, order)
        H, W, C, F = plane.shape

        # RGB route?
        rgb_auto = (C == 3 and feature_type is None and feature_index is None)
        rgb_mode = as_rgb if as_rgb is not None else rgb_auto
        if rgb_mode:
            fidx = 0 if (feature_index is None and feature_position == "first") else (F - 1 if feature_index is None else feature_index)
            fidx = int(np.clip(fidx, 0, F - 1))
            img = _feat_post(plane[:, :, :, fidx])
            ax.imshow(_maybe_norm(img))
            ax.set_axis_off()
            if title:
                ax.set_title(title)
            return

        # Otherwise, choose (channel, feature) slices
        fidx = 0 if (feature_index is None and feature_position == "first") else (F - 1 if feature_index is None else feature_index)
        fidx = int(np.clip(fidx, 0, F - 1))
        cidx = 0 if channel_index is None else int(np.clip(channel_index, 0, C - 1))
        plane2d = _feat_post(plane[:, :, cidx, fidx])

    elif nd == 3:
        H, W, D = plane.shape
        # If D==3, we might show RGB directly
        rgb_auto = (D == 3 and feature_type is None and feature_index is None)
        rgb_mode = as_rgb if as_rgb is not None else rgb_auto
        if rgb_mode:
            img = _feat_post(plane)
            ax.imshow(_maybe_norm(img))
            ax.set_axis_off()
            if title:
                ax.set_title(title)
            return
        # Otherwise select a single slice
        fidx = 0 if (feature_index is None and feature_position == "first") else (D - 1 if feature_index is None else feature_index)
        fidx = int(np.clip(fidx, 0, D - 1))
        plane2d = _feat_post(plane[:, :, fidx])

    elif nd == 2:
        plane2d = _feat_post(plane)

    else:
        raise ValueError(f"Unsupported array with ndim={nd}")

    # Colormap
    if cmap == "auto":
        cmap = colormap_picker(feature_type or (f"feat{feature_index}" if feature_index is not None else None))

    # Display (normalize only the displayed 2D slice)
    data2d = _maybe_norm(plane2d)
    im = ax.imshow(data2d, cmap=cmap, vmin=None if norm else vmin, vmax=None if norm else vmax)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_hist(
    ax,
    data: ArrayLike,
    title: Optional[str] = None,
    bins: int = 256,
    color: Optional[str] = None,
    log: bool = False,
    density: bool = False,
    range: Optional[Tuple[float, float]] = None,
    cumulative: bool = False,
) -> None:
    """
    Plot an intensity histogram of the input data on a Matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis object on which to plot the histogram.
    data : ArrayLike
        Input array of intensity values (1D or flattened 2D/3D).
    title : str, optional
        Title to display above the histogram.
    bins : int, optional
        Number of bins to use in the histogram. Default is 256.
    color : str, optional
        Color of the histogram bars.
    log : bool, optional
        If True, use logarithmic scale on the y-axis.
    density : bool, optional
        If True, normalize the histogram to show probability densities.
    range : tuple of float, optional
        Lower and upper range of the bins. If None, determined automatically.
    cumulative : bool, optional
        If True, plot a cumulative histogram.

    Returns
    -------
    None
        The histogram is drawn on the provided axis. Nothing is returned.
    """

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    flat = np.asarray(data).ravel()
    ax.hist(flat, bins=bins, color=color, log=log, density=density, range=range, cumulative=cumulative)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    if title:
        ax.set_title(title)
        
def plot_images_and_hists(
    images: Sequence[ArrayLike],
    titles: Optional[Sequence[str]] = None,
    cmap: str = "gray",
    bins: int = 256,
    color: Optional[str] = None,
    log: bool = False,
    norm: bool = False,
    density: bool = False,
    range: Optional[Tuple[float, float]] = None,
    cumulative: bool = False,
    show_colorbar: bool = False,
    hist_ratio: float = 0.2,
    ncols: int = 1,
    figsize: Tuple[float, float] = (10, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Display multiple images alongside their intensity histograms.

    Parameters
    ----------
    images : Sequence of ArrayLike
        List of 2D image arrays to display.
    titles : Sequence of str, optional
        Titles to show above each image (same length as `images`).
    cmap : str, optional
        Colormap to use for displaying images. Default is "gray".
    bins : int, optional
        Number of bins for histograms. Default is 256.
    color : str, optional
        Color for histogram bars.
    log : bool, optional
        If True, use logarithmic scale for histogram y-axis.
    norm : bool, optional
        If True, normalize image values to [0, 1] before displaying.
    density : bool, optional
        If True, normalize histograms to show probability density.
    range : tuple of float, optional
        Intensity range to use for histograms. If None, inferred from data.
    cumulative : bool, optional
        If True, plot cumulative histograms.
    show_colorbar : bool, optional
        If True, display a colorbar beside each image.
    hist_ratio : float, optional
        Relative width of the histogram subplot compared to the image subplot.
    ncols : int, optional
        Number of image+histogram columns in the layout. Default is 1.
    figsize : tuple of float, optional
        Overall figure size in inches. Default is (10, 4).
    save_path : str, optional
        If provided, path to save the resulting figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object containing all image and histogram subplots.
    """
    if titles is not None and len(titles) != len(images):
        raise ValueError("Length of titles must match length of images")
    
    n_images = len(images)
    ncols = max(1, int(ncols))
    nrows = (n_images + ncols - 1) // ncols

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows * 1, ncols * 2, figure=fig, width_ratios=[1 - hist_ratio, hist_ratio] * ncols)

    for idx, image in enumerate(images):
        row, col = divmod(idx, ncols)
        img_pos = col * 2 + row * ncols * 2
        hist_pos = img_pos + 1

        ax_img = fig.add_subplot(gs[img_pos])
        ax_hist = fig.add_subplot(gs[hist_pos])

        title = titles[idx] if titles and idx < len(titles) else None
        show_plane(ax_img, image, cmap=cmap, title=title, norm=norm, colorbar=show_colorbar)
        plot_hist(ax_hist, image, title="Histogram", bins=bins, color=color, log=log, density=density, range=range, cumulative=cumulative)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
    
def plot_histogram_bins(
    hist_bins: ArrayLike,
    bins_to_show: Sequence[int] = (0, 1, 2, 3),
    log_scale: bool = True,
    norm: bool = True,
    title_prefix: str = "Bin",
    cmap: str = "plasma",
    colorbar: bool = False,
    figsize: Tuple[float, float] = (15, 6),
    color: Optional[str] = None,
    density: bool = False,
    range: Optional[Tuple[float, float]] = None,
    cumulative: bool = False,
    hist_ratio: float = 0.2,
    ncols: int = 1,
) -> plt.Figure:
    """
    Plot selected bins from a local histogram map of shape (B, H, W) or (B, H, W, 1).

    Parameters
    ----------
    hist_bins : np.ndarray or torch.Tensor
        Histogram bins map, shape (n_bins, H, W) or (n_bins, H, W, 1).
    bins_to_show : tuple of int
        Indices of bins to display.
    log_scale : bool
        Apply log1p to enhance low values.
    norm : bool
        Normalize each image.
    title_prefix : str
        Prefix for each subplot title.
    cmap : str
        Colormap name.
    colorbar : bool
        Whether to show colorbar per image.
    figsize : tuple
        Size of the figure.
    color : str
        Color for histogram bars.
    density : bool
        Use density for histogram.
    range : tuple
        Range for histogram.
    cumulative : bool
        Use cumulative histogram.
    hist_ratio : float
        Ratio of histogram width.
    ncols : int
        Number of columns in the grid.
    
    """
    if isinstance(hist_bins, torch.Tensor):
        hist_bins = hist_bins.detach().cpu().numpy()
    arr = np.asarray(hist_bins)

    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected shape (B,H,W) or (B,H,W,1), got {arr.shape}")

    images: List[ArrayNP] = []
    titles: List[str] = []

    for b in bins_to_show:
        if 0 <= b < arr.shape[0]:
            img = arr[b]
            if log_scale:
                img = np.log1p(np.maximum(img, 0))
            images.append(img)
            titles.append(f"{title_prefix} {b}")

    return plot_images_and_hists(
        images,
        titles=titles,
        cmap=cmap,
        norm=norm,
        bins=arr.shape[0],
        color=color,
        density=density,
        range=range,
        cumulative=cumulative,
        hist_ratio=hist_ratio,
        show_colorbar=colorbar,
        ncols=ncols,
        figsize=figsize,
    )

    
def display_features(
    features: Mapping[str, ArrayLike],
    max_cols: int = 4,
    figsize_per_plot: Tuple[float, float] = (4, 4),
    norm: bool = False,
    colorbar: bool = True,
    postprocess: bool = True,
) -> None:
    """
    Display a grid of feature maps with automatic layout and proper colormaps.
    
    Parameters
    ----------
    features : dict
        Dictionary {feature_name: feature_map}.
    max_cols : int
        Maximum number of columns.
    figsize_per_plot : tuple of int
        Size (width, height) per subplot.
    norm : bool
        Apply normalization to [0, 1] per feature.
    colorbar : bool
        Whether to add colorbars.
    postprocess : bool
        Whether to apply log1p/clip/abs based on feature type.

    Returns
    -------
    None
    """
    n_features = len(features)
    n_cols = max(1, min(n_features, max_cols))
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for idx, (feature_name, fmap) in enumerate(features.items()):
        ax = axes[idx]
        show_plane(
            ax,
            fmap,
            feature_type=feature_name,
            title=feature_name,
            colorbar=colorbar,
            norm=norm,
            postprocess=postprocess,
        )

    # Hide extra axes
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    
def save_features_grid(
    features: Mapping[str, ArrayLike],
    save_path: str,
    max_cols: int = 4,
    figsize_per_plot: Tuple[float, float] = (4, 4),
    norm: bool = False,
    colorbar: bool = True,
    postprocess: bool = True,
    dpi: int = 150,
    show: bool = False,
) -> None:
    """
    Save a grid of feature maps to a file (PNG, PDF, etc.), with automatic layout and proper colormaps.
    
    Parameters
    ----------
    features : dict
        Dictionary {feature_name: feature_map}.
    save_path : str
        Output path for saving the figure (e.g., './features_grid.png').
    max_cols : int
        Maximum number of columns.
    figsize_per_plot : tuple of int
        Size (width, height) per subplot.
    norm : bool
        Normalize feature maps to [0, 1].
    colorbar : bool
        Whether to add colorbars.
    postprocess : bool
        Apply log1p/clip/abs based on feature type.
    dpi : int
        Resolution in dots per inch.
    show : bool
        If True, also displays the figure interactively.

    Returns
    -------
    None
    """
    n_features = len(features)
    n_cols = max(1, min(n_features, max_cols))
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows))
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for idx, (feature_name, fmap) in enumerate(features.items()):
        ax = axes[idx]
        show_plane(
            ax,
            fmap,
            feature_type=feature_name,
            title=feature_name,
            colorbar=colorbar,
            norm=norm,
            postprocess=postprocess,
        )

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


