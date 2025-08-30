# ==================================================
# ================  MODULE: config  ================
# ==================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from core.layout_axes import get_layout_axes

__all__ = [
    "LayoutConfig",
    "GlobalConfig",
    "PreprocessorConfig",
    "DatasetConfig",
    "TransformerConfig",
    "SegmenterConfig",
    "ResizeConfig",
    "ImageProcessorConfig",
    "NDConvolverConfig",
    "DiffOperatorConfig",
    "EdgeDetectorConfig",
    "FilterConfig",
    "AlgorithmConfig",
    "FeatureConfig",
    "TraceableRandomRotation",
    "TraceableRandomHorizontalFlip",
    "TraceableRandomVerticalFlip",
    "TraceableCompose",
]


# ==================================================
# ===============  CLASS: LayoutConfig  ============
# ==================================================
@dataclass
class LayoutConfig:
    """
    Configuration class to describe and override axis layout information.

    This config defines the semantic role of each axis (e.g., channel, batch, depth),
    either by name ("HWC", "NCHW", etc.) or via explicit axis indices. It also allows
    resolving layout for both NumPy and PyTorch backends.

    Attributes
    ----------
    layout : Optional[Dict[str, int]]
        Dictionary mapping semantic roles to axis indices (e.g., {"channel_axis": 2}).
        Used as an override or fallback if `layout_name` is not enough.
    layout_name : Optional[str], default "HWC"
        Shorthand for layout convention (e.g., "HWC", "NCHW").
    layout_framework : Optional[str], default "numpy"
        Target framework for resolving the layout ("numpy" or "torch").
    layout_ensured : Optional[str]
        Optional string used to validate or enforce a specific layout (internal use).
    layout_ensured_name : Optional[str]
        Name used to label the enforced layout.
    channel_axis : Optional[int]
        Index of the channel axis (e.g., 2 in "HWC").
    batch_axis : Optional[int]
        Index of the batch axis (e.g., 0 in "NCHW").
    direction_axis : Optional[int]
        Index of the direction axis (used for stacks or projections).
    height_axis : Optional[int]
        Index of the height axis.
    width_axis : Optional[int]
        Index of the width axis.
    depth_axis : Optional[int]
        Index of the depth axis (for 3D or volumetric data).
    """
    layout: Optional[Dict[str, int]] = None
    layout_name: Optional[str] = "HWC"
    layout_framework: Optional[str] = "numpy"
    layout_ensured: Optional[str] = None
    layout_ensured_name: Optional[str] = None
    channel_axis: Optional[int] = None
    batch_axis: Optional[int] = None
    direction_axis: Optional[int] = None
    height_axis: Optional[int] = None
    width_axis: Optional[int] = None
    depth_axis: Optional[int] = None

    def update_config(self, **kwargs) -> "LayoutConfig":
        """Dynamically update layout configuration (in-place)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"[LayoutConfig] Unknown config key: '{key}'")
        return self

    @staticmethod
    def resolve_axis(*args, default=None) -> Optional[Any]:
        """Return the first non-None value among `args`, else `default`."""
        return next((x for x in args if x is not None), default)

    def resolve(self, include_meta: bool = True) -> Dict[str, Optional[int]]:
        """
        Resolve axes using explicit overrides or fallback to a named layout.

        Returns
        -------
        dict
            Axes mapping and, if include_meta=True, {layout_name, layout, layout_framework}.
        """
        layout_list = [self.layout_name, self.layout_framework, self.layout]
        if all(x is None for x in layout_list):
            return {}

        if all(x is not None for x in layout_list[:-1]) and self.layout is None:
            self.layout = get_layout_axes(self.layout_framework, self.layout_name.upper())

        resolved = {
            "channel_axis": self.resolve_axis(self.channel_axis, self.layout.get("channel_axis")),
            "batch_axis": self.resolve_axis(self.batch_axis, self.layout.get("batch_axis")),
            "direction_axis": self.resolve_axis(self.direction_axis, self.layout.get("direction_axis")),
            "height_axis": self.resolve_axis(self.height_axis, self.layout.get("height_axis")),
            "width_axis": self.resolve_axis(self.width_axis, self.layout.get("width_axis")),
            "depth_axis": self.resolve_axis(self.depth_axis, self.layout.get("depth_axis")),
        }

        if include_meta:
            resolved["layout_name"] = self.layout_name
            resolved["layout"] = self.layout
            resolved["layout_framework"] = self.layout_framework

        return resolved

    def update_from_tag(self, tag: dict) -> None:
        """
        Update the layout configuration from a tag dictionary.

        Parameters
        ----------
        tag : dict
            Tag with potential keys: 'layout_name', axes keys, 'axis_map'.
        """
        if not isinstance(tag, dict):
            raise TypeError("Expected a dictionary for tag.")

        for axis in ["channel_axis", "batch_axis", "direction_axis", "height_axis", "width_axis", "depth_axis"]:
            value = tag.get(axis)
            if value is not None:
                setattr(self, axis, value)

        if "layout_name" in tag:
            self.layout_name = tag["layout_name"]

        if "axis_map" in tag and isinstance(tag["axis_map"], dict):
            self.layout = tag["axis_map"]

    def summary(self, printout: bool = True) -> Dict[str, Optional[int]]:
        """Return or print a summary of resolved axes."""
        resolved = self.resolve()
        resolved["layout_name"] = self.layout_name
        resolved["layout_framework"] = self.layout_framework

        if printout:
            print("=== [ LayoutConfig Summary ] ===")
            for k in resolved:
                print(f"{k:<18}: {resolved[k]}")
        return resolved


# ==================================================
# ===============  CLASS: GlobalConfig  ============
# ==================================================
@dataclass
class GlobalConfig:
    """
    Global configuration for operator behavior, independent of backend.

    This config controls framework settings, device targeting, output format,
    and preprocessing policies used by core pipeline components (e.g., BaseConverter).

    Attributes
    ----------
    framework : str, default "numpy"
        Preferred backend for computations. Must be "numpy" or "torch".
    output_format : str, default "numpy"
        Desired output format for operators. Must match the selected framework.
    device : str, default "cpu"
        Target device ("cpu", "cuda", etc.) if using torch backend.
    add_channel_dim : Optional[bool]
        Whether to add a channel dimension automatically if missing.
    add_batch_dim : Optional[bool]
        Whether to add a batch dimension automatically if missing.
    normalize : bool, default True
        Normalize input arrays from integers to floats (e.g., [0–255] to [0–1]).
    verbose : bool, default False
        Print internal logs or configuration summaries if True.
    
    backend : str, default "sequential"
        Execution strategy across modules ("sequential", "parallel", etc.).
    processor_strategy : str, default "vectorized"
        Strategy used in `ImageProcessor` ("vectorized", "parallel", "torch", etc.).
    diff_strategy : str, default "auto"
        Gradient computation strategy ("auto", "forward", "backward", etc.).
    conv_strategy : str, default "fft"
        Convolution backend used by `NDConvolver` ("fft", "spatial", etc.).
    edge_strategy : str, default "gradient"
        Edge detection logic ("gradient", "laplacian", etc.).

    spacing : Optional[float]
        Pixel spacing for gradient operators (used when spacing-aware).
    include_channel_gradient : bool, default False
        If True, include gradients along the channel axis during processing.
    """

    framework: str = "numpy"
    output_format: str = "numpy"
    device: str = "cpu"
    add_channel_dim: Optional[bool] = None
    add_batch_dim: Optional[bool] = None
    normalize: bool = True
    verbose: bool = False

    # ====[ Strategy Flags ]====
    backend: str = "sequential"
    processor_strategy: str = "vectorized"
    diff_strategy: str = "auto"
    conv_strategy: str = "fft"
    edge_strategy: str = "gradient"

    # ====[ Gradients Options ]====
    spacing: Optional[float] = None
    include_channel_gradient: bool = False

    def update_config(self, **kwargs) -> "GlobalConfig":
        """Dynamically update configuration attributes (in-place)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"[GlobalConfig] Unknown config key: '{key}'")
        return self

    def global_params(self) -> Dict[str, Optional[Union[str, bool]]]:
        """
        Compact dictionary of global flags useful for logs/serialization.
        """
        return {
            "framework": self.framework,
            "output_format": self.output_format,
            "device": self.device,
            "add_batch_dim": self.add_batch_dim,
            "add_channel_dim": self.add_channel_dim,
            "normalize": self.normalize,
        }

    def summary(self, printout: bool = True) -> Dict[str, Optional[Union[str, bool, float]]]:
        """Return or print a summary of the global configuration."""
        info = {
            "framework": self.framework,
            "output_format": self.output_format,
            "device": self.device,
            "add_batch_dim": self.add_batch_dim,
            "backend": self.backend,
            "processor_strategy": self.processor_strategy,
            "diff_strategy": self.diff_strategy,
            "conv_strategy": self.conv_strategy,
            "edge_strategy": self.edge_strategy,
            "spacing": self.spacing,
            "include_channel_grad": self.include_channel_gradient,
        }
        if printout:
            print("=== [ GlobalConfig Summary ] ===")
            for k in info:
                print(f"{k:<25}: {info[k]}")
        return info


# ==================================================
# ============  CLASS: PreprocessorConfig  =========
# ==================================================
@dataclass
class PreprocessorConfig:
    """
    Configuration for lightweight preprocessing operations.

    Defines stretch, clipping, normalization, gamma correction, artifact removal,
    and optional aggregation for input images. Used to prepare data before 
    conversion or processing by core operators.

    Attributes
    ----------
    fallback : Optional[Union[str, bool]], default True
        Fallback strategy in case of processing failure (True, False, or "warn").

    # --- Normalization ---
    normalize_mode : str, default "minmax"
        Type of normalization to apply ("minmax", "zscore", "robust").
    clip : bool, default False
        Whether to apply value clipping before normalization.
    clip_range : tuple, default (0.0, 1.0)
        Value range to enforce when clipping is enabled.

    # --- Stretching ---
    stretch : bool, default False
        Enable contrast stretching based on percentiles.
    p_low : float, default 1.0
        Lower percentile for stretching.
    p_high : float, default 99.0
        Upper percentile for stretching.
    out_range : tuple, default (0.0, 1.0)
        Output range for stretched values.

    # --- Aggregation ---
    aggregate : bool, default False
        If True, apply block-wise aggregation before processing.
    block_size : int, default 5
        Size of the block used for aggregation.
    agg_mode : str, default "mean"
        Aggregation mode ("mean", "median", "max", "min").
    keys : Optional[str], default "coronal"
        Key or direction to apply aggregation on.

    # --- Gamma correction ---
    gamma_correct : bool, default False
        Enable gamma correction.
    gamma : float, default 1.0
        Gamma value to apply.

    # --- Artifact removal ---
    remove_artifacts : bool, default False
        If True, attempt to remove artifacts using internal filters.

    # --- Denoising ---
    denoise : bool, default False
        Enable basic denoising.

    # --- Local contrast ---
    local_contrast : bool, default False
        Enhance local contrast adaptively.

    # --- Histogram equalization ---
    equalize : bool, default False
        Apply histogram equalization to enhance global contrast.
    """
    fallback: Optional[Union[str, bool]] = True

    # Normalization
    normalize_mode: str = "minmax"  # "zscore", "robust"
    clip: bool = False
    clip_range: tuple = (0.0, 1.0)

    # Stretching
    stretch: bool = False
    p_low: float = 1.0
    p_high: float = 99.0
    out_range: tuple = (0.0, 1.0)

    # Aggregation
    aggregate: bool = False
    block_size: int = 5
    agg_mode: str = "mean"  # "mean", "median", "max", "min"
    keys: Optional[str] = "coronal"

    # Gamma
    gamma_correct: bool = False
    gamma: float = 1.0

    # Artifact Removal
    remove_artifacts: bool = False

    # Denoising
    denoise: bool = False

    # Local Contrast
    local_contrast: bool = False

    # Equalization
    equalize: bool = False

    def update_config(self, **kwargs) -> "PreprocessorConfig":
        """Dynamically update configuration attributes (in-place)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"[PreprocessorConfig] Unknown config key: '{key}'")
        return self


# ==================================================
# ==============  CLASS: DatasetConfig  ============
# ==================================================
@dataclass
class DatasetConfig:
    """
    Configuration for synthetic or augmented datasets used in operators or test routines.

    This config controls the application of basic data augmentation, noise, masks,
    and transformation pipelines during dataset generation.

    Attributes
    ----------
    transform : Optional[Union[str, Callable]]
        Transformation pipeline or transformation name to apply on the data.
    operator : Optional[str]
        Name of the operator to apply during synthetic dataset creation.
    mask : Optional[str]
        Optional mask to use for filtering or segmentation tasks.
    noise_level : Optional[float]
        Standard deviation of Gaussian noise to add (if any).
    blur_level : Optional[float]
        Strength or kernel size of the blur to apply.
    mask_threshold : Optional[float]
        Threshold for binarizing masks or filtering input regions.
    random : Optional[bool]
        Whether to introduce randomness in the sample generation.
    mode : Optional[str]
        Mode of the dataset (e.g., "train", "test", or custom modes).
    clip : Optional[bool]
        Whether to clip values after applying transformations or operators.
    to_return : Optional[str], default "untransformed"
        What to return in `__getitem__`: 
        - "untransformed": return raw input,
        - "transformed": return only transformed data,
        - "both": return both input and transformed versions.
    return_param : Optional[bool], default False
        Whether to return parameters used during transformation (e.g., angles, flips).
    """

    transform: Optional[Union[str, Callable]] = None
    operator: Optional[str] = None
    mask: Optional[str] = None
    noise_level: Optional[float] = None
    blur_level: Optional[float] = None
    mask_threshold: Optional[float] = None
    random: Optional[bool] = None
    mode: Optional[str] = None
    clip: Optional[bool] = None
    to_return: Optional[str] = "untransformed"  # "transformed", "both"
    return_param: Optional[bool] = False

    def update_config(self, **kwargs) -> "DatasetConfig":
        """Dynamically update configuration attributes (in-place)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"[DatasetConfig] Unknown config key: '{key}'")
        return self    
    

# ==================================================
# ============  CLASS: TransformerConfig  ==========
# ==================================================
@dataclass
class TransformerConfig:
    """
    Configuration for traceable or standard torchvision-based transformations.

    Allows flexible definition of preprocessing transforms for images and labels,
    with optional parameter tracking for invertibility.

    Attributes
    ----------
    use_transform : bool, default False
        Whether to apply transformations to the input data.
    return_param : bool, default False
        If True, each transform returns parameters (e.g., angle, flip) for traceability.
    is_label : bool, default False
        If True, apply transforms in label-safe mode (e.g., nearest interpolation).
    size : Optional[Tuple[int, int]], default (256, 256)
        Target size for resizing the input (H, W).
    horizontal_flip : Optional[float]
        Probability of applying horizontal flip (between 0 and 1).
    vertical_flip : Optional[float]
        Probability of applying vertical flip (between 0 and 1).
    rotation : Optional[int]
        Maximum rotation angle in degrees (symmetric range).
    brightness : Optional[float]
        Brightness jitter range (e.g., 0.2 allows ±20% brightness).
    contrast : Optional[float]
        Contrast jitter range.
    saturation : Optional[float]
        Saturation jitter range.
    to_tensor : bool, default False
        Convert image to PyTorch tensor.
    normalize : bool, default False
        If True, normalize the tensor using `mean` and `std`.
    mean : Tuple[float, float, float], default (0.5, 0.5, 0.5)
        Mean values for normalization (per channel).
    std : Tuple[float, float, float], default (0.5, 0.5, 0.5)
        Standard deviation values for normalization (per channel).
    """
    use_transform: bool = False
    return_param: bool = False
    is_label: bool = False
    size: Optional[Tuple[int, int]] = (256, 256)
    horizontal_flip: Optional[float] = None
    vertical_flip: Optional[float] = None
    rotation: Optional[int] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    saturation: Optional[float] = None
    to_tensor: bool = False
    normalize: bool = False
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)

    def build_transform(self) -> Callable:
        """
        Build a transformation pipeline.

        Returns
        -------
        Callable
            - If return_param=False: standard torchvision.Compose
            - If return_param=True : TraceableCompose (returns (img, params_list))
        """
        if not self.use_transform:
            return transforms.ToTensor() if self.to_tensor else None

        transforms_list: List[Callable] = []

        def add_resize():
            if self.size is not None:
                interpolation = InterpolationMode.NEAREST if self.is_label else InterpolationMode.BILINEAR
                transforms_list.append(transforms.Resize(self.size, interpolation=interpolation))

        def add_flips():
            if self.horizontal_flip and self.horizontal_flip > 0:
                if self.return_param:
                    transforms_list.append(TraceableRandomHorizontalFlip(p=self.horizontal_flip, return_param=True))
                else:
                    transforms_list.append(transforms.RandomHorizontalFlip(self.horizontal_flip))

            if self.vertical_flip and self.vertical_flip > 0:
                if self.return_param:
                    transforms_list.append(TraceableRandomVerticalFlip(p=self.vertical_flip, return_param=True))
                else:
                    transforms_list.append(transforms.RandomVerticalFlip(self.vertical_flip))

        def add_rotation():
            if self.rotation and self.rotation > 0:
                if self.return_param:
                    transforms_list.append(TraceableRandomRotation(degrees=self.rotation, return_param=True))
                else:
                    transforms_list.append(transforms.RandomRotation(self.rotation))

        def add_color_jitter():
            if any([self.brightness, self.contrast, self.saturation]):
                transforms_list.append(
                    transforms.ColorJitter(
                        brightness=self.brightness,
                        contrast=self.contrast,
                        saturation=self.saturation,
                    )
                )

        def add_tensor_and_normalize():
            if self.to_tensor:
                transforms_list.append(transforms.ToTensor())
            if self.normalize:
                transforms_list.append(transforms.Normalize(mean=self.mean, std=self.std))

        add_resize()
        add_flips()
        add_rotation()
        add_color_jitter()
        add_tensor_and_normalize()

        if self.return_param:
            return TraceableCompose(transforms_list)
        return transforms.Compose(transforms_list) if transforms_list else None

    def build_inverse_transform(self, transform_pipeline, params_list):
        """
        Build an inverse function by invoking `.inverse` of each traceable transform.

        Returns
        -------
        Callable
            Function such that `inverse(img)` approximately undoes the pipeline.
        """
        def inverse_fn(img):
            for t, p in reversed(list(zip(transform_pipeline.transforms, params_list))):
                if p is not None and hasattr(t, "inverse"):
                    img = t.inverse(img, p)
            return img

        return inverse_fn

    def update_config(self, **kwargs) -> "TransformerConfig":
        """Dynamically update configuration attributes (in-place)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"[TransformerConfig] Unknown config key: '{key}'")
        return self


# ==================================================
# =============  CLASS: SegmenterConfig  ===========
# ==================================================
@dataclass
class SegmenterConfig:
    """
    Configuration for rule-based or heuristic segmentation operators.

    Supports single and multi-threshold methods, basic clustering (e.g., k-means),
    and optional feature-based segmentation.

    Attributes
    ----------
    mode : str, default "otsu"
        Segmentation mode to use ("otsu", "threshold", "kmeans", etc.).
    bins : int, default 256
        Number of bins used for histogram-based segmentation.
    threshold : float, default 0.5
        Threshold value used in binary segmentation (if mode="threshold").
    multi_thresholds : tuple, default (0.33, 0.66)
        Threshold values for multi-class segmentation (e.g., ternary).
    seeds : Optional[Union[List[int], int]]
        Seed(s) for reproducibility in random-based methods.
    n_seeds : int, default 3
        Number of seeds to use if `seeds` is not specified.
    window_size : int, default 11
        Size of local window used for adaptive methods.
    num_classes : int, default 2
        Number of output classes (used in clustering or multi-threshold modes).
    return_mask : bool, default True
        Whether to return a segmentation mask (otherwise return raw scores).
    clip_range : tuple, default (0.0, 1.0)
        Value range to clip input data before segmentation.
    epsilon : float, default 1e-3
        Tolerance for iterative convergence (if applicable).
    max_iter : int, default 500
        Maximum number of iterations for iterative methods (e.g., clustering).
    fallback : bool, default True
        Whether to fall back to default segmentation when method fails.
    debug : bool, default False
        If True, print debug information during segmentation.

    # --- KMeans clustering parameters ---
    kmeans_k : int, default 3
        Number of clusters to use if mode="kmeans".
    kmeans_auto : bool, default False
        If True, automatically select best `k` using a criterion.
    kmeans_method : str, default "silhouette"
        Method used to choose `k` if `kmeans_auto=True` ("silhouette", "gap", etc.).
    kmeans_max_k : int, default 10
        Upper limit for number of clusters when searching for best `k`.
    kmeans_batch_size : Optional[int]
        Optional batch size for mini-batch k-means (if supported).
    kmeans_max_iter : Optional[int]
        Maximum iterations for the k-means algorithm.

    # --- Feature-based segmentation ---
    normalize_output : bool, default False
        Whether to normalize output masks or scores.
    use_channels : bool, default True
        If True, apply segmentation across all input channels.
    use_features : bool, default False
        If True, apply segmentation using extracted features.
    feature_axis : Optional[int], default -1
        Axis index corresponding to the feature dimension.
    """
    mode: str = "otsu"
    bins: int = 256
    threshold: float = 0.5
    multi_thresholds: tuple = (0.33, 0.66)
    seeds: Optional[Union[List[int], int]] = None
    n_seeds: int = 3
    window_size: int = 11
    num_classes: int = 2
    return_mask: bool = True
    clip_range: tuple = (0.0, 1.0)
    epsilon: float = 1e-3
    max_iter: int = 500
    fallback: bool = True
    debug: bool = False
    kmeans_k: int = 3
    kmeans_auto: bool = False
    kmeans_method: str = "silhouette"
    kmeans_max_k: int = 10
    kmeans_batch_size: Optional[int] = None
    kmeans_max_iter: Optional[int] = None
    normalize_output: bool = False
    use_channels: bool = True
    use_features: bool = False
    feature_axis: Optional[int] = -1

    def update_config(self, **kwargs) -> "SegmenterConfig":
        """Dynamically update configuration attributes (in-place)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"[SegmenterConfig] Unknown config key: '{key}'")
        return self


# ==================================================
# ===============  CLASS: ResizeConfig  ============
# ==================================================
@dataclass
class ResizeConfig:
    """
    Configuration for N-dimensional image resizing.

    Supports interpolation mode, size control, anti-aliasing, and layout handling.
    Used by operators that perform spatial resizing (e.g., resize_image.py).

    Attributes
    ----------
    size : Optional[Tuple[int, int]], default (256, 256)
        Target spatial size (height, width) for 2D or ND images.
    resize_strategy : str, default "auto"
        Strategy to apply when resizing ("auto", "force", "preserve", etc.).
    mode : str, default "bilinear"
        Interpolation mode used for resizing ("nearest", "bilinear", "bicubic", etc.).
    align_corners : bool, default False
        Whether to align corners during interpolation (relevant in PyTorch).
    preserve_range : bool, default True
        If True, preserve the input intensity range during resizing.
    anti_aliasing : bool, default True
        Whether to apply anti-aliasing filter during downsampling.
    layout_ensured : Optional[dict]
        Optional dictionary of layout information to enforce after resizing
        (e.g., axis order or naming).
    """
    size: Optional[Tuple[int, int]] = (256, 256)
    resize_strategy: str = "auto"
    mode: str = "bilinear"
    align_corners: bool = False
    preserve_range: bool = True
    anti_aliasing: bool = True
    layout_ensured: Optional[dict] = None

    def update_config(self, **kwargs) -> "ResizeConfig":
        """Dynamically update configuration attributes (in-place)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"[ResizeConfig] Unknown config key: '{key}'")
        return self


# ==================================================
# =========  CLASS: ImageProcessorConfig  ==========
# ==================================================
@dataclass
class ImageProcessorConfig:
    """
    Configuration for general-purpose image processing operations.

    Allows customization of the callable function, processing strategy,
    input conversion, output formatting, and parallel execution.

    Attributes
    ----------
    function : Optional[Callable]
        Custom processing function to apply to each image.
    processor_strategy : str, default "auto"
        Processing strategy to use ("auto", "vectorized", "parallel", "torch", etc.).
    convert_inputs : Optional[bool], default True
        Whether to convert inputs to the appropriate format (e.g., NumPy or Torch).
    return_tuple : Optional[bool]
        If True, return results as a tuple (e.g., (image, metadata)).
    return_type : Optional[bool]
        If True, include type annotations or metadata in the result.
    fallback : Optional[Union[str, bool]], default False
        Fallback behavior in case of processing failure (False, True, or "warn").
    n_jobs : int, default -1
        Number of parallel jobs to use (-1 means use all available cores).
    """
    function: Optional[Callable] = None
    processor_strategy: str = "auto"
    convert_inputs: Optional[bool] = True
    return_tuple: Optional[bool] = None
    return_type: Optional[bool] = None
    fallback: Optional[Union[str, bool]] = False
    n_jobs: int = -1

    def update_config(self, **kwargs) -> "ImageProcessorConfig":
        """Dynamically update configuration attributes (in-place)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"[ImageProcessorConfig] Unknown config key: '{key}'")
        return self


# ==================================================
# ============  CLASS: NDConvolverConfig  ==========
# ==================================================
@dataclass
class NDConvolverConfig:
    """
    Configuration for N-dimensional convolution backends.

    Used to control convolution parameters (kernel size, padding, backend, etc.)
    for operations such as Gaussian smoothing or filtering.

    Attributes
    ----------
    conv_strategy : str, default "torch"
        Backend strategy to use for convolution ("torch", "scipy", "fft", etc.).
    padding : Optional[str], default "same"
        Padding mode ("same", "valid", "full", etc.).
    grouped : Optional[bool], default True
        Whether to apply grouped convolutions across channels.
    dim : Optional[int], default 2
        Number of spatial dimensions (1D, 2D, 3D, etc.).
    size : Optional[int]
        Kernel size (e.g., radius for Gaussian filter).
    sigma : Optional[float]
        Standard deviation for Gaussian kernels.
    angle : Optional[float], default 0.0
        Orientation angle in degrees for directional filters (if applicable).
    mode : Optional[str], default "reflect"
        Border mode used during convolution ("reflect", "constant", etc.).
    dtype : Optional[type], default torch.float32
        Data type to use for convolution kernels.
    truncate : Optional[float], default 3.0
        Truncation factor for Gaussian kernel size computation.
    symmetry : Optional[bool], default True
        Whether to enforce symmetric kernels.
    visualize : Optional[bool], default False
        If True, enable visualization of the convolution kernel/grid.
    return_grid : Optional[bool], default False
        If True, return the grid or kernel used during convolution.
    return_numpy : Optional[bool], default False
        If True, return the result as a NumPy array regardless of backend.
    conv_fallback : Optional[str], default "gaussian"
        Fallback strategy in case of failure (e.g., "gaussian", "identity").
    """
    conv_strategy: str = "torch"
    padding: Optional[str] = "same"
    grouped: Optional[bool] = True
    dim: Optional[int] = 2
    size: Optional[int] = None
    sigma: Optional[float] = None
    angle: Optional[float] = 0.0
    mode: Optional[str] = "reflect"
    dtype: Optional[type] = torch.float32
    truncate: Optional[float] = 3.0
    symmetry: Optional[bool] = True
    visualize: Optional[bool] = False
    return_grid: Optional[bool] = False
    return_numpy: Optional[bool] = False
    conv_fallback: Optional[str] = "gaussian"

    def update_config(self, **kwargs) -> "NDConvolverConfig":
        """Dynamically update configuration attributes (in-place)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"[NDConvolverConfig] Unknown config key: '{key}'")
        return self


# ==================================================
# ===========  CLASS: DiffOperatorConfig  ==========
# ==================================================
@dataclass
class DiffOperatorConfig:
    """
    Configuration for finite-difference differential operators.

    Controls how gradients, divergence, and other derivative-based operations
    are computed over N-dimensional arrays or tensors.

    Attributes
    ----------
    spacing : Optional[Union[List[float], float]]
        Physical spacing between pixels/voxels, used to scale derivatives.
        Can be a scalar or a list (one value per axis).
    diff_strategy : str, default "auto"
        Strategy to compute derivatives ("forward", "backward", "centered", or "auto").
    include_channel_gradient : Optional[bool], default False
        If True, compute gradients along the channel axis.
    max_flux : Optional[float]
        Optional clipping value to limit the magnitude of computed flux (used in stability control).
    diff_fallback : Optional[Union[str, bool]], default True
        Fallback strategy in case of failure ("identity", True, or False).

    diff_mode : Optional[Dict[str, str]]
        Per-operation differentiation mode (e.g., {"gradient": "forward", "divergence": "backward"}).
        If not specified for an operation, falls back to "default".

    boundary_mode : Optional[Dict[str, str]]
        Boundary conditions for each operator:
        - "dirichlet": zero at boundaries,
        - "neumann": constant gradient (mirror),
        - "periodic": wrap around.
        Example: {"gradient": "dirichlet", "divergence": "neumann"}.
    """
    spacing: Optional[Union[List[float], float]] = None
    diff_strategy: str = "auto"
    include_channel_gradient: Optional[bool] = False
    max_flux: Optional[float] = None
    diff_fallback: Optional[Union[str, bool]] = True
    diff_mode: Optional[Dict[str, str]] = field(
        default_factory=lambda: {"gradient": "forward", "divergence": "backward", "default": "centered"}
    )
    boundary_mode: Optional[Dict[str, str]] = field(
        default_factory=lambda: {"gradient": "dirichlet", "divergence": "neumann", "default": "dirichlet"}
    )

    def update_config(self, **kwargs) -> "DiffOperatorConfig":
        """Dynamically update configuration attributes (in-place)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"[DiffOperatorConfig] Unknown config key: '{key}'")
        return self


# ==================================================
# ===========  CLASS: EdgeDetectorConfig  ==========
# ==================================================
@dataclass
class EdgeDetectorConfig:
    """
    Configuration for ND edge detectors (operators/edge_detector.py).
    """
    edge_strategy: str = "auto"
    eta: Optional[Union[str, float]] = None
    mode: Optional[str] = "and"
    alpha: Optional[float] = 0.5
    threshold: Optional[Union[str, float]] = "auto"
    as_float: Optional[bool] = True
    complete_nms: Optional[bool] = True
    dtype: Optional[Union[type, str]] = "auto"
    use_padding: Optional[bool] = False

    def update_config(self, **kwargs) -> "EdgeDetectorConfig":
        """Dynamically update configuration attributes (in-place)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"[EdgeDetectorConfig] Unknown config key: '{key}'")
        return self


# ==================================================
# ===============  CLASS: FilterConfig  ============
# ==================================================
@dataclass
class FilterConfig:
    """
    Configuration for generic filters used in image processing pipelines.

    Defines the main parameters required to control filter intensity, behavior,
    and output format across filtering modules.

    Attributes
    ----------
    alpha : float, default 1.0
        Strength or sensitivity of the filter (e.g., diffusion coefficient in Perona–Malik).
    mode : str, default "pm"
        Filter mode or type (e.g., "pm" for Perona–Malik, "tv" for Total Variation, etc.).
    as_float : bool, default False
        If True, convert inputs to float before filtering for higher precision.
    dtype : Optional[Union[type, str]], default "auto"
        Desired output data type (e.g., float32, float64), or "auto" to infer from input.
    """
    alpha: float = 1.0
    mode: str = "pm"
    as_float: bool = False
    dtype: Optional[Union[type, str]] = "auto"

    def update_config(self, **kwargs) -> "FilterConfig":
        """Dynamically update configuration attributes (in-place)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"[FilterConfig] Unknown config key: '{key}'")
        return self


# ==================================================
# ============  CLASS: AlgorithmConfig  ============
# ==================================================
@dataclass
class AlgorithmConfig:
    """
    Algorithm-specific configuration for high-level processing routines.

    Primarily used to control iterative algorithms such as Perona–Malik diffusion,
    including step size, number of iterations, and output behavior.

    Attributes
    ----------
    dt : float, default 0.1
        Time step or integration constant for iterative schemes.
    steps : int, default 20
        Number of iterations to perform.
    clip : bool, default True
        Whether to clip output values to a safe range after each step.
    algorithm_strategy : str, default "pm"
        Strategy identifier for the algorithm to run (e.g., "pm" for Perona–Malik).
    return_evolution : bool, default False
        If True, return the full evolution over time instead of just the final output.
    disable_tqdm : bool, default False
        If True, disable progress bars during iteration.
    """
    # Perona–Malik
    dt: float = 0.1
    steps: int = 20
    clip: bool = True
    algorithm_strategy: str = "pm"
    return_evolution: bool = False
    disable_tqdm: bool = False

    def update_config(self, **kwargs) -> "AlgorithmConfig":
        """Dynamically update configuration attributes (in-place)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"[AlgorithmConfig] Unknown config key: '{key}'")
        return self


# ==================================================
# ==============  CLASS: FeatureConfig  ============
# ==================================================
@dataclass
class FeatureConfig:
    """
    Configuration for feature extraction across multiple domains.

    Includes settings for texture-based features (e.g., GLCM, Gabor), 
    frequency filters (bandpass), wavelets, ridge detection, and morphological profiles.

    Attributes
    ----------
    features : Optional[Union[List[str], str]], default "all"
        List of feature types to extract (or "all" to extract everything).
    stack : Optional[bool], default True
        If True, stack all features along a new axis.
    combined : Optional[bool], default False
        If True, merge all features into a single composite map.
    return_feat_names : Optional[bool], default False
        If True, return feature names alongside the features.
    block_mode : Optional[bool], default False
        If True, compute features in non-overlapping blocks.

    # --- Texture features ---
    window_size : Optional[int], default 5
        Size of the local window for texture-based operators.
    n_bins : int, default 8
        Number of bins used for histogram-based features.
    glcm_levels : int, default 8
        Number of gray levels for GLCM computation.
    glcm_mode : str, default "mean"
        Reduction mode for GLCM features ("mean", "range", etc.).
    gabor_freqs : List[float], default [0.1, 0.2]
        Frequencies used for Gabor filtering.

    # --- Frequency / bandpass features ---
    low : float, default 0.1
        Lower cutoff frequency for bandpass filter.
    high : float, default 0.5
        Upper cutoff frequency for bandpass filter.
    sharpness : float, default 10
        Sharpness of frequency transition in filters.
    soft : bool, default True
        If True, use smooth transitions; otherwise, hard cutoffs.

    # --- Wavelet features ---
    wavelet : str, default "haar"
        Wavelet basis to use for decomposition.
    level : int, default 1
        Decomposition level for wavelet transform.
    aggregate : Optional[str]
        Optional method to aggregate wavelet coefficients ("mean", "max", etc.).

    # --- Ridge / vesselness features ---
    beta : float, default 0.5
        Sensitivity to blob vs ridge structures in vesselness.
    c : float, default 15.0
        Scale normalization parameter for Frangi filter.
    ridge_mode : str, default "frangi"
        Type of ridge detector ("frangi", "sato", etc.).
    n_components : int, default 3
        Number of principal directions/components to extract.

    # --- Morphological features ---
    footprint : Optional[np.ndarray]
        Structuring element to use in morphological operations.
    operation : str, default "tophat"
        Type of morphological feature to extract ("tophat" or "blackhat").

    Methods
    -------
    update_config(self, **kwargs) -> "FeatureConfig"
        Dynamically update configuration attributes (in-place).
    """
    features: Optional[Union[List[str], str]] = field(default_factory=lambda: "all")
    stack: Optional[bool] = True
    combined: Optional[bool] = False
    return_feat_names: Optional[bool] = False
    block_mode: Optional[bool] = False

    # texture/window-based
    window_size: Optional[int] = 5
    n_bins: int = 8
    glcm_levels: int = 8
    glcm_mode: str = "mean"
    gabor_freqs: List[float] = field(default_factory=lambda: [0.1, 0.2])

    # continuous / frequency-based / bandpass filter
    low: float = 0.1
    high: float = 0.5
    sharpness: float = 10
    soft: bool = True

    # wavelets
    wavelet: str = "haar"
    level: int = 1
    aggregate: Optional[str] = None

    # ridge / vesselness
    beta: float = 0.5
    c: float = 15.0
    ridge_mode: str = "frangi"
    n_components: int = 3

    # morpho_hat
    footprint: Optional[np.ndarray] = None
    operation: str = "tophat"  # "tophat" or "blackhat"

    def update_config(self, **kwargs) -> "FeatureConfig":
        """Dynamically update configuration attributes (in-place)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"[FeatureConfig] Unknown config key: '{key}'")
        return self
    
# =====================================================================
# =========================  TRACEABLE TRANSFORMS  ====================
# =====================================================================

class TraceableRandomRotation:
    """
    Rotation with optional parameter return and invertibility.

    Parameters
    ----------
    degrees : float
        Max absolute rotation in degrees; if 90, uses {0,90,180,270}.
    return_param : bool, default False
        When True, returns (img, {'angle': angle}).
    interpolation : InterpolationMode, default BILINEAR
    fill : int | tuple, default 0
    expand : bool, default False
    exact : bool, default False
        When True, force NEAREST and expand=False (useful for labels).
    """

    def __init__(
        self,
        degrees: float,
        return_param: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Union[int, tuple] = 0,
        expand: bool = False,
        exact: bool = False,
    ):
        self.degrees = degrees
        self.return_param = return_param
        self.interpolation = interpolation
        self.fill = fill
        self.expand = expand
        self.exact = exact

    def __call__(self, img: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        angle = (
            random.choice((0, 90, 180, 270)) if self.degrees == 90 else (torch.rand(1).item() * 2 - 1) * self.degrees
        )
        interpolation = InterpolationMode.NEAREST if self.exact else self.interpolation
        expand = False if self.exact else self.expand
        out = TF.rotate(img, angle, interpolation=interpolation, fill=self.fill, expand=expand)
        if self.return_param:
            return out, {"angle": angle}
        return out

    @staticmethod
    def inverse(
        img: torch.Tensor,
        params: dict,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: Union[int, tuple] = 0,
        expand: bool = False,
        exact: bool = False,
    ) -> torch.Tensor:
        interpolation = InterpolationMode.NEAREST if exact else interpolation
        expand = False if exact else expand
        return TF.rotate(img, -params["angle"], interpolation=interpolation, fill=fill, expand=expand)


class TraceableRandomHorizontalFlip:
    """Horizontal flip with optional parameter return and invertibility."""
    def __init__(self, p: float = 0.5, return_param: bool = False):
        self.p = p
        self.return_param = return_param

    def __call__(self, img: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        flipped = torch.rand(1).item() < self.p
        out = TF.hflip(img) if flipped else img
        if self.return_param:
            return out, {"hflip": flipped}
        return out

    @staticmethod
    def inverse(img, params: dict) -> torch.Tensor:
        return TF.hflip(img) if params["hflip"] else img


class TraceableRandomVerticalFlip:
    """Vertical flip with optional parameter return and invertibility."""
    def __init__(self, p: float = 0.5, return_param: bool = False):
        self.p = p
        self.return_param = return_param

    def __call__(self, img : torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        flipped = torch.rand(1).item() < self.p
        out = TF.vflip(img) if flipped else img
        if self.return_param:
            return out, {"vflip": flipped}
        return out

    @staticmethod
    def inverse(img, params: dict) -> torch.Tensor:
        return TF.vflip(img) if params["vflip"] else img


class TraceableCompose:
    """
    A sequence of traceable transforms that can be inverted.

    Each transform should implement:
      - __call__(img) → (img, params) or img
      - inverse(img, params) → img
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, List[dict]]:
        """
        Apply all transforms in order.

        Returns
        -------
        (img, params_list)
            If a transform exposes 'return_param', its params dict is collected.
        """
        params_list: List[dict] = []
        for t in self.transforms:
            if getattr(t, "return_param", False):
                img, params = t(img)
                params_list.append(params)
            else:
                img = t(img)
                params_list.append({})
        return img, params_list

    def inverse(self, img, params_list: List[dict]) -> torch.Tensor:
        """Apply inverse transforms in reverse order using provided params."""
        for t, p in reversed(list(zip(self.transforms, params_list))):
            if not p:
                continue
            if not hasattr(t, "inverse"):
                raise ValueError(f"Transform {t.__class__.__name__} has no 'inverse'")
            try:
                img = t.inverse(img, p)
            except KeyError as e:
                raise KeyError(f"Missing expected param key {e} for {t.__class__.__name__}")
        return img
