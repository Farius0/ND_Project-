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
    Layout description and overrides.

    Parameters
    ----------
    layout : dict | None
        Full axes dict if already known (keys like 'channel_axis', ...).
    layout_name : str | None, default 'HWC'
        Mnemonic describing the layout (e.g., 'HWC', 'NCHW').
    layout_framework : {'numpy','torch'} | None, default 'numpy'
        Framework to use when resolving a layout name.
    *_axis : int | None
        Explicit overrides for individual axes.
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
    Global (backend-agnostic) operator configuration.

    Parameters
    ----------
    framework : {'numpy','torch'}, default 'numpy'
    output_format : {'numpy','torch'}, default 'numpy'
    device : str, default 'cpu'
    add_channel_dim, add_batch_dim : bool | None
        Optional dimension insertion policies.
    normalize : bool, default True
        Enable integer-to-float normalization in BaseConverter.
    verbose : bool, default False
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
    Configuration for lightweight pre-processing chains (stretch/clip/normalize).
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
    Configuration for synthetic/augmented datasets used in operators/tests.
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
    Configuration for (traceable) torchvision transforms.
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
    Configuration for basic/heuristic segmentation operators.
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
    Configuration for ND resizing (operators/resize_image.py).
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
    Configuration for generic processors (operators/image_processor.py).
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
    Configuration for ND convolution backends (operators/gaussian.py).
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
    Configuration for finite-difference style operators (operators/diff_operator.py).
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
    Generic filter parameters (filters/*).
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
    Algorithmic settings for high-level routines (algorithms/*).
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
    Feature extraction settings (operators/feature_extractor.py).
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
