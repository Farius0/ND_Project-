# ==================================================
# =============  MODULE: thresholding  =============
# ==================================================

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Union, Any, Optional, Literal

import numpy as np, torch
from skimage.filters import (threshold_otsu, threshold_yen, threshold_isodata,
                             threshold_li, threshold_triangle)


from core.operator_core import OperatorCore
from core.config import (LayoutConfig, GlobalConfig, ImageProcessorConfig)
from operators.image_processor import ImageProcessor

# Public API
__all__ = ["ThresholdingOperator"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Framework = Literal["numpy", "torch"]

# ==================================================
# ============= ThresholdingOperator ===============
# ==================================================

class ThresholdingOperator(OperatorCore):
    """
    ND thresholding operator for binary mask generation or intensity separation.

    Supports classic thresholding methods (e.g., Otsu) and applies them in a 
    layout-aware and dual-backend manner via `OperatorCore`.

    Notes
    -----
    - Compatible with 2D, 3D, or batched ND images.
    - Can return binary masks (`as_mask=True`) or retain raw intensity splits.
    - Designed to integrate with full layout and UID tagging pipeline.
    """
    def __init__(
        self,
        method: str = "otsu",
        as_mask: bool = True,
        img_process_cfg: ImageProcessorConfig = ImageProcessorConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize the thresholding operator with method selection and layout-aware configs.

        Parameters
        ----------
        method : str, default 'otsu'
            Thresholding method to use (e.g., 'otsu', 'manual', or custom).
        as_mask : bool, default True
            If True, returns a binary mask; otherwise returns the intensity-split image.
        img_process_cfg : ImageProcessorConfig
            Processor configuration for applying the operation (strategy, return format, etc.).
        layout_cfg : LayoutConfig
            Axis layout configuration for spatial awareness and tagging.
        global_cfg : GlobalConfig
            Global processing options (framework, device, output format, etc.).
        """ 
        # ====[ Configuration ]====
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg
        self.img_process_cfg: ImageProcessorConfig = img_process_cfg
        
         # === Layout: resolved axes, layout name, layout dict ===
        self.axes: Dict[str, Any] = self.layout_cfg.resolve(include_meta=True)
        self.layout_name: str = self.axes.get("layout_name")
        self.layout: Dict[str, Any] = self.axes.get("layout")
        self.layout_framework: Framework = self.axes.get("layout_framework")

        self.channel_axis: Optional[int] = self.axes.get("channel_axis")
        self.batch_axis: Optional[int] = self.axes.get("batch_axis")
        self.direction_axis: Optional[int] = self.axes.get("direction_axis")
        self.height_axis: Optional[int] = self.axes.get("height_axis")
        self.width_axis: Optional[int] = self.axes.get("width_axis")
        self.depth_axis: Optional[int] = self.axes.get("depth_axis") 
        
        # ====[ Store ThresholdingOperator-specific parameters ]====
        self.method: str = method
        self.as_mask: bool = as_mask
        self.processor_strategy: str = self.img_process_cfg.processor_strategy
        
        # ====[ Mirror inherited params locally for easy access ]====
        self.framework: Framework = self.global_cfg.framework.lower()
        self.output_format: Framework = self.global_cfg.output_format.lower()
        self.add_batch_dim: Optional[bool] = self.global_cfg.add_batch_dim
        self.add_channel_dim: Optional[bool] = self.global_cfg.add_channel_dim
        self.normalize: bool = bool(self.global_cfg.normalize)
        self.verbose: bool = bool(self.global_cfg.verbose)        
        self.device: str = (
            "cuda"
            if (torch.cuda.is_available() and self.framework == "torch")
            else self.global_cfg.device
        )
        
        # ====[ Initialize OperatorCore with all axes ]====
        super().__init__(
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg,
        )  
        
        # === Create ImageProcessor (ensure matching output_format) ===
        # Fallback if GlobalConfig has no 'update_config' method.
        try:
            tuned_global = self.global_cfg.update_config(output_format=self.framework)  # type: ignore[attr-defined]
        except AttributeError:
            tuned_global = replace(self.global_cfg, output_format=self.framework)

        self.processor = ImageProcessor(
            img_process_cfg=self.img_process_cfg,
            layout_cfg=self.layout_cfg,
            global_cfg=tuned_global,
        )

    def __call__(
        self,
        image: ArrayLike,
        enable_uid: bool = False,
        op_params: dict | None = None,
        framework: Framework | None = None,
        output_format: Framework | None = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: bool | None = None,
    ) -> ArrayLike:
        """
        Apply thresholding to an ND image using the configured method.

        The input is first normalized (if required), tagged, and passed to the thresholding
        function. Output is then converted to the requested backend with full tagging.

        Parameters
        ----------
        image : ndarray | Tensor
            Input ND image or volume to threshold.
        enable_uid : bool, default False
            Whether to attach a UID to the output.
        op_params : dict or None, optional
            Optional metadata parameters to embed in the tag.
        framework : {'numpy', 'torch'} or None, optional
            Backend to use for internal processing. Defaults to current configuration.
        output_format : {'numpy', 'torch'} or None, optional
            Format to return the result in. Defaults to `self.output_format`.
        track : bool, default True
            Whether to propagate axis and layout tags from the input.
        trace_limit : int, default 10
            Number of levels to retain in UID trace history (if UID is enabled).
        normalize_override : bool or None
            Optional override for automatic normalization before thresholding.

        Returns
        -------
        ndarray | Tensor
            Thresholded output. Either a binary mask (if `as_mask=True`) or a thresholded image,
            formatted and tagged according to output preferences.
        """
        image = self.convert_once(
            image=image,
            tag_as="thresholding",
            framework=framework or self.framework,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override,
        )
        result = self._apply_threshold(image)

        return self.to_output(
            result,
            tag_as="thresholded",
            framework=output_format or self.output_format,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override,
        )

    # ------------------- Internals -------------------
    def _get_threshold_func(self) -> Callable:
        """
        Return the thresholding function corresponding to the selected method.

        Supported methods include:
        - 'otsu'
        - 'yen'
        - 'isodata'
        - 'li'
        - 'triangle'

        Returns
        -------
        Callable
            Function that computes the scalar threshold from an ND array.

        Raises
        ------
        ValueError
            If the selected method is not among the supported ones.
        """
        methods: Dict[str, Callable] = {
            "otsu": threshold_otsu,
            "yen": threshold_yen,
            "isodata": threshold_isodata,
            "li": threshold_li,
            "triangle": threshold_triangle,
        }
        key = self.method.lower()
        if key not in methods:
            raise ValueError(
                f"Unknown thresholding method '{self.method}'. "
                f"Supported: {list(methods.keys())}"
            )
        return methods[key]


    def _apply_threshold(self, image: ArrayLike) -> ArrayLike:
        """
        Apply the configured thresholding method to an ND image.

        Notes
        -----
        - Applies slice-wise thresholding for multi-dimensional arrays.
        - Automatically handles both NumPy and Torch backends.
        - Output is either a binary mask or scalar threshold, depending on `as_mask`.

        Parameters
        ----------
        image : ndarray | Tensor
            Input image or volume. Must be real-valued (float or integer).

        Returns
        -------
        ndarray | Tensor
            Binary mask (same shape as input) if `as_mask=True`,
            or scalar threshold(s) if `as_mask=False`, backend-preserved.
        """
        threshold_func = self._get_threshold_func()

        def apply_single_channel(slice_: ArrayLike):
            """
            Apply thresholding to a single-channel slice:
            
            """            
            if isinstance(slice_, torch.Tensor):
                arr = slice_.detach().cpu().numpy()
                threshold = threshold_func(arr)
                threshold = torch.tensor(threshold, dtype=slice_.dtype, device=slice_.device)
            else:
                threshold = threshold_func(slice_)

            return (slice_ > threshold) if self.as_mask else threshold

        self.processor.function = apply_single_channel
        processor = self.processor

        result = processor(image)        
        
        if not self.as_mask:
            # Consolidate thresholds into tensor or array
            if isinstance(image, torch.Tensor):
                return torch.tensor(result, dtype=image.dtype, device=image.device)
            return np.array(result, dtype=image.dtype)

        return self.to_output(result, tag_as="thresholded")
