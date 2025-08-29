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
    def __init__(
        self,
        method: str = "otsu",
        as_mask: bool = True,
        img_process_cfg: ImageProcessorConfig = ImageProcessorConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        
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
        Apply the selected thresholding method.

        Parameters
        ----------
        image : ndarray | Tensor
            Input image/volume (ND). Per-channel thresholding is handled upstream.
        as_mask : bool (from constructor)
            If True, returns a binary mask in the same backend.
            If False, returns the scalar threshold per slice/channel.

        Returns
        -------
        ndarray | Tensor
            Tagged output in the requested format.
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
        return self.to_output(result, tag_as="thresholded")
