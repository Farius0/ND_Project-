# ==================================================
# =========== MODULE: edge_aware_filter ============
# ==================================================
from __future__ import annotations

from typing import Optional, Any, Dict, Union, Literal
import numpy as np, torch

from operators.image_processor import ImageProcessor
from core.operator_core import OperatorCore
from core.config import LayoutConfig, GlobalConfig, ImageProcessorConfig, FilterConfig

# Public API
__all__ = ["EdgeAwareFilter"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Framework = Literal["numpy", "torch"]
DtypeLike = Union[str, type, np.dtype, torch.dtype]

class EdgeAwareFilter(OperatorCore):
    """
    Edge-aware conductivity function used in anisotropic diffusion.

    Supports two modes:
      - 'pm'  : g(x) = 1 / sqrt(1 + (x/alpha)^2)      [Perona–Malik]
      - 'exp' : g(x) = exp(-(x/alpha)^2)              [Exponential decay]

    Dual backend (NumPy / Torch), ND-ready via ImageProcessor.
    """

    def __init__(
        self,
        filter_cfg: FilterConfig = FilterConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
        img_process_cfg: ImageProcessorConfig = ImageProcessorConfig(),
    ):
        """
        Initialize an edge-aware filter for use in anisotropic diffusion.

        Parameters
        ----------
        filter_cfg : FilterConfig
            Configuration for the filter mode ('pm' or 'exp') and alpha parameter.
        layout_cfg : LayoutConfig
            Layout configuration to ensure consistent axis interpretation.
        global_cfg : GlobalConfig
            Global backend and strategy configuration.
        img_process_cfg : ImageProcessorConfig
            Image processor configuration for ND handling and parallelism.
        """
        
        # ---- Config mirrors ----
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg
        self._filter_cfg: FilterConfig = filter_cfg
        self.img_process_cfg: ImageProcessorConfig = img_process_cfg

        # ---- Resolve layout/axes meta ----
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

        # ---- Filter params ----
        self.alpha: float = float(self._filter_cfg.alpha)
        self.mode: str = self._filter_cfg.mode
        self.as_float: bool = bool(self._filter_cfg.as_float)
        self.dtype: Optional[DtypeLike] = self._filter_cfg.dtype  # "auto" | numpy/torch dtype | python type | str

        # ---- Global / device ----
        self.framework: Framework = self.global_cfg.framework.lower()
        self.output_format: Framework = self.global_cfg.output_format.lower()
        self.add_batch_dim: Optional[bool] = self.global_cfg.add_batch_dim
        self.add_channel_dim: Optional[bool] = self.global_cfg.add_channel_dim
        self.normalize: bool = bool(self.global_cfg.normalize)
        self.device: str = "cuda" if (torch.cuda.is_available() and self.framework == "torch") else self.global_cfg.device

        # ---- Validate params ----
        if not (isinstance(self.alpha, (float, int)) and self.alpha > 0):
            raise ValueError("Parameter 'alpha' must be a positive number.")
        if self.mode not in {"pm", "exp"}:
            raise ValueError("Parameter 'mode' must be either 'pm' or 'exp'.")

        # ---- Init OperatorCore ----
        super().__init__(layout_cfg=self.layout_cfg, global_cfg=self.global_cfg)

        # ---- Per-channel processor (keeps framework/output consistent) ----
        self.processor: ImageProcessor = ImageProcessor(
            img_process_cfg=self.img_process_cfg,
            layout_cfg=self.layout_cfg,
            global_cfg=self.global_cfg.update_config(output_format=self.framework),
        )


    def __call__(self, u: ArrayLike) -> ArrayLike:
        """
        Apply edge-aware filtering to the input image or volume.

        The filter modulates smoothing intensity based on local gradients,
        preserving sharp edges while reducing noise in homogeneous regions.

        Parameters
        ----------
        u : ArrayLike
            Input image or ND volume (NumPy array or PyTorch tensor).

        Returns
        -------
        ArrayLike
            Filtered image or volume in the operator's configured output format.
        """

        u = self.convert_once(u)
        tagger = self.track(u)
        result = self._filter(u)
        tracker = tagger.clone_to(result, updates = 
                                  {"status": "filtered", 
                                   "shape_after": result.shape})
        
        return self.to_output(tracker.get(), tag_as="filtered")

    # ---------------- Internals ----------------

    def _filter(self, u: ArrayLike) -> ArrayLike:
        """
        Internal edge-aware filtering function applied per channel/slice.

        The method dispatches an edge-preserving non-linear filter (`pm` or `exp` mode)
        through the `ImageProcessor` to handle ND inputs with flexible layouts.

        Parameters
        ----------
        u : ArrayLike
            Input image or ND volume (NumPy array or PyTorch tensor).

        Returns
        -------
        ArrayLike
            Filtered image or volume, optionally cast to a configured dtype.
        """
        def edge_filter(channel: ArrayLike) -> ArrayLike:
            if self.mode not in {"pm", "exp"}:
                raise ValueError(f"Unsupported mode '{self.mode}' in edge_filter.")

            if isinstance(channel, torch.Tensor):    
                with torch.no_grad():
                    val = torch.clamp(channel / self.alpha, -10.0, 10.0)
                    return 1 / torch.sqrt(val ** 2 + 1) if self.mode == "pm" \
                        else torch.exp(-val ** 2)
            
            else:
                val = np.clip(channel / self.alpha, -10.0, 10.0)
                return 1 / np.sqrt(val ** 2 + 1) if self.mode == "pm" \
                    else np.exp(-val ** 2)

        self.processor.function = edge_filter

        result = self.processor(u)

        if self.dtype != "auto":
            return result.astype(self.dtype, copy=False) if isinstance(result, np.ndarray) \
                else result.to(self.dtype)
        elif self.as_float:
            return result.float() if isinstance(result, torch.Tensor) else result.astype(np.float32, copy=False)
        else:
            return result
