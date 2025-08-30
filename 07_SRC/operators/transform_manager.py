# ==================================================
# ==========  MODULE: transform_manager  ===========
# ==================================================
from __future__ import annotations

from typing import Any, Callable, Union, Dict, Optional, Tuple, Literal
import torchvision.transforms as T
import numpy as np, torch
from PIL import Image

from core.config import LayoutConfig, GlobalConfig
from core.operator_core import OperatorCore

# Public API
__all__ = ["TransformManager"]

ArrayLike = Union[np.ndarray, torch.Tensor, Image.Image]
Framework = Literal["numpy", "torch"]

# ==================================================
# =========== TRANSFORM MANAGER CLASS ==============
# ==================================================

class TransformManager(OperatorCore):
    """
    Apply torchvision-style transformation pipelines to images via a robust PIL bridge,
    with axis tracking and dual-backend support.

    Notes
    -----
    - Inputs: NumPy, Torch, or PIL.Image.
    - Internally, we convert to NumPy (channel-last, uint8) for PIL compatibility.
    - Outputs are converted back to the requested backend and tagged.
    """

    # ====[ INITIALIZATION – TransformManager ]====
    def __init__(
        self,
        *,
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize TransformManager with full axis management and configuration.
        """
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg

        # Axis resolution
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

        super().__init__(layout_cfg=self.layout_cfg, global_cfg=self.global_cfg)

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

    # ====[ APPLY TRANSFORMS – Robust PIL Bridge & Axis Tracking ]====
    def apply_transforms(
        self,
        image: ArrayLike,
        transform: Callable,
        *,
        return_format: str = "torch",
        tag_as: str = "output",
        enable_uid: bool = False,
        op_params: dict | None = None,
        allow_pil_input: bool = True,
        return_tracker: bool = False,
    ) -> Union[ArrayLike, Tuple[ArrayLike, Any]]:
        """
        Apply torchvision-style transforms to an image using a safe PIL bridge.

        Parameters
        ----------
        image : np.ndarray | torch.Tensor | PIL.Image
            Input image in NumPy, Torch, or PIL format.
        transform : torchvision.transforms.Compose | callable
            Transform pipeline to apply.
        return_format : {'torch','numpy'}
            Desired output backend.
        tag_as : str
            Status tag for the final output.
        enable_uid : bool
            Whether to attach a UID.
        op_params : dict | None
            Metadata to store in the tag.
        allow_pil_input : bool
            Accept PIL input directly (no initial conversion).
        return_tracker : bool
            If True, return the AxisTracker instead of raw data.

        Returns
        -------
        ndarray | Tensor | AxisTracker
            Transformed data in the chosen backend (or the tracker).
        """
        if not callable(transform):
            raise TypeError("Transform must be callable (e.g., transforms.Compose).")

        # ---- Step 0: Make/obtain a tracker on the input ----
        if isinstance(image, Image.Image) and allow_pil_input:
            # We'll assume channel-last for PIL inputs.
            image = np.array(image)
            
        tracker_in = self.track(
            self.convert_once(
                image=image,
                framework="numpy",
                tag_as="input",
                enable_uid=enable_uid,
                op_params=op_params,
            )
        )

        arr_in = tracker_in.get()  # NumPy array (channel-last enforced below)

        # ---- Step 1: Prepare for PIL (channel-last, uint8) ----
        # Detect current channel axis from tag if available; otherwise from config.
        tag = self.get_tag(arr_in, "numpy") if self.has_tag(arr_in, "numpy") else {}
        ch_ax = tag.get("channel_axis", self.channel_axis)

        # Move channel to last if needed
        if ch_ax is not None and ch_ax != -1:
            tracker_in = tracker_in.moveaxis(ch_ax, -1)
            arr_in = tracker_in.get()

        # Ensure dtype uint8 for PIL
        if np.issubdtype(arr_in.dtype, np.floating):
            # Assume in [0,1] or arbitrary float → clip to [0,1], then scale
            tracker_in = tracker_in.apply_to_all(lambda x: x.clip(0.0, 1.0))
            tracker_in = tracker_in.apply_to_all(lambda x: (x * 255.0).astype(np.uint8, copy=False))
        elif arr_in.dtype != np.uint8:
            # Generic cast
            tracker_in = tracker_in.apply_to_all(lambda x: x.astype(np.uint8, copy=False))
            arr_in = tracker_in.get()

        # Squeeze stray singleton channel if grayscale 3D (H,W,1) → (H,W)
        if arr_in.ndim == 3 and arr_in.shape[-1] == 1:
            tracker_in = tracker_in.apply_to_all(lambda x: x[..., 0])
            tracker_in.update_tags({"channel_axis": None, "ndim": 2, "shape_after": arr_in.shape[0:-1]})  # Update tag to reflect removal
            arr_in = tracker_in.get()
            
        # ---- Step 2: Apply transforms with PIL ----
        pil_in = Image.fromarray(arr_in)
        transformed = transform(pil_in)

        # Normalize transform output to ndarray for convert_once
        if isinstance(transformed, Image.Image):
            out_np = np.array(transformed)
        elif isinstance(transformed, torch.Tensor):
            # TorchVision often returns CHW float tensors in [0,1]
            out_np = transformed.detach().cpu().numpy()
            if out_np.ndim >= 3 and out_np.shape[0] in (1, 3, 4):
                # Convert CHW -> HWC for consistency with PIL bridge
                out_np = np.moveaxis(out_np, 0, -1)
            # If float, scale back to uint8
            if np.issubdtype(out_np.dtype, np.floating):
                out_np = np.clip(out_np, 0.0, 1.0)
                out_np = (out_np * 255.0).astype(np.uint8, copy=False)
        elif isinstance(transformed, np.ndarray):
            out_np = transformed
        else:
            raise TypeError(
                "Transform output must be PIL.Image, np.ndarray or torch.Tensor."
            )

        # ---- Step 3: Convert to requested backend and tag ----
        out_data = self.convert_once(
            image=out_np,
            framework=return_format,
            tag_as=tag_as,
            enable_uid=enable_uid,
            op_params=op_params if op_params else {"transform_applied": str(transform)},
        )

        return out_data if not return_tracker else (out_data, self.track(out_data))
    
    # ====[ SUMMARY – TransformManager Configuration ]====
    def summary(self) -> None:
        """
        Print a complete configuration summary for debugging and clarity.
        """
        print("[TransformManager] Configuration Summary:")
        print(f"  Normalize         : {self.normalize}")
        print(f"  Add Batch Dim     : {self.add_batch_dim}")
        print(f"  Add Channel Dim   : {self.add_channel_dim}")
        print(f"  Device            : {self.device}")
        print(f"  Framework         : {self.framework}")
        print(f"  Output Format     : {self.output_format}")
        print(f"  Layout Name       : {self.layout_name}")

        print("  Axes Configuration:")
        for axis_name, axis_value in self.axes.items():
            print(f"    - {axis_name:<15}: {axis_value}")
