# ==================================================
# ============= MODULE: resize_image ===============
# ==================================================
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, List, Dict, Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from skimage.transform import resize as sk_resize

from core.config import LayoutConfig, GlobalConfig, ResizeConfig, ImageProcessorConfig
from core.operator_core import OperatorCore
from operators.image_processor import ImageProcessor

# Public API
__all__ = ["ResizeOperator"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Framework = Literal["numpy", "torch"]

# ==================================================
# ================ ResizeOperator ==================
# ==================================================
class ResizeOperator(OperatorCore):
    """
    ND resizing operator with layout-aware and backend-specific logic.

    Supports flexible resizing of 2D, 3D, or batched ND images using NumPy or Torch,
    while preserving layout metadata and UID tagging via OperatorCore.

    Notes
    -----
    - Torch backend:
        * Input is normalized to (N, C, [D,] H, W), resized with `F.interpolate`, then restored.
    - NumPy backend:
        * Uses `skimage.transform.resize` for ND resizing (layout-aware),
          or OpenCV for fast 2D resizing when applicable.
    - If `match_to` is provided, the target spatial size is inferred from its layout-aware shape.
    - Handles both singleton and batched images (e.g., with or without batch/channel dimensions).
    """

    # ====[ INIT ]====
    def __init__(
        self,
        resize_cfg: ResizeConfig = ResizeConfig(),
        img_process_cfg: ImageProcessorConfig = ImageProcessorConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize the ResizeOperator with full axis tracking and backend configuration.

        Parameters
        ----------
        resize_cfg : ResizeConfig
            Resizing strategy, interpolation mode, anti-aliasing, and layout options.
        img_process_cfg : ImageProcessorConfig
            Processing options (strategy, return type, parallelism).
        layout_cfg : LayoutConfig
            Axis layout description and overrides.
        global_cfg : GlobalConfig
            Global execution behavior: backend, format, device, etc.
        """

        """Initialize with full axis control and processor configuration."""
        # --- Configs ---
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg
        self.resize_cfg: ResizeConfig = resize_cfg
        self.img_process_cfg: ImageProcessorConfig = img_process_cfg

        # --- Resolved axes / meta ---
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

        # --- Resize options ---
        self.size: Union[int, Sequence[int]]  = self.resize_cfg.size  # (H, W) or (D, H, W)
        self.strategy: str = self.resize_cfg.resize_strategy
        self.mode: str = self.resize_cfg.mode
        self.align_corners: bool = self.resize_cfg.align_corners
        self.preserve_range: bool = self.resize_cfg.preserve_range
        self.anti_aliasing: bool = self.resize_cfg.anti_aliasing
        self.layout_ensured: Dict[str, Any] = self.resize_cfg.layout_ensured

        # --- Globals ---
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

        # --- Processor (built for initial target size) ---
        self.processor = self._build_processor(self.size)

        # --- Parent init ---
        super().__init__(layout_cfg=self.layout_cfg, global_cfg=self.global_cfg)
        
    # ====[ BUILD PROCESSOR – Robust ND ResizeProcessor ]====
    def _build_processor(self, spatial_shape: Optional[Sequence[int]]) -> ImageProcessor:
        """
        Build and configure an ImageProcessor for ND image resizing.

        Automatically adapts the resizing function based on the current framework
        (Torch or NumPy), and ensures layout-awareness through axis tracking.

        Parameters
        ----------
        spatial_shape : tuple of int or None
            Desired spatial output shape (e.g., (H, W) or (D, H, W)).
            If None, the shape must be inferred later via `match_to`.

        Returns
        -------
        processor : ImageProcessor
            A layout-aware, backend-compatible processor ready to resize inputs.

        Notes
        -----
        - Torch backend:
            * Forces input format to (N, C, [D,] H, W).
            * Uses `F.interpolate` with mode "bilinear" or "trilinear".
            * Automatically restores original batch/channel axes.
        
        - NumPy backend:
            * Uses `cv2.resize` for 2D images.
            * Uses `skimage.transform.resize` for ND images.
            * Handles channel/batch axes explicitly for consistent results.
        
        - In both cases:
            * Singleton batch or channel axes are inserted and restored if missing.
            * The processor is built using current `LayoutConfig`, `GlobalConfig`,
            and `ImageProcessorConfig`.
            * Tags are updated with `status='resized'` and `shape_after`.

        Raises
        ------
        ValueError
            If the current framework is not supported.
        """
        if self.framework == "torch":
            @torch.no_grad()
            def resize_fn(img):
                # Ensure consistent axis format via ensure_format
                tracker = self.ensure_format(img,
                                            framework=self.framework,
                                            tag_as="resized",
                                            layout=self.layout_ensured,)  
                
                # Ensure batch and channel axes are at the beginning
                batch_axis = self.get_axis(tracker.get(),"batch_axis")
                channel_axis = self.get_axis(tracker.get(),"channel_axis")
                
                if channel_axis is not None and batch_axis is not None:                    
                    if not channel_axis == 1 and not batch_axis == 0:
                        # Move batch and channel axes to the front for resizing
                        tracker = tracker.moveaxis(channel_axis, 0)
                        tracker = tracker.moveaxis(batch_axis, 1)
                elif channel_axis is not None or batch_axis is not None:
                    axis = batch_axis or channel_axis
                    # Move the single axis to the front for resizing
                    tracker = tracker.moveaxis(axis, 0)
                    tracker = tracker.apply_to_all(lambda x: x.unsqueeze(0))
                else:
                    # No batch or channel axis, ensure at least 2D
                    tracker = tracker.apply_to_all(lambda x: x.unsqueeze(0).unsqueeze(0))
                              
                # Apply resize via PyTorch
                resized_tracker = tracker.apply_to_all(
                    F.interpolate,
                    size=spatial_shape,
                    mode=self.mode or ("trilinear" if len(spatial_shape) == 3 else "bilinear"),
                    align_corners=self.align_corners
                )
                
                # Restore original axes order
                if channel_axis is not None and batch_axis is not None:
                    # Move batch and channel axes back to their original positions
                    resized_tracker = resized_tracker.moveaxis(0, channel_axis)
                    resized_tracker = resized_tracker.moveaxis(1, batch_axis)
                elif channel_axis is not None or batch_axis is not None:
                    axis = batch_axis or channel_axis
                    resized_tracker = resized_tracker.apply_to_all(lambda x: x.squeeze(0))
                    # Move the single axis back to its original position
                    resized_tracker = resized_tracker.moveaxis(0, axis)
                else:
                    # No batch or channel axis, ensure at least 2D
                    resized_tracker = resized_tracker.apply_to_all(lambda x: x.squeeze(0).squeeze(0))
                
                resized_tracker.update_tags({"status": "resized", "shape_after": resized_tracker.image.shape})
                
                return resized_tracker.image

        elif self.framework == "numpy":

            def _resize_2d_numpy(arr2d: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
                h, w = int(out_hw[0]), int(out_hw[1]) 
                return cv2.resize(arr2d, dsize=(w, h), interpolation=cv2.INTER_LINEAR) # (W, H) for OpenCV dsize

            def _resize_nd_numpy(arr: np.ndarray, 
                                 out_spatial: Sequence[int] = None, 
                                 channel_ax: Optional[int] = None, 
                                 batch_ax: Optional[int] = None
                                 ) -> np.ndarray:
                """
                Generic ND resize using skimage.resize with channel_axis.
                out_spatial is (H, W) or (D, H, W).
                channel_ax: index of channel axis if any (relative to arr).
                batch_ax: index of batch axis if any (relative to arr).
                """
                if arr.ndim == 2 and len(out_spatial) == 2:
                    return _resize_2d_numpy(arr, (out_spatial[-2], out_spatial[-1]))

                # skimage.resize handles ND and channel_axis cleanly
                if batch_ax is None and channel_ax is not None:
                    results = sk_resize(
                    arr,
                    output_shape=tuple(out_spatial),
                    preserve_range=self.preserve_range,
                    anti_aliasing=self.anti_aliasing,
                    order=1,  # bilinear
                    )
                    
                elif batch_ax is not None:
                    
                    stacked: List[np.ndarray] = []
                    
                    for i in range(arr.shape[batch_ax]):
                        result = sk_resize(
                            arr[..., i],
                            output_shape=tuple(out_spatial),
                            preserve_range=self.preserve_range,
                            anti_aliasing=self.anti_aliasing,
                            order=1,  # bilinear
                        )
                        stacked.append(result)
                    
                    results = np.stack(stacked, axis=batch_ax)
                
                return results          
            
            def resize_fn(img: ArrayLike) -> np.ndarray:
                # Ensure consistent axis format via ensure_format
                tracker = self.ensure_format(img,
                                            framework=self.framework,
                                            tag_as="resized",
                                            layout=self.layout_ensured,)
                
                # Ensure batch and channel axes are at the end
                batch_axis = self.get_axis(tracker.get(),"batch_axis")
                channel_axis = self.get_axis(tracker.get(),"channel_axis")
                ndim = img.ndim
                
                ch_last: Optional[int] = None
                batch_last: Optional[int] = None
                
                 # Move batch and channel axes to the end for resizing
                if channel_axis is not None:
                    tracker = tracker.moveaxis(channel_axis, ndim - 1)
                    ch_last = ndim - 1
                if batch_axis is not None:
                    tracker = tracker.moveaxis(batch_axis, ndim - 1)
                    batch_last = ndim - 1
                    ch_last = ndim - 2 if channel_axis is not None else ndim - 1

                # Apply resize via scikit-image
                # resized_tracker = tracker.apply_to_all(
                #     sk_resize,
                #     output_shape=shape,
                #     preserve_range=self.preserve_range,
                #     anti_aliasing=self.anti_aliasing
                # )
                
                # Resize using OpenCV for better performance
                resized_tracker = tracker.apply_to_all(
                                                   _resize_nd_numpy,
                                                   out_spatial=spatial_shape,
                                                   channel_ax=ch_last,
                                                   batch_ax=batch_last
                                                   )
                
                # Restore original axes order
                if channel_axis is not None:
                    resized_tracker = resized_tracker.moveaxis(ndim - 1, channel_axis)
                if batch_axis is not None:
                    resized_tracker = resized_tracker.moveaxis(ndim - 1, batch_axis)
                    
                resized_tracker.update_tags({"status": "resized", "shape_after": resized_tracker.image.shape})                    
                    
                return resized_tracker.image

        else:
            raise ValueError(f"Unsupported framework '{self.framework}'. Use 'torch' or 'numpy'.")

        # Initialize ImageProcessor
        processor = ImageProcessor(
            img_process_cfg = self.img_process_cfg,
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg,
        )
    
        processor.function = resize_fn
        
        return processor
    
    def _get_axes(self, arr: ArrayLike) -> List[int]:
        """
        Determine the spatial axes for applying differential operators (e.g., gradients, divergence).

        Excludes non-spatial axes such as batch, channel, or direction axes,
        based on the image tag or internal axis configuration.

        Parameters
        ----------
        arr : np.ndarray or torch.Tensor
            Input image or volume, potentially tagged with layout metadata.

        Returns
        -------
        axes : list of int
            List of axis indices corresponding to spatial dimensions.

        Notes
        -----
        - For 2D inputs, returns all axes (i.e., [0, 1]) as fallback.
        - For higher-dimensional inputs, uses the image tag (if available) or
        `self.axes` to exclude:
            * 'channel_axis'
            * 'batch_axis'
            * 'direction_axis'
        - Supports negative axis indices via automatic normalization to positive values.
        - Preserves axis order as in the original data.
        """
        ndim = arr.ndim
        axes = list(range(ndim))

        if ndim == 2:
            return axes  # Default fallback for simple 2D

        tag = self.get_tag(arr, self.framework) if self.has_tag(arr, self.framework) else {}

        def to_positive(axis):
            return axis if axis is None or axis >= 0 else axis + ndim

        c_ax = to_positive(tag.get("channel_axis", self.axes.get("channel_axis")))
        b_ax = to_positive(tag.get("batch_axis", self.axes.get("batch_axis")))
        g_ax = to_positive(tag.get("direction_axis", self.axes.get("direction_axis")))

        for ax in (c_ax, b_ax, g_ax):
            if ax is not None and ax in axes:
                axes.remove(ax)

        return axes

    # ====[ CALL METHOD – ND Resize Operation ]====
    def __call__(self, image: ArrayLike, match_to: Optional[ArrayLike] = None) -> ArrayLike:
        """
        Resize the input image to a fixed spatial size or match a reference image.

        If `match_to` is provided, the spatial shape is inferred from its layout-aware
        axes and overrides `self.size`.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image or volume to resize.
        match_to : np.ndarray or torch.Tensor or None, optional
            Reference input from which to extract the spatial dimensions.
            Must have at least 2 spatial axes (e.g., H, W).

        Returns
        -------
        np.ndarray or torch.Tensor
            Resized image in the current framework (`self.framework`),
            preserving layout and dtype compatibility.

        Raises
        ------
        ValueError
            If `match_to` is provided but has fewer than 2 spatial dimensions.

        Notes
        -----
        - The resizing behavior is delegated to a dynamically built `ImageProcessor`.
        - If `match_to` is None, uses the fixed size configured in `self.resize_cfg`.
        - Axis layout is respected through `LayoutConfig`, including batch and channel reordering.
        - Output tags and trace metadata can optionally be propagated (commented section).
        """
        # ====[ Step 1: Determine processor according to match_to ]====
        if match_to is not None:
            # Ensure format before resizing
            spatial_axes = self._get_axes(match_to)
            spatial_shape = [match_to.shape[axis] for axis in spatial_axes]

            if len(spatial_shape) < 2:
                raise ValueError("match_to must have at least height and width axes defined.")

            processor = self._build_processor(tuple(spatial_shape))
        else:
            processor = self.processor

        # ====[ Step 2: Apply resizing via processor ]====
        resized = processor(image)

        # # ====[ Step 3: Robust conversion with complete tagging ]====
        # final_tracker = self.track(self.convert_once(
        #     image=resized,
        #     framework=self.output_format,
        #     tag_as=tag_as,
        #     enable_uid=enable_uid,
        #     op_params=op_params or {"resized_to": processor.function.__closure__[0].cell_contents}
        # ))

        # # ====[ Step 4: Update and propagate axis tags ]====
        # final_tracker.copy_tag_from(self.track(image), keys=['batch_axis', 'channel_axis', 'direction_axis', 
        #                                                      'height_axis', 'width_axis', 'depth_axis'])

        # return final_tracker.get()

        return resized

    # ====[ SUMMARY – ResizeOperator Configuration ]====
    def summary(self) -> None:
        """
        Display a detailed summary of ResizeOperator's current configuration.
        """
        print("[ResizeOperator] Configuration Summary:")
        print(f"  Resize Target      : {self.size}")
        print(f"  Framework          : {self.framework}")
        print(f"  Output Format      : {self.output_format}")
        print(f"  Device             : {self.device}")
        print(f"  Strategy           : {self.strategy}")
        print(f"  Normalize          : {self.normalize}")
        print(f"  Add Batch Dim      : {self.add_batch_dim}")
        print(f"  Interpolation Mode : {self.mode}")
        print(f"  Align Corners      : {self.align_corners}")
        print(f"  Preserve Range     : {self.preserve_range}")
        print(f"  Anti-Aliasing      : {self.anti_aliasing}")
        print(f"  Layout Name        : {self.layout_name}")

        print("  Axes Configuration:")
        for axis_name, axis_value in self.axes.items():
            print(f"    - {axis_name:<15}: {axis_value}")
