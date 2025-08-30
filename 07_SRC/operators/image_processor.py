# ==================================================
# ============ MODULE:image_processor ==============
# ==================================================
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Sequence, Dict, Union, Literal
from joblib import Parallel, delayed

import numpy as np
import torch

from core.operator_core import OperatorCore
from core.config import LayoutConfig, GlobalConfig, ImageProcessorConfig
# from utils.decorators import safe_timer

# Public API
__all__ = ["ImageProcessor"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Framework = Literal["numpy", "torch"]

# ==================================================
# ================== Utilities =====================
# ==================================================
def is_scalar_value(x) -> bool:
    """
    Return True if x is a scalar (Python scalar, 0-d array/tensor, or scalars
    inside a list/tuple), False otherwise.
    """
    if isinstance(x, (int, float, complex, np.generic)):
        return True
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return is_scalar_value(x.item()) if (x.ndim == 0 or x.shape == (1,)) else False
    if isinstance(x, (list, tuple)):
        return all(is_scalar_value(i) for i in x)
    return False

# ==================================================
# ================ ImageProcessor ==================
# ==================================================
class ImageProcessor(OperatorCore):
    """
    Multi-strategy image processor (torch / vectorized / classic / parallel).

    Notes
    -----
    - Dual-backend: NumPy in → NumPy out; Torch in → Torch out.
    - ND-ready: channel/batch aware; slice/batch processing supported.
    - Tag-preserving: AxisTracker metadata propagated to outputs.
    """

    # ====[ INIT ]====
    def __init__(
        self,
        *,
        img_process_cfg: ImageProcessorConfig = ImageProcessorConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
        **kwargs: Any,
    ) -> None:
        """
        Initialize image processor with strategy and layout/global configs.

        Parameters
        ----------
        img_process_cfg : ImageProcessorConfig
            Processing strategy, function handle, and IO options.
        layout_cfg : LayoutConfig
            Axis roles and preferred layout.
        global_cfg : GlobalConfig
            Backend, device, normalization and verbosity flags.
        """
        # --- Config mirrors ---
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg
        self.img_process_cfg: ImageProcessorConfig = img_process_cfg

        # --- Resolved axes & layout meta ---
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

        # --- Global mirrors ---
        self.framework: Framework = self.global_cfg.framework.lower()
        self.output_format: Framework = self.global_cfg.output_format.lower()
        self.add_batch_dim: Optional[bool] = self.global_cfg.add_batch_dim
        self.add_channel_dim: Optional[bool] = self.global_cfg.add_channel_dim
        self.normalize: bool = bool(self.global_cfg.normalize)
        self.verbose: bool = bool(self.global_cfg.verbose)
        self.backend: str = self.global_cfg.backend        
        self.device: str = (
            "cuda"
            if (torch.cuda.is_available() and self.framework == "torch")
            else self.global_cfg.device
        )

        # --- Processor-specific ---
        self.function: Callable[..., Any] = self.img_process_cfg.function
        # If user asked "vectorized" but framework is torch, prefer torch fast path
        self.strategy: str = (
            "torch"
            if (self.framework == "torch" and self.img_process_cfg.processor_strategy == "vectorized")
            else self.img_process_cfg.processor_strategy
        )
        self.n_jobs: int = self.img_process_cfg.n_jobs
        self.return_tuple: Optional[bool] = self.img_process_cfg.return_tuple
        self.convert_inputs: Optional[bool] = self.img_process_cfg.convert_inputs
        self.return_type: Optional[bool] = self.img_process_cfg.return_type
        self.img_process_fallback: Optional[Union[str, bool]] = self.img_process_cfg.fallback
        self.kwargs: Dict[str, Any] = kwargs

        # --- Parent init ---
        super().__init__(layout_cfg=self.layout_cfg, global_cfg=self.global_cfg)

    # ====[ CALLABLE ENTRY POINT ]====
    def __call__(self, *images: ArrayLike) -> ArrayLike:
        """
        Apply the configured function to one or more images.

        Parameters
        ----------
        images : ndarray | Tensor
            One or more images to process (same shape).

        Returns
        -------
        ndarray | Tensor
            Processed output in the desired format; tags preserved.
        """
        if len(images) < 1:
            raise ValueError("At least one image must be provided.")
        if not callable(self.function):
            raise TypeError("Processor function must be callable.")

        # Shape consistency (lightweight)
        base_shape = images[0].shape
        for img in images:
            if img.shape != base_shape:
                raise ValueError("All input images must have the same shape.")

        # Optional conversion + tagging
        if self.convert_inputs:
            images = [
                self.track(
                    self.convert_once(img, tag_as="input", enable_uid=self.verbose)
                ).get()
                for img in images
            ]

        strategy_used = self._detect_strategy(images) if self.strategy == "auto" else self.strategy

        # Route by batch presence
        return (
            self._process_batch(*images, strategy=strategy_used, func=None)
            if self.get_axis(images[0], "batch_axis") is not None
            else self._process_images(*images, strategy=strategy_used, func=None)
        )

    # ====[ STRATEGY AUTO-DETECTION ]====
    def _detect_strategy(self, images: Sequence[ArrayLike]) -> str:       
        """
        Infer a safe processing strategy from input types.
        """
        if all(isinstance(x, torch.Tensor) for x in images):
            return "torch"
        if all(isinstance(x, np.ndarray) for x in images):
            return "vectorized"
        if all(isinstance(x, (np.ndarray, torch.Tensor)) for x in images):
            return "classic"
        raise ValueError("Unsupported image types for strategy detection.")

    # ====[ BATCH PROCESSING ]====
    @torch.no_grad()
    def _process_batch(self, *images: ArrayLike, strategy: str, func: Optional[Callable] = None):
        """
        Process images with a batch axis (N).

        Parameters
        ----------
        images : list[ndarray | Tensor]
            Inputs with a batch axis.
        strategy : {'torch','vectorized','classic','parallel'}
            Processing strategy.
        func : callable, optional
            Alternative function to apply instead of self.function.

        Returns
        -------
        ndarray | Tensor
            Aggregated output with tags restored.
        """
        batch_axis = self.get_axis(images[0], "batch_axis")
        if batch_axis is None:
            raise ValueError("batch_axis must be defined for batch processing.")

        # Fast path if batch already leading and fallback enabled
        if batch_axis == 0 and self.img_process_fallback:
            if all(isinstance(x, torch.Tensor) for x in images):
                return self._apply_torch(images, originals=images, func=func)
            if all(isinstance(x, np.ndarray) for x in images):
                return self._apply_vectorized(images, originals=images, func=func)
            raise ValueError("Unsupported image types for processing.")
                        
        # elif self.strategy is None:
        #     outputs = self.function(*images, **self.kwargs) 
        #     tagger = self.track(images[0])  
        #     tracker = tagger.clone_to(outputs, updates = 
        #                         {"status": "batch_output", 
        #                         "shape_after": outputs.shape})  
        #     return tracker.get()
        
        # Move N → 0 for iteration
        images_batched = (
            [self.track(img).moveaxis(batch_axis, 0).get() for img in images]
            if batch_axis != 0
            else images
        )
        num_batches = images_batched[0].shape[0]

        def process_batch(i: int) -> Any:
            batch_inputs = [self.track_slice(img, img[i], remove_axes=["N"]) for img in images_batched]
            out = (func or self.function)(*batch_inputs, **self.kwargs)

            if not isinstance(out, (torch.Tensor, np.ndarray)):
                return out

            if not self.has_tag(out, self.framework):
                out = self.track(batch_inputs[0]).copy_to(out)
                out.update_tag("status", "batch_output")
                out = out.get()
            return out

        # Dispatch
        if strategy == "torch":
            return self._apply_torch(images_batched, originals=images, func=func)
        if strategy == "vectorized":
            return self._apply_vectorized(images_batched, originals=images, func=func)
        if strategy == "classic":
            results = [process_batch(i) for i in range(num_batches)]
            return self._aggregate(results, originals=images, swapped=images_batched)
        if strategy == "parallel":
            results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(process_batch)(i) for i in range(num_batches)
            )
            return self._aggregate(results, originals=images, swapped=images_batched)
        raise ValueError(f"Unsupported processing strategy: '{strategy}'")


    # ====[ SLICE/CHANNEL PROCESSING ]====
    @torch.no_grad()
    def _process_images(self, *images: ArrayLike, strategy: str, func: Optional[Callable] = None):
        """
        Slice-wise (channel-first) processing when there is no batch axis.

        Parameters
        ----------
        images : list[ndarray | Tensor]
            Same-shape inputs; first dimension is sliced if channel_axis != 0.
        strategy : {'torch','vectorized','classic','parallel'}
            Processing strategy.
        func : callable, optional
            Alternative function to apply.

        Returns
        -------
        ndarray | Tensor
            Reconstructed output with tags.
        """
        channel_axis = self.get_axis(images[0], "channel_axis")
        # Fallback to fast path when no channel axis (treat as single-slice)
        self.img_process_fallback = True if channel_axis is None else self.img_process_fallback

        if channel_axis in (0, None) and self.img_process_fallback:
            if all(isinstance(x, torch.Tensor) for x in images):
                return self._apply_torch(images, originals=images, func=func)
            if all(isinstance(x, np.ndarray) for x in images):
                return self._apply_vectorized(images, originals=images, func=func)
            raise ValueError("Unsupported image types for processing.")
            
        # elif self.strategy is None:
        #     outputs = self.function(*images, **self.kwargs) 
        #     tagger = self.track(images[0])  
        #     tracker = tagger.clone_to(outputs, updates = 
        #                         {"status": "slice_output", 
        #                         "shape_after": outputs.shape})  
        #     return tracker.get()            

        # Move C → 0 for slicing
        images_swapped = (
            [self.track(img).moveaxis(channel_axis, 0).get() for img in images]
            if channel_axis not in (0, None)
            else images
        )
        n_slices = images_swapped[0].shape[0]

        def process_slice(i: int) -> Any:
            slice_inputs = [self.track_slice(img, img[i], remove_axes=["C"]) for img in images_swapped]
            out = (func or self.function)(*slice_inputs, **self.kwargs)

            if not isinstance(out, (torch.Tensor, np.ndarray)):
                return out

            if not self.has_tag(out, self.framework):
                out = self.track(slice_inputs[0]).copy_to(out)
                out.update_tag("status", "slice_output")
                out = out.get()
            return out

        # Dispatch
        if strategy == "torch":
            return self._apply_torch(images_swapped, originals=images, func=func)
        if strategy == "vectorized":
            return self._apply_vectorized(images_swapped, originals=images, func=func)
        if strategy == "parallel":
            results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(process_slice)(i) for i in range(n_slices)
            )
        elif strategy == "classic":
            results = [process_slice(i) for i in range(n_slices)]
        else:
            raise ValueError(f"Unsupported processing strategy: '{strategy}'")

        return self._aggregate(results, originals=images, swapped=images_swapped)

    # ====[ Torch Strategy Execution ]====
    @torch.no_grad()
    def _apply_torch(self, images_swapped: List[ArrayLike], originals: List[ArrayLike], func: Optional[Callable] = None):
        """
        Stack along leading dim and process on device (Torch strategy).

        Parameters
        ----------
        images_swapped : list[Tensor | ndarray]
            Inputs with leading dim = channel or batch.
        originals : list[Tensor | ndarray]
            Original full images.
        func : callable, optional
            Alternative function to apply.

        Returns
        -------
        Tensor | ndarray
            Reconstructed output with tags.
        """
        stacked : List[torch.Tensor] = []

        for img_list in images_swapped:
            if img_list is None:
                raise ValueError("Image slice list is empty.")

            # Convert to torch and move to device
            if isinstance(img_list[0], torch.Tensor):
                slices = [s.to(self.device) for s in img_list]
            elif isinstance(img_list[0], np.ndarray):
                slices = [torch.from_numpy(s).to(self.device) for s in img_list]
            else:
                raise TypeError("Unsupported slice type in _apply_torch.")

            # Stack and track
            tracker = self.track(img_list)
            tensor = torch.stack(slices)

            stacked_tensor = tracker.stack_from(
                slices,
                axis=0,
                update_tags={"status": "stacked", "shape_after": tensor.shape,}
            ).get()

            stacked.append(stacked_tensor)

        outputs = (func or self.function)(*stacked, **self.kwargs)
            
        return self._reconstruct_output(outputs, originals=originals, swapped=images_swapped)

    # ====[ Vectorized Strategy Execution ]====
    def _apply_vectorized(self, images_swapped: List[ArrayLike], originals: List[ArrayLike], func: Optional[Callable] = None):
        """
        Stack and apply a vectorized function across leading dim (NumPy).

        Parameters
        ----------
        images_swapped : list[ndarray]
            Inputs with leading dim = channel or batch.
        originals : list[ndarray]
            Original full images.
        func : callable, optional
            Alternative function to apply.

        Returns
        -------
        ndarray | Tensor
            Reconstructed output with tags.
        """
        stacked : List[np.ndarray] = []

        for img_list in images_swapped:
            if img_list is None:
                raise ValueError("Image slice list is empty.")

            if not all(isinstance(s, np.ndarray) for s in img_list):
                raise TypeError("Vectorized strategy requires NumPy slices.")

            tracker = self.track(img_list)
            array = np.stack(img_list)

            stacked_array = tracker.stack_from(
                img_list,
                axis=0,
                update_tags={"status": "stacked", "shape_after": array.shape,}
            ).get()

            stacked.append(stacked_array)
        
        outputs = (func or self.function)(*stacked, **self.kwargs)
            
        return self._reconstruct_output(outputs, originals=originals, swapped=images_swapped)

    # ====[ OUTPUT RECONSTRUCTION ]====
    def _reconstruct_output(self, outputs: Any, originals: Optional[List[ArrayLike]] = None, swapped: Optional[List[ArrayLike]] = None):
        """
        Rebuild a full output from per-slice/batch results with tag consistency.

        Parameters
        ----------
        outputs : scalar | ndarray | Tensor | list | tuple
            Result from the processing function.
        originals : list, optional
            Original images (for axis restoration).
        swapped : list, optional
            Inputs with leading slice dim.

        Returns
        -------
        ndarray | Tensor
            Final output in self.output_format with tag.
        """
        if outputs is None:
            raise TypeError("[ImageProcessor] Function returned None.")

        # --- Scalar or tuple/list of scalars ---
        if is_scalar_value(outputs) or (isinstance(outputs, (list, tuple)) \
            and all(is_scalar_value(o) for o in outputs)):
            
            if self.framework == "numpy":
                outputs = np.asarray(outputs, dtype=np.float32)
            else:
                outputs = torch.tensor(outputs, dtype=torch.float32, device=self.device)
            
            if outputs.ndim == 0:
                outputs = outputs[None]  # shape (1,)

            tracker = self.track(swapped[0]).copy_and_tag_scalar(outputs)
            tracker.update_tags({"status": "reconstructed", "shape_after": outputs.shape,})
            return self.to_output(tracker.get(), tag_as="output")

        # --- ND array (Torch or NumPy) ---
        if isinstance(outputs, (np.ndarray, torch.Tensor)):
            tracker = self.track(swapped[0]).copy_to(outputs)
            channel_axis = self.get_axis(originals[0], "channel_axis")
            batch_axis = self.get_axis(originals[0], "batch_axis")

            if channel_axis not in (0, None) and batch_axis is None:
                tracker = tracker.moveaxis(0, channel_axis)
            elif batch_axis not in (0, None):
                tracker = tracker.moveaxis(0, batch_axis)
            else:
                pass

            tracker.update_tags({"status": "reconstructed",})
            return self.to_output(tracker.get(), tag_as="output")

        # --- List/tuple of nd arrays ---
        if isinstance(outputs, (list, tuple)) and outputs and isinstance(outputs[0], (np.ndarray, torch.Tensor)):
            tracker = self.track(swapped[0])
            channel_axis = self.get_axis(originals[0],"channel_axis")
            batch_axis = self.get_axis(originals[0],"batch_axis")

            stacked = tracker.stack_from(
                outputs,
                axis=0,
                update_tags={
                    "status": "reconstructed",
                    "shape_after": torch.stack(outputs).shape if isinstance(outputs[0], torch.Tensor) else np.stack(outputs).shape,
                })

            if channel_axis not in (0, None) and batch_axis is None:
                stacked = stacked.moveaxis(0, channel_axis)
            elif batch_axis not in (0, None):
                stacked = stacked.moveaxis(0, batch_axis)
            else:
                pass

            return self.to_output(stacked.get(), tag_as="output")

        # === Unsupported output ===
        raise TypeError(f"[ImageProcessor] Unsupported output type: {type(outputs)}.")

    # ====[ AGGREGATION ]====
    def _aggregate(self, results: List[Any], originals: Optional[List[ArrayLike]] = None, swapped: Optional[List[ArrayLike]] = None):
        """
        Aggregate per-slice/batch results (scalars, arrays, or tuples of arrays).

        Parameters
        ----------
        results : list
            Outputs produced per slice/batch.
        originals : list, optional
            Original images for axis restoration.
        swapped : list, optional
            Inputs with leading slice dim.

        Returns
        -------
        ndarray | Tensor | tuple
            Aggregated output, tags preserved.
        """
        if not results:
            raise ValueError("Cannot aggregate an empty list of results.")

        channel_axis = self.get_axis(originals[0], "channel_axis")
        batch_axis = self.get_axis(originals[0], "batch_axis")

        # --- Scalar reduction ---
        if is_scalar_value(results) or (isinstance(results, (list, tuple)) \
            and all(is_scalar_value(o) for o in results)):
            
            if self.framework == "numpy":
                results = np.asarray(results, dtype=np.float32)
            else:
                results = torch.tensor(results, dtype=torch.float32, device=self.device)
            
            if results.ndim == 0:
                results = results[None]  # shape (1,)

            tracker = self.track(swapped[0]).copy_and_tag_scalar(results)
            tracker.update_tags({"status": "reconstructed", "shape_after": results.shape,})
            return self.to_output(tracker.get(), tag_as="output")

        # --- Tuple outputs preserved element-wise ---
        if self.return_tuple:
            grouped = list(zip(*results))
            def _stack_and_restore(group):
                stacked = self.track_and_stack(swapped[0], group, axis=0, status="aggregated")
                if channel_axis not in (0, None) and batch_axis is None:
                    stacked = stacked.moveaxis(0, channel_axis)
                elif batch_axis not in (0, None):
                    stacked = stacked.moveaxis(0, batch_axis)
                return self.to_output(stacked.get(), tag_as="output")

            return tuple(_stack_and_restore(group) for group in grouped)

        # === Default ND case ===
        tracker = self.track_and_stack(swapped[0], results, axis=0, status="aggregated")
        
        if batch_axis is None and channel_axis not in (0, None):
            tracker = tracker.moveaxis(0, channel_axis)
        elif batch_axis not in (0, None):
            tracker = tracker.moveaxis(0, batch_axis)
        else:
            pass
            
        return self.to_output(tracker.get(), tag_as="output")

    # ====[ SLICE TRACKER SHORTCUTS ]====
    def track_slice(self, full_img: ArrayLike, slice_img: ArrayLike, remove_axes: Iterable[str] = ("G", "N", "C")) -> ArrayLike:
        """
        Build a tracked slice from a full image, removing specified axes.

        Parameters
        ----------
        full_img : ndarray | Tensor
            Full image holding the original tag.
        slice_img : ndarray | Tensor
            Slice extracted from full_img.
        remove_axes : tuple[str]
            Axes to drop from the tag (e.g., 'C' or 'N').

        Returns
        -------
        AxisTracker
            Tracked slice wrapper.
        """
        if full_img is None or slice_img is None:
            raise ValueError("Both full_img and slice_img must be provided.")

        return self.track(None).from_sliced(
            full_img, slice_img, self, self.framework, remove_axes=list(remove_axes)
        )

    # ====[ Slice Stacking Tracker Shortcut ]====
    def track_and_stack(self, ref: ArrayLike, slices: Sequence[ArrayLike], axis: int = 0, status: str = "aggregated") -> ArrayLike:
        """
        Stack slices with tag propagation.

        Parameters
        ----------
        ref : ndarray | Tensor
            Reference array/tensor to inherit tag from.
        slices : list[ndarray | Tensor]
            List of slices to stack.
        axis : int
            Axis along which to stack.
        status : str
            Tag status to set.

        Returns
        -------
        AxisTracker
            Tracker wrapping the stacked array/tensor.
        """
        if not slices or slices[0] is None:
            raise ValueError("Invalid or empty slice list for stacking.")

        tracker = self.track(ref)
        shape = torch.stack(slices).shape if isinstance(ref, torch.Tensor) else np.stack(slices).shape

        return tracker.stack_from(
            slices, axis=axis, update_tags={"status": status, "shape_after": shape}
        )

    # ====[ ASSERTION / TOOLS / SUMMARY ]====
    def _assert_tracked(self, image: ArrayLike) -> None:
        """
        Ensure the image has an AxisTracker tag; raise otherwise.
        """
        fw = self.framework or "torch"
        if not self.has_tag(image, fw):
            raise RuntimeError(
                f"[ImageProcessor] Missing AxisTracker tag on image "
                f"(type: {type(image)}, shape: {getattr(image, 'shape', 'N/A')})."
            )

    def _infer_stack_key(self) -> str:
        """
        Return stack axis key based on resolved axes.
        """
        return "batch_axis" if self.axes.get("batch_axis", None) is not None else "channel_axis"

    def map(self, *batches: ArrayLike) -> ArrayLike:
        """
        Map the processor to a batch of images (apply per element).

        Parameters
        ----------
        *batches : ndarray | Tensor
            Batches to process (iterate over leading N).

        Returns
        -------
        ndarray | Tensor
            Processed batch.
        """
        slices_and_tagged = [
            self.track_slice(images, image, remove_axes=["N"]) for images in batches for image in images
        ]
        self.sync_axes_from_tag(slices_and_tagged[0], override_axes=True)
        return self(*slices_and_tagged)

    def summary(self) -> None:
        """
        Print a concise summary of the processor configuration.
        """
        print("=== ImageProcessor Summary ===")
        print(f"Function        : {getattr(self.function, '__name__', str(self.function))}")
        print(f"Strategy        : {self.strategy}")
        print(f"Framework       : {self.framework}")
        print(f"Output format   : {self.output_format}")
        print(f"Return type     : {self.return_type}")
        print(f"Return tuple    : {self.return_tuple}")
        print(f"Convert inputs  : {self.convert_inputs}")
        print(f"Normalize       : {self.normalize}")
        print(f"Add batch dim   : {self.add_batch_dim}")
        print(f"n_jobs          : {self.n_jobs}")
        print(f"Backend         : {self.backend}")
        print("Axes:")
        for axis, value in self.axes.items():
            print(f"  {axis:<15}: {value}")