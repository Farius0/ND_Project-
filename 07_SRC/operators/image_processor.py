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
    Determine whether the input is a scalar or contains only scalar values.

    Supports native Python scalars, NumPy scalars, 0-dimensional arrays/tensors,
    and recursively checks lists/tuples of scalars.

    Parameters
    ----------
    x : Any
        Input object to test.

    Returns
    -------
    bool
        True if `x` is a scalar value or a container of scalars; False otherwise.

    Examples
    --------
    >>> is_scalar_value(3.14)
    True
    >>> is_scalar_value(np.array(5))
    True
    >>> is_scalar_value(torch.tensor(7.))
    True
    >>> is_scalar_value([1, 2, 3])
    True
    >>> is_scalar_value(np.array([1, 2, 3]))
    False
    >>> is_scalar_value("hello")
    False

    Notes
    -----
    - For NumPy or Torch arrays, only 0-dimensional or shape (1,) arrays are accepted as scalars.
    - Lists or tuples must contain only scalar values to return True.
    - Strings and structured types are not considered scalars.
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
    Multi-strategy ND-compatible image processor with backend awareness.

    Inherits from OperatorCore and supports configurable processing pipelines
    with automatic format preservation, layout tracking, and batch-wise handling.

    Notes
    -----
    - Dual-backend: NumPy in → NumPy out; Torch in → Torch out.
    - ND-ready: handles channel and batch axes via layout tagging.
    - Flexible: supports function injection, vectorized ops, parallel loops, and Torch-native ops.
    - Tag-preserving: all outputs inherit AxisTracker metadata for traceability.
    - Input validation, shape preservation, and layout conversion are automatic.
    """

    # ====[ INIT ]====
    def __init__(
        self,
        img_process_cfg: ImageProcessorConfig = ImageProcessorConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
        **kwargs: Any,
    ) -> None:
        """
        Initialize the image processor with layout, backend, and strategy configuration.

        Parameters
        ----------
        img_process_cfg : ImageProcessorConfig
            Defines the processing function, strategy ('auto', 'vectorized', 'torch', etc.),
            and various I/O behavior flags (e.g., return_tuple, fallback).
        layout_cfg : LayoutConfig
            Provides axis roles (channel, batch, feature) and layout expectations.
        global_cfg : GlobalConfig
            Controls framework, output format, device, verbosity, and backend preferences.
        kwargs : dict
            Additional keyword arguments passed to the OperatorCore base class.
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
        Apply the configured processing function to one or more ND images.

        Supports automatic dispatch based on processing strategy and layout-aware
        handling of batches, channels, and features. Preserves AxisTracker tags
        and backend compatibility.

        Parameters
        ----------
        images : np.ndarray or torch.Tensor
            One or more input images. All must have the same shape.

        Returns
        -------
        np.ndarray or torch.Tensor
            Processed image (or batch), with same shape and backend as inputs.
            Tags are preserved and updated if applicable.

        Raises
        ------
        ValueError
            If no image is provided or if shapes are inconsistent.
        TypeError
            If no callable processing function is configured.

        Notes
        -----
        - If `convert_inputs=True`, inputs are tracked and converted using layout config.
        - Strategy is chosen from `self.strategy`, or auto-detected if set to 'auto'.
        - If batch axis is detected, dispatches to `_process_batch`; else to `_process_images`.
        - This method serves as the main entry point for processing logic.
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
        Infer the most appropriate processing strategy based on input types.

        Determines whether to apply a Torch-based, vectorized NumPy, or classic fallback
        processing strategy depending on the types of all provided images.

        Parameters
        ----------
        images : Sequence of np.ndarray or torch.Tensor
            List of input images to inspect.

        Returns
        -------
        str
            One of the following:
            - 'torch'       : if all inputs are Torch tensors.
            - 'vectorized'  : if all inputs are NumPy arrays.
            - 'classic'     : if inputs are mixed NumPy and Torch (fallback).

        Raises
        ------
        ValueError
            If the input types are unsupported or inconsistent.
        
        Notes
        -----
        - This method is used when `strategy='auto'`.
        - Mixed inputs trigger a fallback to 'classic' strategy.
        - All inputs must be either array-like and of the same backend type, or mixed NumPy/Torch.
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
        Process a batch of ND images using a configured or custom function.

        Handles inputs with a batch axis (typically axis 0), and dispatches
        to the appropriate strategy: torch-native, vectorized NumPy, classic loop,
        or parallel processing.

        Parameters
        ----------
        images : list of np.ndarray or torch.Tensor
            Input images or tensors, all with a batch dimension.
            Shapes must be compatible across inputs.
        strategy : {'torch', 'vectorized', 'classic', 'parallel'}
            Processing strategy to use:
            - 'torch'      → call `_apply_torch` directly.
            - 'vectorized' → call `_apply_vectorized` directly.
            - 'classic'    → apply function slice-by-slice.
            - 'parallel'   → apply function slice-by-slice in parallel (joblib).
        func : callable, optional
            Optional override for the processing function (`self.function`).

        Returns
        -------
        np.ndarray or torch.Tensor
            Output with batch dimension preserved.
            AxisTracker tags are applied and updated per batch output.

        Raises
        ------
        ValueError
            If batch axis is not defined or if input types are unsupported.

        Notes
        -----
        - Automatically reorders batch axis to front if needed (N → 0).
        - Tags from inputs are tracked and restored on outputs.
        - If `img_process_fallback` is enabled and batch axis is already leading,
        a fast-path dispatch to `torch` or `vectorized` is used.
        - For 'classic' and 'parallel' modes, individual slices are tracked,
        processed, and reassembled via `_aggregate`.
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
        Apply slice-wise processing on ND images when no batch axis is present.

        Handles processing along the channel axis (if present), using the configured
        or specified strategy (torch, vectorized, classic, or parallel).
        Supports channel-aware reshaping, tag propagation, and backend dispatching.

        Parameters
        ----------
        images : list of np.ndarray or torch.Tensor
            Input images with the same shape. If `channel_axis` is defined and not 0,
            data is sliced along that axis for independent processing.
        strategy : {'torch', 'vectorized', 'classic', 'parallel'}
            Processing strategy to apply:
            - 'torch'      → backend-native operation using `_apply_torch`.
            - 'vectorized' → NumPy vectorized processing.
            - 'classic'    → sequential slice-by-slice execution.
            - 'parallel'   → parallel slice execution using joblib.
        func : callable, optional
            Override function to apply instead of `self.function`.

        Returns
        -------
        np.ndarray or torch.Tensor
            Processed output with reconstructed shape.
            Tags are preserved and updated (status='slice_output').

        Raises
        ------
        ValueError
            If input types are unsupported or shapes mismatch.

        Notes
        -----
        - If no channel axis is defined, falls back to fast-path full-image processing.
        - Automatically reorders axes so that slicing occurs along the first dimension (C → 0).
        - Each slice is tracked individually for metadata propagation.
        - Final outputs are reassembled and tagged using `_aggregate`.
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
        Stack, track, and apply the processing function using Torch strategy.

        Converts image slices to Torch tensors (if needed), stacks them along
        the leading dimension (channel or batch), and runs the function on-device.

        Parameters
        ----------
        images_swapped : list of Tensor or ndarray
            Inputs prepared for processing, with leading dimension ready for stacking
            (e.g., batch or channel axis moved to dim=0).
        originals : list of Tensor or ndarray
            Original images before axis reordering. Used for reconstructing final shape/tags.
        func : callable, optional
            Optional override for the processing function (`self.function`).

        Returns
        -------
        torch.Tensor or np.ndarray
            Final output after processing and reassembly.
            Tags are preserved and updated via AxisTracker.

        Raises
        ------
        TypeError
            If input slices are not NumPy arrays or Torch tensors.
        ValueError
            If image slice list is empty or None.

        Notes
        -----
        - Slices are moved to the appropriate device (as per `self.device`).
        - AxisTracker is used to tag the stacked tensor before processing.
        - The output is passed to `_reconstruct_output` for final reassembly and tag restoration.
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
        Stack and apply a NumPy vectorized function across the leading dimension.

        Designed for fast NumPy-native operations where batch or channel-wise
        processing can be applied on a stacked array. Tags and layout metadata
        are preserved via AxisTracker.

        Parameters
        ----------
        images_swapped : list of np.ndarray
            Inputs with leading axis (typically batch or channel) moved to dim=0.
        originals : list of np.ndarray
            Original unmodified images used for reconstructing the final layout and tags.
        func : callable, optional
            Custom function to apply instead of the default `self.function`.

        Returns
        -------
        np.ndarray or torch.Tensor
            Output after vectorized processing, reassembled with correct layout and tags.

        Raises
        ------
        TypeError
            If any image slice is not a NumPy array.
        ValueError
            If any image list is empty.

        Notes
        -----
        - This method is only valid for NumPy inputs. Torch tensors are not supported.
        - Inputs are stacked along axis 0 before being passed to the processing function.
        - Uses `AxisTracker.stack_from` to ensure tag propagation and layout consistency.
        - The output is passed through `_reconstruct_output` for final restoration.
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
        Rebuild the full ND output from per-slice or per-batch results, restoring axes and tags.

        This function reassembles the output tensor or array into its original layout 
        using metadata tracked during processing (e.g., axis positions, tags). It ensures 
        consistency in dimension order (batch/channel) and preserves axis semantics.

        Parameters
        ----------
        outputs : scalar | np.ndarray | torch.Tensor | list | tuple
            Output returned by the processing function. Can be a single value, ND array, 
            or list/tuple of slices to be stacked.
        originals : list of ArrayLike, optional
            Original input images used to determine correct axis restoration (channel/batch).
        swapped : list of ArrayLike, optional
            Inputs after channel/batch axis was moved to the front (dim=0); used for tag inheritance.

        Returns
        -------
        np.ndarray or torch.Tensor
            Final reconstructed output in the requested framework (`self.output_format`),
            with axes and layout tags correctly restored.

        Raises
        ------
        TypeError
            If the output is `None` or of an unsupported type.

        Notes
        -----
        - Handles both scalar and ND-array outputs.
        - Restores the correct `channel_axis` or `batch_axis` via `moveaxis` if necessary.
        - All outputs are wrapped and tracked using `AxisTracker` to preserve processing tags.
        - Supports automatic stacking for list/tuple of slices and tensor/array results.

        Examples
        --------
        >>> # Scalar case
        >>> _reconstruct_output(0.85)
        >>> # ND case with swapped shape (N, H, W)
        >>> _reconstruct_output([arr1, arr2], originals=[orig], swapped=[swp])
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
        Aggregate per-slice or per-batch results into a single ND output, restoring original layout and tags.

        This method consolidates the outputs produced slice-by-slice or batch-by-batch using the specified
    strategy (classic, parallel, etc.). It handles scalar outputs, array outputs, and optionally tuples of arrays.

        Parameters
        ----------
        results : list
            List of outputs from each processed slice or batch. Can contain:
            - Scalars (int, float, or 0D arrays/tensors),
            - ND arrays or tensors,
            - Tuples of ND arrays/tensors (if `self.return_tuple` is True).
        originals : list of ArrayLike, optional
            Original input images (before axis permutation), used to restore layout.
        swapped : list of ArrayLike, optional
            Inputs after channel/batch axis was moved to leading position; used for tag tracking.

        Returns
        -------
        np.ndarray or torch.Tensor or tuple
            Aggregated output reconstructed in the correct layout, with axis tags and metadata restored.

        Raises
        ------
        ValueError
            If the results list is empty.

        Notes
        -----
        - Axis restoration uses channel_axis or batch_axis based on the layout of `originals`.
        - If `self.return_tuple` is True, each element of the tuple is stacked and restored independently.
        - For scalar results, a tensor/array is created and tagged accordingly.
        - Tagging is handled via `AxisTracker` and `track_and_stack` utilities.

        Examples
        --------
        >>> # Example with scalar outputs
        >>> _aggregate([0.1, 0.2, 0.15])

        >>> # Example with 3D slices
        >>> _aggregate([slice1, slice2, slice3], originals=[original], swapped=[swapped])
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
        Create a tracked view of a slice extracted from a full ND image, with axis tag adjustment.

        This method propagates the AxisTracker metadata from the original full image (`full_img`)
        to the extracted slice (`slice_img`), while removing specific axes (e.g., 'C' for channel,
        'N' for batch, etc.) that are no longer relevant in the context of the slice.

        Parameters
        ----------
        full_img : np.ndarray or torch.Tensor
            Full input image containing the original `AxisTracker` metadata (e.g., axis roles, UID).
        slice_img : np.ndarray or torch.Tensor
            Slice extracted from `full_img`, typically during per-slice or per-batch processing.
        remove_axes : Iterable[str], default=("G", "N", "C")
            Axes to remove from the propagated metadata. Typical values:
            - 'C' for channel axis,
            - 'N' for batch axis,
            - 'G' for group axis.

        Returns
        -------
        AxisTracker
            Tagged slice image with updated metadata and reduced axis roles.

        Raises
        ------
        ValueError
            If either `full_img` or `slice_img` is None.

        Notes
        -----
        - The propagated metadata includes axis mapping, layout name, UID, and processing status.
        - This function is central to per-slice or per-batch dispatching in `ImageProcessor`.
        - Internally uses `AxisTracker.from_sliced`.

        Examples
        --------
        >>> tracker = image_processor.track_slice(full_img, full_img[0], remove_axes=("C",))
        """
        if full_img is None or slice_img is None:
            raise ValueError("Both full_img and slice_img must be provided.")

        return self.track(None).from_sliced(
            full_img, slice_img, self, self.framework, remove_axes=list(remove_axes)
        )

    # ====[ Slice Stacking Tracker Shortcut ]====
    def track_and_stack(self, ref: ArrayLike, slices: Sequence[ArrayLike], axis: int = 0, status: str = "aggregated") -> ArrayLike:
        """
        Stack a list of slices into a single array/tensor and propagate AxisTracker metadata.

        This method stacks multiple slices (e.g., per-channel or per-batch outputs)
        and transfers the AxisTracker tag from a reference array (`ref`) to the newly
        stacked object. Axis metadata, status and output shape are updated accordingly.

        Parameters
        ----------
        ref : np.ndarray or torch.Tensor
            Reference input used to retrieve original AxisTracker metadata.
        slices : list of np.ndarray or torch.Tensor
            Slices to be stacked along a new or existing axis.
        axis : int, default=0
            Axis along which to stack the slices.
        status : str, default="aggregated"
            Status tag to assign to the resulting stacked object (e.g., "stacked", "aggregated").

        Returns
        -------
        AxisTracker
            Tracked object wrapping the stacked array or tensor, with updated metadata.

        Raises
        ------
        ValueError
            If `slices` is empty or contains a None element.

        Notes
        -----
        - The resulting object retains the axis metadata (`AxisMap`) of the reference.
        - The `shape_after` tag is updated based on the stacked shape.
        - This is commonly used to reconstruct outputs in `_aggregate` or `_reconstruct_output`.

        Examples
        --------
        >>> stacked = image_processor.track_and_stack(ref=img, slices=[img1, img2], axis=0, status="reconstructed")
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
        Ensure that the given image has an attached AxisTracker tag.

        This method verifies that the input image (NumPy array or Torch tensor)
        is associated with a valid AxisTracker. If the tag is missing, an informative
        RuntimeError is raised to prevent further processing without axis metadata.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Image or tensor to check for AxisTracker tagging.

        Raises
        ------
        RuntimeError
            If the input is not tagged with AxisTracker for the current framework.

        Notes
        -----
        - This check is used internally to enforce metadata consistency.
        - The framework is inferred from `self.framework`, defaulting to 'torch'.
        - Typically used before axis-aware operations such as moveaxis, reshape, or slicing.

        Examples
        --------
        >>> self._assert_tracked(my_tensor)  # Raises if not tagged
        """
        fw = self.framework or "torch"
        if not self.has_tag(image, fw):
            raise RuntimeError(
                f"[ImageProcessor] Missing AxisTracker tag on image "
                f"(type: {type(image)}, shape: {getattr(image, 'shape', 'N/A')})."
            )

    def _infer_stack_key(self) -> str:
        """
        Infer the axis key used for stacking operations.

        Returns
        -------
        str
            'batch_axis' if defined in self.axes, otherwise 'channel_axis'.

        Notes
        -----
        This utility determines the most appropriate axis for stacking slices
        during image processing. It prioritizes the batch axis when available,
        falling back to the channel axis otherwise.
        """
        return "batch_axis" if self.axes.get("batch_axis", None) is not None else "channel_axis"

    def map(self, *batches: ArrayLike) -> ArrayLike:
        """
        Apply the processor element-wise to a batch (or multiple batches).

        Iterates over the first axis ("N") of each input and applies the configured
        processor individually to each slice, tracking axes and metadata.

        Parameters
        ----------
        *batches : np.ndarray or torch.Tensor
            One or more ND batches. Each batch must be iterable along its first axis
            (i.e., shape (N, ...)).

        Returns
        -------
        np.ndarray or torch.Tensor
            Batch of processed results, with shape and backend matching the input.

        Notes
        -----
        - Uses `track_slice` to preserve axis tags and UID information per image.
        - The leading batch axis ("N") is ignored for tagging consistency.
        - `sync_axes_from_tag` ensures all slices use a common axis layout before processing.
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