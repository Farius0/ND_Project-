# ==================================================
# =============  MODULE: edge_detector  ============
# ==================================================
from __future__ import annotations

from functools import partial
import numpy as np, torch

from skimage.filters import threshold_otsu
from typing import Dict, List, Optional, Union, Any, Sequence, Literal

from core.operator_core import OperatorCore
from core.layout_axes import resolve_and_clean_layout_tags
from core.config import (LayoutConfig, GlobalConfig, EdgeDetectorConfig, 
                         NDConvolverConfig, ImageProcessorConfig, DiffOperatorConfig,)
from operators.diff_operator import DiffOperator
from operators.image_processor import ImageProcessor
from operators.gaussian import (NDConvolver as convolver,
                                GaussianKernelGenerator as kernel_generator)

# Public API
__all__ = ["EdgeDetector", "edge_detect"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Framework = Literal["numpy", "torch"]

class EdgeDetector(OperatorCore):
    """
    N-dimensional edge detector supporting multiple strategies (gradient, sign change,
    Laplacian zero-crossing, combined, Marr-Hildreth, Canny) with dual-backend (NumPy/Torch)
    and layout-aware tagging.

    Notes
    -----
    - Preserves backend: NumPy in → NumPy out; Torch in → Torch out.
    - Tagging pipeline records status, shapes, layout_name, and axis map.
    - 2D/3D first-class; higher-D partially supported when meaningful.
    """

    def __init__(
        self,
        *,
        edge_detector_cfg: EdgeDetectorConfig = EdgeDetectorConfig(),
        diff_operator_cfg: DiffOperatorConfig = DiffOperatorConfig(),
        ndconvolver_cfg: NDConvolverConfig = NDConvolverConfig(),
        img_process_cfg: ImageProcessorConfig = ImageProcessorConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:

        # ====[ Configuration ]====
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg
        self.conv_cfg: NDConvolverConfig = ndconvolver_cfg
        self.diff_cfg: DiffOperatorConfig = diff_operator_cfg
        self.img_process_cfg: ImageProcessorConfig = img_process_cfg
        self.edge_detect_cfg: EdgeDetectorConfig = edge_detector_cfg
        
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
        
        # ====[ Store edge-specific parameters ]====
        self.method: str = self.edge_detect_cfg.edge_strategy
        self.eta: Optional[Union[str, float]] = self.edge_detect_cfg.eta
        self.as_float: bool = bool(self.edge_detect_cfg.as_float)
        self.dtype: Optional[Union[type, str]] = self.edge_detect_cfg.dtype 
        self.mode: str = self.edge_detect_cfg.mode
        self.alpha: float = float(self.edge_detect_cfg.alpha)
        self.threshold: Optional[Union[str, float]] = self.edge_detect_cfg.threshold
        self.complete_nms: bool = bool(self.edge_detect_cfg.complete_nms)
        self.use_paddingbool = bool(self.edge_detect_cfg.use_padding)
    
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
        self.conv_strategy: str = self.conv_cfg.conv_strategy
        self.diff_strategy: str = self.diff_cfg.diff_strategy
        self.backend: str = self.global_cfg.backend
        self.dim: int = int(self.conv_cfg.dim)
        self.size: Union[int, Sequence[int]] = self.conv_cfg.size
        self.sigma: Union[float, Sequence[float]] = self.conv_cfg.sigma
        self.angle: Union[float, Sequence[float]] = self.conv_cfg.angle
        
        # ====[ Initialize OperatorCore with all axes ]====
        super().__init__(
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg,
        )  
        
        # ====[ Create KernelGenerator ]====
        self.kernel_gen: kernel_generator = kernel_generator(
            layout_cfg = LayoutConfig(layout_name="HW" if self.dim == 2 else "DHW", layout_framework=self.framework),
            global_cfg = self.global_cfg.update_config(output_format=self.framework),
        )
        
        self.kernel: ArrayLike = self.kernel_gen.generate(
        dim=self.dim,
        size=self.size, 
        sigma=self.sigma,  
        angle=self.angle, 
        symmetry=False
        )     
        
        # @property
        # def kernel(self, dim=dim, size=size, sigma=sigma, angle=angle):
        #     if not hasattr(self, "_kernel"):
        #         self._kernel = self.kernel_gen.generate(
        #         dim=dim,
        #         size=size, 
        #         sigma=sigma,  
        #         angle=angle, 
        #         symmetry=False
        #         )
        #     return self._kernel

        # ====[ Create ImageProcessor ]====
        self.processor: ImageProcessor = ImageProcessor(
            img_process_cfg = self.img_process_cfg,
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg,
        )
        
        # ====[ Create Convolver ]====
        self.convolve: convolver = convolver(
            ndconvolver_cfg = self.conv_cfg,
            img_process_cfg = self.img_process_cfg,
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg,
      )
            
        # ====[ Create DiffOperator ]====
        self.diff: DiffOperator = DiffOperator(
            diff_operator_cfg = self.diff_cfg,
            ndconvolver_cfg = self.conv_cfg,
            layout_cfg = self.layout_cfg.update_config(layout=None, layout_name=None, 
                                            layout_framework=None,),
            global_cfg = self.global_cfg,
        )      

    def __call__(self, u: ArrayLike) -> ArrayLike:
        """
        Run the configured edge detector on an input image/tensor.

        Parameters
        ----------
        u : ndarray | Tensor
            Input image/volume.

        Returns
        -------
        ndarray | Tensor
            Edge map in the same backend, tagged with status='edges'.
        """
        u = self.convert_once(u, tag_as="input")
        result = self._detect(u)
        return self.to_output(result, tag_as="edges")

    def _detect(self, u: ArrayLike) -> ArrayLike:
        methods = {
            "gradient"     : self._gradient_edges,
            "sobel_gradient": self._gradient_edges,
            "sign_change"  : self._sign_change_edges,
            "laplacian"    : self._laplacian_edge,
            "combined"     : self._combined_edge,
            "marr_hildreth": self._marr_hildreth,
            "canny"        : self._canny_edges,
        }
        if self.method not in methods:
            raise ValueError(f"Unknown edge detection method: {self.method}")
        
        if self.method == "sobel_gradient":
            return methods[self.method](u, gradient="sobel")
        
        return methods[self.method](u)
    
    # -------- Gradient-based edges --------
    def _gradient_edges(self, image: ArrayLike, gradient: str = "classic") -> ArrayLike:        
        partial_func = partial(self._detect_edges_numpy, gradient=gradient) if self.framework == "numpy" \
            else partial(self._detect_edges_torch, gradient=gradient)
        
        self.processor.function = partial_func
        result = self.processor(image)
        return result
    
    def _detect_edges_numpy(self, image: np.ndarray, gradient: str = "classic") -> np.ndarray:
        """
        Compute edge map from gradient magnitude with Otsu/eta threshold.
        """
        self.diff.sync_axes_from_tag(image)
        
        if gradient == "classic":
            grad = self.diff.gradient(image)
        elif gradient == "sobel":
            grad = self.diff.sobel_gradient(image)
        else:
            raise ValueError(f"Unknown gradient type: {gradient}")   
        
        magnitude = np.sqrt(np.sum(grad ** 2, axis=self.diff.direction_axis or 0))
        
        eta_val = threshold_otsu(magnitude) if self.eta in (None, "otsu") else self.eta
        result = magnitude > eta_val
        
        if self.dtype == "auto":
            result = result.astype(np.float32, copy=False) if self.as_float else result
        else:
            return result.astype(self.dtype_numpy, copy=False)  
                
        tagger = self.track(grad)
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=True
        )
        tracker = tagger.copy_to(result)       
        tracker.update_tags({
            "status": "edge",
            "layout_name": layout_name,
            "shape_after": result.shape,
            **axes_tags
        })
        return tracker.get()

    @torch.no_grad()
    def _detect_edges_torch(self, image: torch.Tensor, gradient: str = "classic") -> torch.Tensor:
        """
        Torch variant for gradient-based edges with Otsu/eta threshold.
        """
        self.diff.sync_axes_from_tag(image)
        
        if gradient == "classic":
            grad = self.diff.gradient(image)
        elif gradient == "sobel":
            grad = self.diff.sobel_gradient(image)
        else:
            raise ValueError(f"Unknown gradient type: {gradient}")
        magnitude = torch.sqrt(torch.sum(grad ** 2, dim=self.diff.direction_axis or 0))
        
        eta_val = threshold_otsu(magnitude.detach().cpu().numpy()) if self.eta in (None, "otsu") else self.eta
        result = magnitude > torch.tensor(eta_val, dtype=magnitude.dtype, device=magnitude.device)
        
        if self.dtype == "auto":
            result = result.float() if self.as_float else result
        else:
            return result.to(dtype=self.dtype_torch)  
            
        tagger = self.track(grad)
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=True
        )
        tracker = tagger.copy_to(result)       
        tracker.update_tags({
            "status": "edge",
            "layout_name": layout_name,
            "shape_after": result.shape,
            **axes_tags
        })
        return tracker.get()

    # -------- Axes helper --------    
    def _get_axes(self, arr: ArrayLike) -> List[int]:
        """
        Determine the spatial axes to apply differential operators on.

        Parameters
        ----------
        arr : np.ndarray | torch.Tensor
            Input image or tensor.

        Returns
        -------
        axes : list[int]
            List of axes eligible for gradient/divergence computation.
        """
        ndim = arr.ndim
        axes = list(range(ndim))

        if ndim == 2:
            return axes  # Default fallback for simple 2D

        tag = self.get_tag(arr, self.framework) if self.has_tag(arr, self.framework) else {}

        def to_positive(axis: Optional[int]) -> Optional[int]:
            return axis if axis is None or axis >= 0 else axis + ndim

        channel_ax = to_positive(tag.get("channel_axis", self.axes.get("channel_axis")))
        batch_ax = to_positive(tag.get("batch_axis", self.axes.get("batch_axis")))
        direction_ax = to_positive(tag.get("direction_axis", self.axes.get("direction_axis")))

        # Remove non-spatial axes
        for ax in (channel_ax, batch_ax, direction_ax):
            if ax is not None and ax in axes:
                axes.remove(ax)
        return axes

    # -------- Sign-change edges (zero-crossings dominance) --------
    def _sign_change_edges(self, u: ArrayLike) -> ArrayLike:
        if self.framework == "torch": 
            func = self._torch_sign_change_padded if self.use_padding else self._torch_sign_change 
        else: 
            func = self._numpy_sign_change_padded if self.use_padding else self._numpy_sign_change
        
        self.processor.function = func
        result = self.processor(u)

        return result

    def _numpy_sign_change(self, u: np.ndarray) -> np.ndarray:
        tagger = self.track(u)
        ndim = u.ndim
        shape = u.shape
        bool_map = np.zeros_like(u, dtype=bool)
        
        for axis in self._get_axes(u):

            # Get forward and backward slices
            a = np.take(u, indices=range(1, shape[axis]), axis=axis)
            b = np.take(u, indices=range(0, shape[axis] - 1), axis=axis)

            # Compute sign change and dominance
            prod = a * b <= 0
            delta = np.abs(a) - np.abs(b)

            # Prepare mask slices for both positions
            selector_a = [slice(None)] * ndim
            selector_b = [slice(None)] * ndim
            selector_a[axis] = slice(1, shape[axis])
            selector_b[axis] = slice(0, shape[axis] - 1)

            # Activate dominant pixels
            bool_map[tuple(selector_a)] |= prod & (delta >= 0)
            bool_map[tuple(selector_b)] |= prod & (delta < 0)
            
        result = bool_map.astype(np.float32, copy=False)
        tracker = tagger.copy_to(result)
        tracker.update_tags({
            "status": "sign_change",
            "shape_after": result.shape,
        })
        return tracker.get()
    
    def _numpy_sign_change_padded(self, u: np.ndarray) -> np.ndarray:
        tagger = self.track(u)
        ndim = u.ndim
        shape = u.shape
        bool_map = np.zeros_like(u, dtype=bool)
        
        for axis in self._get_axes(u):
            # Get forward and backward slices
            a = np.take(u, indices=range(1, shape[axis]), axis=axis)
            b = np.take(u, indices=range(0, shape[axis] - 1), axis=axis)
            
            # Compute sign change and dominance
            prod = a * b <= 0
            delta = np.abs(a) - np.abs(b)
            
            # Prepare mask slices for both positions
            pad_a = [0, 0] * ndim
            pad_b = [0, 0] * ndim
            pad_a[2 * (ndim - axis - 1) + 1] = 1
            pad_b[2 * (ndim - axis - 1) + 0] = 1
            padded_a = np.pad(prod & (delta >= 0), pad_a)
            padded_b = np.pad(prod & (delta < 0), pad_b)            
            
            # Activate dominant pixels
            bool_map |= padded_a
            bool_map |= padded_b            
            
        result = bool_map.astype(np.float32, copy=False)
        tracker = tagger.copy_to(result)
        tracker.update_tags({
            "status": "sign_change",
            "shape_after": result.shape,
        })
        return tracker.get()

    @torch.no_grad()
    def _torch_sign_change_padded(self, u: torch.Tensor) -> torch.Tensor:
        tagger = self.track(u)
        ndim = u.ndim
        bool_map = torch.zeros_like(u, dtype=torch.bool)
        
        for axis in self._get_axes(u):
            # Get forward and backward slices
            a = u[(slice(None),) * axis + (slice(1, None),)]
            b = u[(slice(None),) * axis + (slice(0, -1),)]
            
            # Compute sign change and dominance
            sign_change = (a * b) <= 0
            delta = torch.abs(a) - torch.abs(b)
            
            # Prepare mask slices for both positions
            pad_a = [0, 0] * ndim
            pad_b = [0, 0] * ndim
            pad_a[2 * (ndim - axis - 1) + 1] = 1
            pad_b[2 * (ndim - axis - 1) + 0] = 1
            padded_a = torch.nn.functional.pad(sign_change & (delta >= 0), pad_a)
            padded_b = torch.nn.functional.pad(sign_change & (delta < 0), pad_b)
            
            # Activate dominant pixels
            bool_map |= padded_b
            bool_map |= padded_a
            
        result = bool_map.float()
        tracker = tagger.copy_to(result)
        tracker.update_tags({
            "status": "sign_change",
            "shape_after": result.shape,
        })
        return tracker.get()
 
    @torch.no_grad()
    def _torch_sign_change(self, u: torch.Tensor) -> torch.Tensor:
        tagger = self.track(u)
        ndim = u.ndim
        shape = u.shape 
        bool_map = torch.zeros_like(u, dtype=torch.bool)
        
        for axis in self._get_axes(u):
            # Get forward and backward slices
            a = u[(slice(None),) * axis + (slice(1, None),)]
            b = u[(slice(None),) * axis + (slice(0, -1),)]
            
            # Compute sign change and dominance
            prod  = (a * b) <= 0
            delta = torch.abs(a) - torch.abs(b)
             
            # Prepare mask slices for both positions
            selector_a = [slice(None)] * ndim
            selector_b = [slice(None)] * ndim
            selector_a[axis] = slice(1, shape[axis])
            selector_b[axis] = slice(0, shape[axis] - 1)

            # Activate dominant pixels
            bool_map[tuple(selector_a)] |= prod  & (delta >= 0)
            bool_map[tuple(selector_b)] |= prod  & (delta < 0)
            
        result = bool_map.float()
        tracker = tagger.copy_to(result)
        tracker.update_tags({
            "status": "sign_change",
            "shape_after": result.shape,
        })
        return tracker.get()
 
    # -------- Laplacian zero-crossing --------
    def _laplacian_edge(self, u: ArrayLike) -> ArrayLike:
        def lap_edge_channel(channel: ArrayLike) -> ArrayLike:
            self.diff.sync_axes_from_tag(channel)
            lap = self.diff.laplacian(channel)
            return self._sign_change_edges(lap)
        
        self.processor.function = lap_edge_channel
        result = self.processor(u)
        if self.dtype == "auto":
            return result.float() if self.as_float and isinstance(result, torch.Tensor) else result.astype(np.float32, copy=False)
        else:
            if isinstance(result, torch.Tensor):
                return result.to(dtype=self.dtype_torch)
            else:
                return result.astype(self.dtype_numpy, copy=False)

    # -------- Combined edges --------
    def _combined_edge(self, u: ArrayLike) -> ArrayLike:
        def fused_edge_channel(channel: ArrayLike) -> ArrayLike:
            # --- Get gradient mask (framework-aware) ---
            lap = self._laplacian_edge(channel)
            grad = self._gradient_edges(channel)
            
            # --- Apply fusion mode ---
            if self.mode == "and":
                return (grad > 0) & (lap > 0)
            elif self.mode == "or":
                return (grad > 0) | (lap > 0)
            elif self.mode == "weighted":
                fusion = self.alpha * grad + (1 - self.alpha) * lap
                
                if self.threshold == "auto" and isinstance(fusion, torch.Tensor):
                    thresh_val = torch.tensor(threshold_otsu(fusion.detach().cpu().numpy()) , 
                                              dtype=fusion.dtype, device=fusion.device)
                else:
                    thresh_val = threshold_otsu(fusion)
                    
                return fusion > thresh_val
            else:
                raise ValueError("Unknown fusion mode.")
            
        self.processor.function = fused_edge_channel
        result = self.processor(u)
        if self.dtype == "auto":
            return result.float() if self.as_float and isinstance(result, torch.Tensor) else result.astype(np.float32, copy=False)
        else:
            if isinstance(result, torch.Tensor):
                return result.to(dtype=self.dtype_torch)
            else:
                return result.astype(self.dtype_numpy, copy=False)
            
    # -------- Marr-Hildreth --------
    def _marr_hildreth(self, u: ArrayLike) -> ArrayLike:
        def channel_marr_hildreth(channel: ArrayLike) -> ArrayLike:
            lap = self._laplacian_edge(channel)
            grad = self._gradient_edges(channel)
                
            if isinstance(lap, np.ndarray) and isinstance(grad, np.ndarray):
                return np.logical_and(lap, grad)

            if isinstance(lap, torch.Tensor) and isinstance(grad, torch.Tensor):
                return lap.bool() & grad.bool()

        self.processor.function = channel_marr_hildreth
        result = self.processor(u)
        if self.dtype == "auto":
            return result.float() if self.as_float and isinstance(result, torch.Tensor) else result.astype(np.float32, copy=False)
        else:
            if isinstance(result, torch.Tensor):
                return result.to(dtype=self.dtype_torch)
            else:
                return result.astype(self.dtype_numpy, copy=False)    
            
    # -------- Canny --------
    @torch.no_grad()
    def _canny_edges(self, u: ArrayLike) -> ArrayLike:
        """
        Canny edge detection (2D/3D). Uses Gaussian smoothing, gradient magnitude,
        NMS (complete or simple), double threshold, and hysteresis.
        """
        tagger=self.track(u)
        framework = self.framework
        eps = 1e-8

        # --- Step 1: Gaussian smoothing ---       
        u_smoothed = self.convolve(u, 
                                    self.kernel,
                                    sigma=self.sigma,
                                    size=self.size,
                                    )
        
        # --- Step 2: Gradient (magnitude and direction) ---
        self.diff.sync_axes_from_tag(u)
        grad = self.diff.gradient(u_smoothed)
        
        if framework == "torch":
            magnitude = torch.sqrt(torch.sum(grad**2, dim=self.diff.direction_axis or 0) + eps)
            orientation = torch.atan2(grad[1], grad[0]) if grad.shape[0] >= 2 else torch.zeros_like(magnitude)
        else:
            magnitude = np.sqrt(np.sum(grad**2, axis=self.diff.direction_axis or 0) + eps)
            orientation = np.arctan2(grad[1], grad[0]) if grad.shape[0] >= 2 else np.zeros_like(magnitude)

        # --- Step 3: Non-maximum suppression (simple version) ---

        if self.complete_nms:
            magnitude = self._non_maximum_suppression_nd(magnitude, orientation, original=u)
        else:
            # Simple non-maximum suppression (2D only)    
            if framework == "torch":
                nms = torch.zeros_like(magnitude, dtype=torch.bool)
                for axis in range(magnitude.ndim):
                    shifted_p = torch.roll(magnitude, shifts=-1, dims=axis)
                    shifted_n = torch.roll(magnitude, shifts=1, dims=axis)
                    mask = (magnitude >= shifted_p) & (magnitude >= shifted_n)
                    nms |= mask
                nms = nms.float()
            else:
                nms = np.zeros_like(magnitude, dtype=bool)
                for axis in range(magnitude.ndim):
                    shifted_p = np.roll(magnitude, -1, axis=axis)
                    shifted_n = np.roll(magnitude, 1, axis=axis)
                    mask = (magnitude >= shifted_p) & (magnitude >= shifted_n)
                    nms |= mask
                nms = nms.astype(np.float32, copy=False)

            magnitude = magnitude * nms  # keep only local maxima

        # --- Step 4: Double thresholding ---
        if self.eta in (None, "otsu"):
            if framework == "torch":
                thresh_high = torch.tensor(threshold_otsu(magnitude.detach().cpu().numpy()), device=magnitude.device)
            else:
                thresh_high = threshold_otsu(magnitude)
        else:
            thresh_high = self.eta

        thresh_low = 0.5 * thresh_high

        if framework == "torch":
            strong_edges = (magnitude >= thresh_high).float()
            weak_edges = ((magnitude >= thresh_low) & (magnitude < thresh_high)).float()
        else:
            strong_edges = (magnitude >= thresh_high).astype(np.float32, copy=False)
            weak_edges = ((magnitude >= thresh_low) & (magnitude < thresh_high)).astype(np.float32, copy=False)

        # --- Step 5: Hysteresis (connect weak edges to strong ones) ---
        if framework == "torch":
            from torch.nn.functional import max_pool2d, max_pool3d
            if magnitude.ndim == 2:
                pooled = max_pool2d(strong_edges.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze()
            else:
                pooled = max_pool3d(strong_edges.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze()
            final_edges = ((pooled > 0) & (weak_edges > 0)) | (strong_edges > 0)
            final_edges = final_edges.float()
        else:
            from scipy.ndimage import maximum_filter
            pooled = maximum_filter(strong_edges, size=3, mode='reflect')
            final_edges = ((pooled > 0) & (weak_edges > 0)) | (strong_edges > 0)
            final_edges = final_edges.astype(np.float32, copy=False)
            
        result = final_edges
        tracker = tagger.copy_to(result)
        tracker.update_tags({
            "status": "canny",
            "shape_after": result.shape,
        })
        return tracker.get()

    # -------- Orientation-aware NMS (2D/3D) --------
    @torch.no_grad()
    def _non_maximum_suppression_nd(
        self,
        magnitude: ArrayLike,
        orientation: ArrayLike,
        original: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """
        Perform non-maximum suppression using gradient orientation.
        Supports 2D/3D images, with channel axis handling.

        """
        framework = "torch" if torch.is_tensor(magnitude) else "numpy"

        # === Detect channel_axis if present ===
        tag = self.get_tag(original, framework) if self.has_tag(original, framework) else {}
        channel_axis = tag.get("channel_axis", self.axes.get("channel_axis"))

        # Helper to apply NMS slice per channel
        def process_slice(mag_slice: ArrayLike, ori_slice: ArrayLike) -> ArrayLike:
            """
            Fine non-maximum suppression (NMS) for 2D and 3D images.
            Supports NumPy and PyTorch tensors.
            """
            # === Auto-detect framework ===
            is_torch = torch.is_tensor(mag_slice)
            framework = "torch" if is_torch else "numpy"
            
            if mag_slice.ndim == 2:
                if framework == "torch":
                    ori = torch.rad2deg(ori_slice) % 180
                else:
                    ori = np.rad2deg(ori_slice) % 180

                shifted_pos = mag_slice.clone() if is_torch else np.zeros_like(mag_slice)
                shifted_neg = mag_slice.clone() if is_torch else np.zeros_like(mag_slice)

                # Direction masks
                horiz = (ori <= 22.5) | (ori > 157.5)
                diag1 = (ori > 22.5) & (ori <= 67.5)
                vert  = (ori > 67.5) & (ori <= 112.5)
                diag2 = (ori > 112.5) & (ori <= 157.5)

                roll_fn = torch.roll if is_torch else np.roll

                # Roll according to direction
                shifted_pos[horiz] = roll_fn(mag_slice, shifts=(-1, 0), dims=(0, 1))[horiz] if is_torch else \
                                                        roll_fn(mag_slice, shift=(-1, 0), axis=(0, 1))[horiz]
                
                shifted_neg[horiz] = roll_fn(mag_slice, shifts=(1, 0), dims=(0, 1))[horiz] if is_torch else \
                                                        roll_fn(mag_slice, shift=(1, 0), axis=(0, 1))[horiz]

                shifted_pos[diag1] = roll_fn(mag_slice, shifts=(-1, -1), dims=(0, 1))[diag1] if is_torch else \
                                                        roll_fn(mag_slice, shift=(-1, -1), axis=(0, 1))[diag1]
                                                        
                shifted_neg[diag1] = roll_fn(mag_slice, shifts=(1, 1), dims=(0, 1))[diag1] if is_torch else \
                                                        roll_fn(mag_slice, shift=(1, 1), axis=(0, 1))[diag1]

                shifted_pos[vert] = roll_fn(mag_slice, shifts=(0, -1), dims=(0, 1))[vert] if is_torch else \
                                                        roll_fn(mag_slice, shift=(0, -1), axis=(0, 1))[vert]

                shifted_neg[vert] = roll_fn(mag_slice, shifts=(0, 1), dims=(0, 1))[vert] if is_torch else \
                                                        roll_fn(mag_slice, shift=(0, 1), axis=(0, 1))[vert]

                shifted_pos[diag2] = roll_fn(mag_slice, shifts=(1, -1), dims=(0, 1))[diag2] if is_torch else \
                                                        roll_fn(mag_slice, shift=(1, -1), axis=(0, 1))[diag2]
                                                        
                shifted_neg[diag2] = roll_fn(mag_slice, shifts=(-1, 1), dims=(0, 1))[diag2] if is_torch else \
                                                        roll_fn(mag_slice, shift=(-1, 1), axis=(0, 1))[diag2]

                keep = (mag_slice >= shifted_pos) & (mag_slice >= shifted_neg)
                return mag_slice * (keep.float() if is_torch else keep.astype(mag_slice.dtype, copy=False))

            elif mag_slice.ndim == 3:
                
                return self.process_slice_3d_patchwise(mag_slice, ori_slice, patch_size=4)

                # if is_torch:
                #     # Mean direction
                #     gx = torch.mean(torch.cos(ori_slice)).item()
                #     gy = torch.mean(torch.sin(ori_slice)).item()
                #     gz = torch.mean(torch.sin(ori_slice)).item()

                #     # Global shift
                #     shift = [int(torch.sign(torch.tensor(v)).item()) for v in [gx, gy, gz]]
                # else:
                #     gx = np.mean(np.cos(ori_slice))
                #     gy = np.mean(np.sin(ori_slice))
                #     gz = np.mean(np.sin(ori_slice))
                #     shift = [int(np.sign(v)) for v in [gx, gy, gz]]

                # # Clamp (-1 to 1)
                # shift = [max(-1, min(1, s)) for s in shift]

                # # Apply roll
                # shifted_pos = roll_fn(mag_slice, shifts=tuple(-s for s in shift), dims=(0, 1, 2)) if is_torch \
                #             else roll_fn(mag_slice, shift=tuple(-s for s in shift), axis=(0, 1, 2))

                # shifted_neg = roll_fn(mag_slice, shifts=tuple(s for s in shift), dims=(0, 1, 2)) if is_torch \
                #             else roll_fn(mag_slice, shift=tuple(s for s in shift), axis=(0, 1, 2))

                # keep = (mag_slice >= shifted_pos) & (mag_slice >= shifted_neg)
                # return mag_slice * (keep.float() if is_torch else keep.astype(mag_slice.dtype, copy=False))

            else:
                raise NotImplementedError("Only 2D and 3D images supported.")

        # === Apply NMS per channel if needed ===
        if channel_axis is not None:
            if framework == "torch":
                slices = torch.unbind(magnitude, dim=channel_axis)
                ori_slices = torch.unbind(orientation, dim=channel_axis)
                nms_slices = [process_slice(m, o) for m, o in zip(slices, ori_slices)]
                return torch.stack(nms_slices, dim=channel_axis)
            else:
                slices = np.moveaxis(magnitude, channel_axis, 0)
                ori_slices = np.moveaxis(orientation, channel_axis, 0)
                nms_slices = [process_slice(m, o) for m, o in zip(slices, ori_slices)]
                return np.moveaxis(np.stack(nms_slices, axis=0), 0, channel_axis)

        else:
            return process_slice(magnitude, orientation)

    @staticmethod
    def process_slice_3d_patchwise(mag_slice: ArrayLike, ori_slice: ArrayLike, patch_size: int = 16) -> ArrayLike:
        """
        Apply non-maximum suppression (NMS) on a 3D volume using a patch-wise approximation.

        For each 3D block (patch), a dominant gradient direction is estimated from the local orientation map,
        and NMS is applied using forward and backward shifts along this dominant direction.
        This method provides a compromise between full voxel-wise precision and computational efficiency.

        Parameters
        ----------
        mag_slice : torch.Tensor or np.ndarray
            3D magnitude volume of shape (D, H, W).
        ori_slice : torch.Tensor or np.ndarray
            3D orientation map of the same shape as mag_slice, typically in radians.
        patch_size : int, optional
            Size of the 3D blocks used to estimate local orientation. Default is 16.

        Returns
        -------
        torch.Tensor or np.ndarray
            Suppressed magnitude volume, same shape and type as mag_slice.
        """
        is_torch = torch.is_tensor(mag_slice)
        roll_fn = torch.roll if is_torch else np.roll
        D, H, W = mag_slice.shape
        output = torch.zeros_like(mag_slice) if is_torch else np.zeros_like(mag_slice)

        for d in range(0, D, patch_size):
            for h in range(0, H, patch_size):
                for w in range(0, W, patch_size):
                    # Délimiter patch
                    d1, h1, w1 = min(d + patch_size, D), min(h + patch_size, H), min(w + patch_size, W)
                    patch_mag = mag_slice[d:d1, h:h1, w:w1]
                    patch_ori = ori_slice[d:d1, h:h1, w:w1]

                    # Estimer la direction dominante du patch
                    if is_torch:
                        gx = torch.mean(torch.cos(patch_ori)).item()
                        gy = torch.mean(torch.sin(patch_ori)).item()
                        gz = torch.mean(torch.sin(patch_ori)).item()
                    else:
                        gx = np.mean(np.cos(patch_ori))
                        gy = np.mean(np.sin(patch_ori))
                        gz = np.mean(np.sin(patch_ori))

                    shift = [int(np.sign(x)) for x in (gx, gy, gz)]
                    shift = [max(-1, min(1, s)) for s in shift]

                    # Shifté localement
                    shift_pos = tuple(-s for s in shift)
                    shift_neg = tuple(s for s in shift)

                    if is_torch:
                        shifted_pos = roll_fn(patch_mag, shifts=shift_pos, dims=(0, 1, 2))
                        shifted_neg = roll_fn(patch_mag, shifts=shift_neg, dims=(0, 1, 2))
                    else:
                        shifted_pos = roll_fn(patch_mag, shift=shift_pos, axis=(0, 1, 2))
                        shifted_neg = roll_fn(patch_mag, shift=shift_neg, axis=(0, 1, 2))

                    keep = (patch_mag >= shifted_pos) & (patch_mag >= shifted_neg)
                    output[d:d1, h:h1, w:w1] = patch_mag * (keep.float() if is_torch else keep.astype(mag_slice.dtype, copy=False))

        return output

# ======================================================================
#                      Convenience wrapper
# ======================================================================

def edge_detect(
    img: ArrayLike,
    framework: Framework = "numpy",
    output_format: Framework = "numpy",
    layout_name: str = "HWC",
    layout_framework: Framework = "numpy",
    diff_strategy: Optional[str] = None,
    processor_strategy: Optional[str] = None,
    conv_strategy: Optional[str] = None,
    edge_strategy: str = "gradient",
) -> ArrayLike:

    """
    Convenience wrapper to instantiate and run EdgeDetector with minimal params.
    """    

    # ====[ Fallback ]====
    edge_strategy=edge_strategy or "gradient"
    diff_strategy=diff_strategy or "vectorized" if framework == "numpy" else "torch"
    conv_strategy=conv_strategy or "fft" if framework == "numpy" else "torch"
    processor_strategy=processor_strategy or "vectorized" if framework == "numpy" else "torch"   
        
    # ====[ Configuration ]====
    diff_params: Dict[str, Any] = {"spacing": None, "diff_strategy":diff_strategy}
    conv_params: Dict[str, Any] = {"conv_strategy": conv_strategy, "sigma": 1.0}
    edge_params: Dict[str, Any] = {"edge_strategy": edge_strategy, "eta": "otsu", "mode": "and", "complete_nms": True}
    proc_params: Dict[str, Any] = {"processor_strategy": processor_strategy,}
    layout_params: Dict[str, Any] = {"layout_name": layout_name, "layout_framework": layout_framework}
    global_params: Dict[str, Any] = {"framework": framework, "output_format": output_format}

    edge = EdgeDetector(
                        edge_detector_cfg = EdgeDetectorConfig(**edge_params),
                        diff_operator_cfg = DiffOperatorConfig(**diff_params),
                        ndconvolver_cfg = NDConvolverConfig(**conv_params),
                        img_process_cfg = ImageProcessorConfig(**proc_params),
                        layout_cfg = LayoutConfig(**layout_params),
                        global_cfg = GlobalConfig(**global_params),
                        )
    
    img_copy = edge.safe_copy(img)

    return edge(img_copy)
