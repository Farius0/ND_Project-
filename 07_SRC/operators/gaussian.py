# ==================================================
# ==============  MODULE: gaussian  ================
# ==================================================
from __future__ import annotations

from scipy.ndimage import convolve as nd_convolve, gaussian_filter, median_filter, uniform_filter
from torchvision.transforms.functional import gaussian_blur as torch_gaussian_blur
from typing import Optional, Union, Tuple, List, Dict, Any, Callable, Literal
import numpy as np, math, matplotlib.pyplot as plt, sys, torch
from scipy.signal import fftconvolve, convolve2d
import torch.nn.functional as F

from operators.image_processor import ImageProcessor
from core.operator_core import OperatorCore
from core.config import LayoutConfig, GlobalConfig, NDConvolverConfig, ImageProcessorConfig

# Public API
__all__ = ["NDConvolver", "GaussianKernelGenerator", "conv"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Framework = Literal["numpy", "torch"]

# ==================================================
# ================ NDConvolver =====================
# ==================================================

class NDConvolver(OperatorCore):
    """
    N-dimensional convolution module supporting multiple strategies.

    Provides flexible convolution across ND arrays with optional batching and
    channel-awareness. Automatically dispatches to backend-specific routines 
    (NumPy or Torch) and preserves axis tags via OperatorCore.

    Supported convolution modes include:
    - 'fft' (scipy FFT convolution)
    - 'ndimage' (scipy.ndimage.convolve)
    - 'uniform', 'gaussian', 'median' (SciPy filters)
    - 'torch' (native PyTorch convolution)
    - 'convolve2d' (SciPy 2D only)
    - 'gaussian_torch' (custom torch-based Gaussian)

    Notes
    -----
    - Compatible with images of arbitrary shape: 2D, 3D, ND.
    - Layout and axis semantics are managed via `LayoutConfig`.
    - Uses `ImageProcessor` internally to support channel-wise processing and batch handling.
    """

    def __init__(
        self,
        ndconvolver_cfg: NDConvolverConfig = NDConvolverConfig(),
        img_process_cfg: ImageProcessorConfig = ImageProcessorConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize NDConvolver with configurable convolution behavior.

        Parameters
        ----------
        ndconvolver_cfg : NDConvolverConfig
            Strategy and kernel parameters (e.g., conv_strategy='fft', sigma, padding mode, etc.).
        img_process_cfg : ImageProcessorConfig
            ImageProcessor backend to control parallelization, channel-slicing, etc.
        layout_cfg : LayoutConfig
            Layout handling for semantic axis detection (e.g., channel_axis).
        global_cfg : GlobalConfig
            Framework-level options (backend, dtype, device, normalization).
        """
        # ====[ Configuration ]====
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg
        self.conv_cfg: NDConvolverConfig = ndconvolver_cfg
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

        # ====[ Store processor-specific parameters ]====
        self.processor_strategy: str = self.img_process_cfg.processor_strategy
        self.strategy: str = self.conv_cfg.conv_strategy
        self.padding: Optional[str] = self.conv_cfg.padding
        self.grouped: Optional[bool] = self.conv_cfg.grouped
        self.fallback: Optional[str] = self.conv_cfg.conv_fallback
        
        if self.strategy not in ["fft", "ndimage", "uniform", "gaussian", "gaussian_torch", "median", "torch", "convolve2d"]:
            raise ValueError(f"[NDConvolver] Unsupported strategy '{self.strategy}'.")
        
        # ====[ Mirror inherited params locally for easy access ]====
        self.framework: Framework = self.global_cfg.framework.lower()
        self.output_format: Framework = self.global_cfg.output_format.lower()
        self.add_batch_dim: Optional[bool] = self.global_cfg.add_batch_dim
        self.add_channel_dim: Optional[bool] = self.global_cfg.add_channel_dim
        self.normalize: bool = bool(self.global_cfg.normalize)
        self.verbose: bool = bool(self.global_cfg.verbose)
        self.backend: str = self.global_cfg.backend.lower()        
        self.device: str = (
            "cuda"
            if (torch.cuda.is_available() and self.framework == "torch")
            else self.global_cfg.device
        )

        # Build the internal processor (function will be injected later)
        self.processor: ImageProcessor = ImageProcessor(
            img_process_cfg = self.img_process_cfg,
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg,            
        )
        # ====[ Initialize OperatorCore with all axes ]====
        super().__init__(
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg,
        )        

    # --------------- Public API ---------------

    def __call__(
        self,
        image: ArrayLike,
        kernel: ArrayLike,
        size: Optional[int] = None,
        sigma: float = 1.0,
        mode: str = "reflect",
        enable_uid: bool = False,
        track: bool = True,
        trace_limit: int = 10,
        op_params: Optional[dict] = None,
    ) -> ArrayLike:
        """
        Apply N-dimensional convolution using the configured strategy.

        Supports multiple convolution modes (FFT, Gaussian, median, torch, etc.)
        and applies them to the given input using `ImageProcessor` for backend-aware
        and layout-consistent processing.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image to convolve.
        kernel : np.ndarray or torch.Tensor
            Convolution kernel. Should match the framework of the image.
        size : int, optional
            Window size for strategies like 'uniform' or 'median' (used when no kernel is given).
        sigma : float, default 1.0
            Standard deviation for Gaussian filtering.
        mode : str, default 'reflect'
            Border mode for strategies like 'ndimage' (e.g., 'reflect', 'nearest').
        enable_uid : bool, default False
            Whether to assign a UID for traceability in the tag.
        track : bool, default True
            If True, maintain operation history and metadata tracking.
        trace_limit : int, default 10
            Maximum number of operations to store in the trace tag.
        op_params : dict, optional
            Additional metadata to embed in the tag (e.g., {'sigma': 1.0, 'mode': 'fft'}).

        Returns
        -------
        np.ndarray or torch.Tensor
            Convolved output in the same backend as the input, with tags preserved.

        Notes
        -----
        - Input and kernel are automatically converted to the target framework.
        - Layout and channel handling are managed internally via `OperatorCore`.
        - Output is returned via `to_output()` with tagging and optional UID.
        """
        image_tracked = self.convert_once(
            image,
            tag_as="input",
            direction_axis=None,
            framework=self.framework,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
        )

        kernel_tracked = self.convert_once(
            kernel,
            framework=self.framework,
            tag_as="kernel",
            direction_axis=None,
            enable_uid=enable_uid,
            track=track,
            trace_limit=trace_limit,
        )
        
        result = self._convolve(image_tracked, 
                                kernel_tracked,
                                sigma=sigma,
                                size=size,
                                mode=mode,
                                )

        return self.to_output(
            result,
            tag_as="output",
            enable_uid=enable_uid,
            op_params=op_params
        )
        
    # ====[ Private methods ]====    
    def _get_axes(self, arr: ArrayLike) -> List[int]:
        """
        Determine the spatial axes to apply differential or convolution operators.

        Excludes non-spatial axes such as channel, batch, or direction axes,
        based on tags or internal axis configuration. Handles negative indices
    and missing tags gracefully.

        Parameters
        ----------
        arr : np.ndarray or torch.Tensor
            Input ND array or tensor to analyze.

        Returns
        -------
        List[int]
            List of spatial axis indices eligible for processing.

        Notes
        -----
        - For 2D images, returns [0, 1] by default.
        - Axis roles are retrieved from image tags if available; otherwise falls back
        to `self.axes` (from layout config).
        - Negative axes are converted to positive values based on `arr.ndim`.
        - Verbose mode prints selected spatial axes.
        """
        ndim = arr.ndim
        axes = list(range(ndim))

        if ndim == 2:
            return axes  # Default fallback for simple 2D

        tag = self.get_tag(arr, self.framework) if self.has_tag(arr, self.framework) else {}

        def to_positive(axis: Optional[int]) -> Optional[int]:
            return axis if axis is None or axis >= 0 else axis + ndim

        channel_axis = to_positive(tag.get("channel_axis", self.axes.get("channel_axis")))
        batch_axis = to_positive(tag.get("batch_axis", self.axes.get("batch_axis")))
        direction_axis = to_positive(tag.get("direction_axis", self.axes.get("direction_axis")))

        # Remove non-spatial axes
        for ax in (channel_axis, batch_axis, direction_axis):
            if ax is not None and ax in axes:
                axes.remove(ax)

        if self.verbose:
            print(f"[NDConvolver] Spatial axes selected: {axes}")

        return axes

    # ====[ Convolution strategy dispatcher ]====
    def _convolve(
        self,
        image: ArrayLike,
        kernel: ArrayLike,
        sigma: float = 1.0,
        size: Optional[int] = None,
        mode: str = "reflect",
    ) -> ArrayLike:
        """
        Dispatch and apply the configured convolution strategy on an ND image.

        Supports a wide range of strategies including FFT, Gaussian, median, uniform,
        PyTorch-based, and hybrid approaches with automatic fallback.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image to convolve.
        kernel : np.ndarray or torch.Tensor
            Convolution kernel. May be ignored in strategies like 'uniform' or 'median'.
        sigma : float, default 1.0
            Standard deviation for Gaussian-based filters.
        size : int, optional
            Filter window size (used in 'uniform', 'median', or Gaussian).
            If None, defaults to 2*floor(3*sigma)+1.
        mode : str, default 'reflect'
            Border handling mode for scipy filters (e.g., 'reflect', 'nearest').

        Returns
        -------
        np.ndarray or torch.Tensor
            Convolved image, with same backend as the input and layout preserved.

        Raises
        ------
        ValueError
            If the selected strategy is not supported or kernel is incompatible.
        RuntimeError
            If the convolution process fails due to execution errors.

        Notes
        -----
        - Strategy is defined in `self.strategy` and must be one of:
        'fft', 'ndimage', 'uniform', 'gaussian', 'median', 'torch',
        'convolve2d', 'gaussian_torch'.
        - For large kernels and 'torch' strategy, the method auto-switches to a fallback
        (e.g., 'fft' or 'gaussian') for performance and memory safety.
        - Spatial axes are resolved via `_get_axes()` using tag or layout configuration.
        - Internally delegates to `self.processor` for ND-slicing and framework handling.
        """
        if image.ndim < kernel.ndim:
            raise ValueError("[NDConvolver] Kernel has more dimensions than image.")
        if size is None:
            size = int(2 * math.floor(3*sigma) + 1)

        # Define strategy-specific function
        if self.strategy == "fft":
            def conv_fn(x: ArrayLike) -> ArrayLike:
                return fftconvolve(x, kernel, mode=self.padding)
        elif self.strategy == "ndimage":
            def conv_fn(x: ArrayLike) -> ArrayLike:
                axes = tuple(self._get_axes(x))
                return nd_convolve(x, kernel, mode=mode, axes=axes)
        elif self.strategy == "uniform":
            def conv_fn(x: ArrayLike) -> ArrayLike: 
                axes = tuple(self._get_axes(x))
                return uniform_filter(x, size=size, mode=mode, axes=axes)
        elif self.strategy == "gaussian":
            def conv_fn(x: ArrayLike) -> ArrayLike: 
                axes = tuple(self._get_axes(x))
                return gaussian_filter(x, sigma=sigma, mode=mode, axes=axes)
        elif self.strategy == "convolve2d":
            def conv_fn(x: ArrayLike) -> ArrayLike:
                return convolve2d(x, kernel, mode="same", boundary="symm")
        elif self.strategy == "median":
            def conv_fn(x: ArrayLike) -> ArrayLike:
                axes = tuple(self._get_axes(x))
                return median_filter(x, size=size, mode=mode, axes=axes)
        elif self.strategy == "gaussian_torch":
            def conv_fn(x: ArrayLike) -> ArrayLike: 
                channel_axis = self.get_axis(x, "channel_axis")
                kernel_size = [size] * (x.ndim - 1) \
                    if channel_axis is not None else [size] * (x.ndim)
                return torch_gaussian_blur(x, kernel_size=kernel_size, sigma=sigma).to(self.device)
        elif self.strategy == "torch":
            # Fallback to torch-based convolution
            kernel_shape = kernel.shape if isinstance(kernel, (torch.Tensor, np.ndarray)) else ()
            max_kernel_dim = max(kernel_shape[-2:]) if len(kernel_shape) >= 2 else 0
            use_fallback = max_kernel_dim > 15 # Example threshold for fallback

            if use_fallback:
                if self.verbose:
                    print(f"[NDConvolver] Kernel too large ({max_kernel_dim}×{max_kernel_dim}) → switching to {self.fallback} fallback.")
                
                self.processor.framework = "numpy"
                self.processor.strategy = "vectorized" if self.fallback not in ["fft", "convolve2d"] else self.fallback 
                    
                def conv_fn(x: ArrayLike) -> ArrayLike:
                    x_np = self.convert_once(image,
                                            tag_as="input",
                                            framework="numpy",
                                            ) if isinstance(x, torch.Tensor) else x
                    
                    k_np = self.convert_once(kernel,
                                            tag_as="kernel",
                                            framework="numpy",
                                            ) if isinstance(kernel, torch.Tensor) else kernel
                    
                    if self.fallback not in ["fft", "convolve2d"]:
                        axes = tuple(self._get_axes(x_np))
                        
                    return fftconvolve(x_np, k_np, mode=self.padding) if self.fallback == "fft" \
                        else gaussian_filter(x_np, sigma=sigma, mode=mode, axes=axes)
            else:
                    conv_fn = self._torch_convolve(kernel)            

        else:
            raise ValueError(f"[NDConvolver] Unsupported strategy '{self.strategy}'.")

        self.processor.function = conv_fn
        
        try:
            result = self.processor(image)
            # update processor
            self.processor.strategy = "torch"
            self.processor.framework = "torch"
            return result
        except Exception as e:
            raise RuntimeError(f"[NDConvolver] Convolution failed using strategy '{self.strategy}'") from e


    # ====[ Torch-based convolution wrapper ]====
    def _torch_convolve(self, kernel: ArrayLike) -> Callable[[ArrayLike], ArrayLike]:
        """
        Build a Torch-native convolution closure for ND inputs (2D or 3D), with or without channels.

        Supports grouped and ungrouped convolutions, and handles different axis layouts
        transparently (based on tagged channel_axis). Internally reshapes tensors to match
        PyTorch's expected (N,C,...) format, applies convolution, then restores original layout.

        Parameters
        ----------
        kernel : np.ndarray or torch.Tensor
            Convolution kernel to use. Shape must match the spatial dimensions (2D or 3D).

        Returns
        -------
        convolve_fn : Callable[[ArrayLike], ArrayLike]
            A callable function that accepts a tensor slice and returns the convolved output.

        Supported Cases
        ---------------
        - 2D image          : (H, W)               → conv2d with [1, 1, Kh, Kw]
        - 3D image no ch.   : (D, H, W)            → conv3d with [1, 1, Kd, Kh, Kw]
        - 3D image w/ ch.   : (C, H, W)            → conv2d with groups=C if `grouped=True`
        - 4D image w/ ch.   : (C, D, H, W)         → conv3d with groups=C if `grouped=True`

        Notes
        -----
        - Input is automatically cast to float32 and moved to the configured device.
        - Kernel is repeated or expanded depending on grouping strategy.
        - Padding is automatically computed to preserve input size (`same` padding).
        - Raises ValueError for unsupported input dimensions.
        """
        @torch.no_grad()
        def convolve_slice(x: ArrayLike) -> ArrayLike:
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device) if not isinstance(x, torch.Tensor) else x
            k = torch.as_tensor(kernel, dtype=torch.float32, device=self.device) if not isinstance(kernel, torch.Tensor) else kernel

            channel_axis = self.get_axis(x, "channel_axis")

            # Case 2D image (H, W)
            if x.ndim == 2:
                x = x.unsqueeze(0).unsqueeze(0)       # [1, 1, H, W]
                k = k.unsqueeze(0).unsqueeze(0)       # [1, 1, Kh, Kw]
                y = F.conv2d(x, k, padding=self._auto_padding(k.shape[-2:]))
                return y.squeeze(0).squeeze(0)

            # Case 3D image without channels
            if x.ndim == 3 and channel_axis is None:
                x = x.unsqueeze(0).unsqueeze(0)       # [1, 1, D, H, W]
                k = k.unsqueeze(0).unsqueeze(0)
                y = F.conv3d(x, k, padding=self._auto_padding(k.shape[-3:]))
                return y.squeeze(0).squeeze(0)

            # Case 3D with channels → conv2d with groups
            if x.ndim == 3:
                C = x.shape[0]
                x = x.unsqueeze(0)                    # [1, C, H, W]
                if self.grouped:
                    # Independent convolution per channel
                    k = k.unsqueeze(0).repeat(C, 1, 1, 1)  # [C, 1, Kh, Kw]
                    y = F.conv2d(x, k, padding=self._auto_padding(k.shape[-2:]), groups=C)
                    return y.squeeze(0)
                else:
                    # Shared kernel across all channels
                    k = k.unsqueeze(0).expand(1, C, *k.shape[-2:])  # [1, C, Kh, Kw]
                    y = F.conv2d(x, k, padding=self._auto_padding(k.shape[-2:]), groups=1)
                    return y.squeeze(0)

            # Case 4D with channels → conv3d with groups
            if x.ndim == 4:
                C = x.shape[0]
                x = x.unsqueeze(0)                    # [1, C, D, H, W]
                if self.grouped:
                    k = k.unsqueeze(0).repeat(C, 1, 1, 1, 1)  # [C, 1, Kd, Kh, Kw]
                    y = F.conv3d(x, k, padding=self._auto_padding(k.shape[-3:]), groups=C)
                    return y.squeeze(0)
                else:
                    k = k.unsqueeze(0).expand(1, C, *k.shape[-3:])  # [1, C, Kd, Kh, Kw]
                    y = F.conv3d(x, k, padding=self._auto_padding(k.shape[-3:]), groups=1)
                    return y.squeeze(0)

            raise ValueError(f"[NDConvolver] Unsupported input shape: {x.shape}")

        return convolve_slice

    # ====[ Auto-padding calculation ]====
    def _auto_padding(self, kernel_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute padding values for convolution based on kernel shape and padding mode.

        Supports scalar, tuple, and named padding modes ('same', 'valid', 'full') for N-dimensional kernels.

        Parameters
        ----------
        kernel_shape : tuple of int
            Shape of the convolution kernel along spatial dimensions (e.g., (3, 3) for 2D, (5, 5, 5) for 3D).

        Returns
        -------
        Tuple[int, ...]
            Per-dimension padding values to apply on each spatial axis.

        Raises
        ------
        ValueError
            If the padding mode is invalid or inconsistent with kernel shape.

        Supported Modes
        ---------------
        - int            : Applies same integer padding to all dimensions.
        - tuple/list     : Custom per-axis padding (must match kernel dimensions).
        - 'same'         : Auto-padding to preserve input shape (padding = kernel_size // 2).
        - 'valid'        : No padding (all padding = 0).
        - 'full'         : Maximal padding (padding = kernel_size - 1).
        """
        if isinstance(self.padding, int):
            return tuple([self.padding] * len(kernel_shape))

        if isinstance(self.padding, (tuple, list)):
            if len(self.padding) != len(kernel_shape):
                raise ValueError(f"[NDConvolver] Padding {self.padding} does not match kernel shape {kernel_shape}.")
            return tuple(self.padding)

        if self.padding == "same":
            return tuple(k // 2 for k in kernel_shape)

        if self.padding == "full":
            return tuple(k - 1 for k in kernel_shape)

        if self.padding == "valid":
            return tuple(0 for _ in kernel_shape)

        raise ValueError(f"[NDConvolver] Unsupported padding mode: {self.padding}")

    # ====[ Summary info ]====
    def summary(self) -> None:
        """
        Print a detailed summary of NDConvolver configuration.
        """
        print("====[ NDConvolver Summary ]====")
        print(f"Strategy           : {self.strategy}")
        print(f"Processor Strategy : {self.processor_strategy}")
        print(f"Framework          : {self.framework}")
        print(f"Output Format      : {self.output_format}")
        print(f"Device             : {self.device}")
        print(f"Padding            : {self.padding}")
        print(f"Backend            : {self.backend}")
        print(f"Normalize          : {self.normalize}")
        print(f"Layout Name        : {self.layout_name}")
        print(f"Processor Ready    : {'Yes' if hasattr(self, 'processor') else 'No'}")
        print("Axes:")
        for ax, val in self.axes.items():
            print(f"  {ax:<15}: {val}")

# ==================================================
# ========== GaussianKernelGenerator ===============
# ==================================================
class GaussianKernelGenerator(OperatorCore):
    """
    Generate N-dimensional (or 2D rotated) Gaussian kernels for convolution.

    Compatible with both NumPy and PyTorch backends, this utility constructs
    isotropic or anisotropic Gaussian kernels, with optional rotation (2D only),
    symmetry enforcement, and dtype/device handling via `GlobalConfig`.

    Notes
    -----
    - Outputs are backend-aware and layout-consistent.
    - If `global_cfg.normalize=True`, the kernel is L1-normalized.
    - For 2D kernels, a rotation angle (in degrees) can be specified.
    - If `ndconvolver_cfg.symmetry=True`, the kernel is averaged across symmetric axes.
    """

    def __init__(
        self,
        ndconvolver_cfg: NDConvolverConfig = NDConvolverConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize the GaussianKernelGenerator with configurable convolution settings.

        Parameters
        ----------
        ndconvolver_cfg : NDConvolverConfig
            Controls the convolution dimensions, kernel size, sigma, symmetry, and backend mode.
        layout_cfg : LayoutConfig
            Provides layout information for interpreting spatial axes.
        global_cfg : GlobalConfig
            Controls backend preferences, normalization, and device placement.
        """
        # ====[ Configuration ]====
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg
        self.conv_cfg: NDConvolverConfig = ndconvolver_cfg
        
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

        # ====[ Store processor-specific parameters ]====
        self.dtype_torch: Optional[type] = self.conv_cfg.dtype
        self.dtype_numpy = {
            torch.float32: np.float32,
            torch.float64: np.float64
        }.get(self.dtype_torch, np.float32)
        
        # ====[ Mirror inherited params locally for easy access ]====
        self.framework: Framework = self.global_cfg.framework.lower()
        self.output_format: Framework = self.global_cfg.output_format.lower()
        self.add_batch_dim: Optional[bool] = self.global_cfg.add_batch_dim
        self.add_channel_dim: Optional[bool] = self.global_cfg.add_channel_dim
        self.normalize: bool = bool(self.global_cfg.normalize)
        self.verbose: bool = bool(self.global_cfg.verbose)
        self.backend: str = self.global_cfg.backend.lower()        
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

    def generate(
        self,
        dim: int = 2,
        size: Optional[Union[int, List[int]]] = None,
        sigma: Union[float, List[float]] = 1.0,
        truncate: float = 3.0,
        symmetry: bool = True,
        angle: float = 0.0,
        visualize: bool = False,
        return_grid: bool = False,
        return_numpy: bool = False,
    ) -> ArrayLike:
        """
        Generate a Gaussian kernel in N dimensions (or 2D with optional rotation).

        The kernel can be isotropic or anisotropic, and supports optional visualization,
        symmetry enforcement, and coordinate grid return for 2D cases.

        Parameters
        ----------
        dim : int, default 2
            Dimension of the kernel (e.g., 2 for 2D, 3 for 3D).
        size : int or list of int, optional
            Kernel size per dimension. Must be odd. If None, computed from `sigma` and `truncate`.
        sigma : float or list of float, default 1.0
            Standard deviation(s) of the Gaussian function.
        truncate : float, default 3.0
            Radius in standard deviations beyond which the kernel is truncated.
        symmetry : bool, default True
            Enforce symmetry by averaging the kernel with its mirror.
        angle : float, default 0.0
            Rotation angle in degrees (only used when `dim == 2`).
        visualize : bool, default False
            If True, display a 2D plot of the kernel (only if `dim == 2`).
        return_grid : bool, default False
            If True, return the (X, Y) grid used to build the 2D rotated kernel.
        return_numpy : bool, default False
            If True, return the kernel as a NumPy array (else Torch tensor).

        Returns
        -------
        kernel : torch.Tensor or np.ndarray
            The generated Gaussian kernel.
        grid : tuple of arrays, optional
            The (X_rot, Y_rot) coordinate grid, only returned if `return_grid=True` and `dim == 2`.

        Raises
        ------
        ValueError
            If the dimension, sigma, or size arguments are invalid or inconsistent.

        Notes
        -----
        - For 2D kernels with rotation, symmetry enforcement may be affected if angle ≠ 0.
        - Kernel is auto-tagged and formatted via `to_output()`, and cast to the desired backend.
        """
        if dim < 1:
            raise ValueError("Dimension must be ≥ 1.")

        if isinstance(sigma, (int, float)):
            sigma = [sigma] * dim
        elif isinstance(sigma, list) and len(sigma) != dim:
            raise ValueError("Length of sigma must match dim.")

        if size is None:
            size = [int(2 * math.floor(truncate * s) + 1) for s in sigma]
        elif isinstance(size, int):
            size = [size] * dim
        elif isinstance(size, list) and len(size) != dim:
            raise ValueError("Length of size must match dim.")

        if any(s <= 0 or s % 2 == 0 for s in size):
            raise ValueError("Each kernel size must be a positive odd integer.")
        if any(s <= 0 for s in sigma):
            raise ValueError("Each sigma must be strictly positive.")

        if angle != 0.0 and symmetry and dim == 2:
            print("[Warning] Angle will distort symmetry. Set symmetry=False for full rotation.")

        if dim == 2:
            kernel, grid = self._generate_2d(size[0], sigma[0], symmetry, angle)
        else:
            kernel = self._generate_nd(size, sigma)
            grid = None

        if visualize and dim == 2:
            self._visualize(kernel)

        # Trace and convert
        kernel = self.to_output(kernel, tag_as="output")

        if return_numpy:
            kernel = self.to_numpy(kernel)

        return (kernel, grid) if return_grid and grid is not None else kernel

    def _generate_2d(self, size: int, sigma: float, symmetry: bool = True, angle: float = 0.0) -> ArrayLike:
        """
        Generate a 2D Gaussian kernel with optional rotation and symmetry enforcement.

        Constructs a 2D isotropic Gaussian filter, rotates the coordinate grid if needed,
        and symmetrizes the result by averaging it with its mirror. Supports both NumPy
        and Torch backends, and preserves data layout and dtype via `to_output()`.

        Parameters
        ----------
        size : int
            Size of the kernel (must be a positive odd integer).
        sigma : float
            Standard deviation of the Gaussian distribution.
        symmetry : bool, default True
            If True, average the kernel with its flipped version to enforce symmetry.
        angle : float, default 0.0
            Rotation angle in degrees applied to the 2D grid before computing the kernel.

        Returns
        -------
        kernel : torch.Tensor or np.ndarray
            The generated 2D Gaussian kernel, with dtype and backend matching configuration.
        grid : tuple of arrays
            The rotated coordinate grid (X_rot, Y_rot), useful for visualization or analysis.

        Notes
        -----
        - The kernel is normalized to sum to 1 if `self.normalize=True` in `GlobalConfig`.
        - Rotation is applied before computing the Gaussian values, not by rotating the kernel post hoc.
        - The use of symmetry + rotation may slightly distort isotropy (warning is issued).
        - Output is tagged with `tag_as='gaussian'` and passed through `to_output()`.
        """
        theta = math.radians(angle)
        var = sigma ** 2

        if self.framework == "torch":
            with torch.no_grad():
                coords = torch.arange(-(size // 2), size // 2 + 1, device=self.device, dtype=self.dtype_torch)
                X, Y = torch.meshgrid(coords, coords, indexing='ij')
                X_rot = X * math.cos(theta) + Y * math.sin(theta)
                Y_rot = -X * math.sin(theta) + Y * math.cos(theta)
                exponent = -(X_rot ** 2 + Y_rot ** 2) / (2 * var)
                exponent = torch.clamp(exponent, min=-50)
                kernel = torch.exp(exponent) / (2 * math.pi * var)

                if self.normalize:
                    kernel /= kernel.sum()
                if symmetry:
                    kernel = (kernel + torch.flip(kernel, dims=[-2, -1])) / 2

            return self.to_output(kernel, tag_as="gaussian"), (X_rot, Y_rot)

        else:
            coords = np.arange(-(size // 2), size // 2 + 1)
            X, Y = np.meshgrid(coords, coords)
            X_rot = X * np.cos(theta) + Y * np.sin(theta)
            Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

            kernel = np.exp(-(X_rot ** 2 + Y_rot ** 2) / (2 * var)) / (2 * np.pi * var)

            if self.normalize:
                kernel /= kernel.sum()
            if symmetry:
                kernel = (kernel + np.flip(kernel, axis=(0, 1))) / 2

            kernel = kernel.astype(self.dtype_numpy, copy=False)
            return self.to_output(kernel, tag_as="gaussian"), (X_rot, Y_rot)

    def _generate_nd(self, size: List[int], sigma: List[float]) -> ArrayLike:
        """
        Generate an N-dimensional isotropic or anisotropic Gaussian kernel.

        Constructs a normalized ND Gaussian kernel by evaluating the Gaussian
        function across a multi-dimensional grid. Supports both NumPy and PyTorch backends.

        Parameters
        ----------
        size : list of int
            Kernel size per dimension. Each value must be a positive odd integer.
        sigma : list of float
            Standard deviation of the Gaussian along each dimension. Must match `size` in length.

        Returns
        -------
        kernel : np.ndarray or torch.Tensor
            The generated Gaussian kernel with shape defined by `size`, and dtype/device
            matching backend configuration. Automatically tagged as `'gaussian'`.

        Notes
        -----
        - If `self.normalize=True`, the kernel is scaled to sum to 1.
        - Very small kernel sums (< 1e-8) trigger a warning and skip normalization.
        - Values are clamped to avoid numerical overflow in exponentials (e.g., exp(-x^2)).
        - Returned kernel is ND-compatible and can be used in convolution pipelines.
        """
        if self.framework == "torch":
            with torch.no_grad():
                ranges = [
                    torch.arange(-(s // 2), s // 2 + 1, device=self.device, dtype=self.dtype_torch)
                    for s in size
                ]
                grids = torch.meshgrid(*ranges, indexing='ij')
                kernel = torch.ones_like(grids[0])

                for g, s in zip(grids, sigma):
                    exponent = -(g ** 2) / (2 * s ** 2)
                    exponent = torch.clamp(exponent, min=-50)    
                    kernel *= torch.exp(exponent)

                if self.normalize:
                    total = kernel.sum()
                    if total.abs() < 1e-8:
                        print("[Warning] Kernel sum is near zero. Normalization skipped.")
                    else:
                        kernel /= total

                return self.to_output(kernel, tag_as="gaussian")

        else:
            ranges = [np.arange(-(s // 2), s // 2 + 1) for s in size]
            grids = np.meshgrid(*ranges, indexing='ij')
            kernel = np.ones_like(grids[0], dtype=self.dtype_numpy)

            for g, s in zip(grids, sigma):
                kernel *= np.exp(-(g ** 2) / (2 * s ** 2))

            if self.normalize:
                total = kernel.sum()
                if abs(total) < 1e-8:
                    print("[Warning] Kernel sum is near zero. Normalization skipped.")
                else:
                    kernel /= total

            return self.to_output(kernel.astype(self.dtype_numpy, copy=False), tag_as="gaussian")

    def _visualize(self, kernel: ArrayLike) -> None:
        """
        Visualize a 2D Gaussian kernel using matplotlib.

        Converts the input to a NumPy array if needed and displays it as a heatmap.

        Parameters
        ----------
        kernel : torch.Tensor or np.ndarray
            2D kernel to visualize. Must have exactly 2 dimensions.

        Raises
        ------
        TypeError
            If the input is not a NumPy array or torch.Tensor.
        
        Notes
        -----
        - If the input is a torch.Tensor, it is detached and moved to CPU.
        - If the kernel is not 2D, a warning is printed and nothing is shown.
        - Uses 'viridis' colormap and adds a colorbar.
        - Intended for debugging or inspection purposes only.
        """
        if isinstance(kernel, torch.Tensor):
            kernel = kernel.detach().cpu().numpy()

        if not isinstance(kernel, np.ndarray):
            raise TypeError("Kernel must be a 2D NumPy array or torch.Tensor.")

        if kernel.ndim != 2:
            print("[GaussianKernelGenerator] Visualization supports only 2D kernels (got shape: {}).".format(kernel.shape))
            return

        plt.imshow(kernel, cmap="viridis")
        plt.colorbar()
        plt.title(f"2D Gaussian Kernel ({kernel.shape[0]}x{kernel.shape[1]})")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def summary(self) -> None:
        """
        Print the current configuration of the GaussianKernelGenerator.
        """
        print("====[ GaussianKernelGenerator Summary ]====")
        print(f"Framework        : {self.framework}")
        print(f"Output Format    : {self.output_format}")
        print(f"Device           : {self.device}")
        print(f"Normalize        : {self.normalize}")
        print(f"Dtype Torch      : {self.dtype_torch}")
        print(f"Dtype NumPy      : {self.dtype_numpy}")
        print(f"Layout Name      : {self.layout_name}")
        print(f"Layout Framework : {self.layout_framework}")
        print("Axes:")
        for key, val in self.axes.items():
            print(f"  {key:<15}: {val}")

# ======================================================================
#                      Convenience wrapper
# ======================================================================

def conv(
    img: ArrayLike,
    dim: int,
    size: int,
    sigma: float,
    angle: float,
    framework: Framework = "numpy",
    output_format: Framework = "numpy",
    backend: str = "sequential",
    conv_strategy: Optional[str] = "fft",
    processor_strategy: Optional[str] = "vectorized",
    layout_framework: Framework = "numpy",
    layout_name: str = "HWC",
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Generate a Gaussian kernel and apply ND convolution to an image.

    This utility wraps `GaussianKernelGenerator` and `NDConvolver` with automatic
    configuration of layout, backend, and processing strategy. Useful for quick
    application of spatial filtering in 2D or 3D.

    Parameters
    ----------
    img : np.ndarray or torch.Tensor
        Input image to convolve.
    dim : int
        Dimensionality of the Gaussian kernel (e.g., 2 or 3).
    size : int
        Size of the kernel (must be odd).
    sigma : float
        Standard deviation of the Gaussian.
    angle : float
        Rotation angle in degrees (used only for 2D kernels).
    framework : {'numpy', 'torch'}, default 'numpy'
        Backend for internal processing and kernel generation.
    output_format : {'numpy', 'torch'}, default 'numpy'
        Format of the returned result (same options as `framework`).
    backend : str, default 'sequential'
        Processing backend used by `ImageProcessor`.
    conv_strategy : str, optional
        Convolution strategy ('fft', 'torch', 'ndimage', etc.). Auto-selected by default.
    processor_strategy : str, optional
        Processor dispatch mode ('vectorized', 'parallel', etc.). Auto-selected by default.
    layout_framework : {'numpy', 'torch'}, default 'numpy'
        Framework used to interpret layout names.
    layout_name : str, default 'HWC'
        Layout string describing axis ordering of the input image.

    Returns
    -------
    convolved : np.ndarray or torch.Tensor
        Convolved image with the same backend as specified.
    kernel : np.ndarray or torch.Tensor
        The Gaussian kernel used for the convolution.

    Notes
    -----
    - `conv_strategy` defaults to 'fft' for NumPy and 'torch' for Torch.
    - `processor_strategy` defaults to 'vectorized' for NumPy and 'torch' for Torch.
    - Internally sets up `NDConvolver` and `GaussianKernelGenerator` with layout/tag support.
    """
    # ====[ Fallback ]====
    conv_strategy=conv_strategy or "fft" if framework == "numpy" else "torch"
    processor_strategy=processor_strategy or "vectorized" if framework == "numpy" else "torch"   
        
    # ====[ Configuration ]====
    conv_params: Dict[str, Any] = {"conv_strategy": conv_strategy, "sigma": sigma, "grouped": False}
    proc_params: Dict[str, Any] = {"processor_strategy": processor_strategy,}
    layout_kernel_params: Dict[str, Any] = {"layout_name": "HW" if dim == 2 else "DHW", 
                            "layout_framework": framework}
    layout_params: Dict[str, Any] = {"layout_name": layout_name, "layout_framework": layout_framework}
    global_params: Dict[str, Any] = {"framework": framework, "output_format": output_format, "backend": backend}
    
    convolve = NDConvolver(
                        ndconvolver_cfg=NDConvolverConfig(**conv_params),        
                        img_process_cfg=ImageProcessorConfig(**proc_params),
                        layout_cfg=LayoutConfig(**layout_params),
                        global_cfg=GlobalConfig(**global_params),
                        )
    
    kernel_gen = GaussianKernelGenerator(
                                layout_cfg=LayoutConfig(**layout_kernel_params),
                                global_cfg=GlobalConfig(**global_params),
                                )

    kernel=kernel_gen.generate(
                                dim=dim,
                                size= size, 
                                sigma=sigma,  
                                angle=angle, 
                                symmetry=True
                                )
    
    return convolve(img, 
                    kernel, 
                    size=size, 
                    sigma=sigma, 
                    mode="reflect"), kernel