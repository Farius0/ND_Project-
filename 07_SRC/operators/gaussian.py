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
    ND convolution with multiple strategies (FFT, ndimage, uniform, gaussian,
    median, torch, convolve2d, gaussian_torch). Works with ND/chan/batch via
    ImageProcessor, preserving backend (NumPy/Torch) and tags.
    """

    def __init__(
        self,
        ndconvolver_cfg: NDConvolverConfig = NDConvolverConfig(),
        img_process_cfg: ImageProcessorConfig = ImageProcessorConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        NDConvolver supports FFT, ndimage, uniform, gaussian, median and torch convolution strategies.
        Uses ImageProcessor for slice-wise or channel-wise ND processing.

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
        Apply ND convolution using the configured strategy.

        Parameters
        ----------
        image : ndarray | Tensor
            Input image.
        kernel : ndarray | Tensor
            Convolution kernel (compatible with selected backend).
        size : int, optional
            Window size for uniform/median filters if needed.
        sigma : float, default 1.0
            Std for Gaussian filtering strategies.
        mode : str, default 'reflect'
            Border handling for ndimage strategies.

        Returns
        -------
        ndarray | Tensor
            Convolved output, tagged and formatted via OperatorCore.
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
        Dispatch the appropriate convolution strategy and process the image.

        Parameters
        ----------
        image : np.ndarray | torch.Tensor
            Input image.
        kernel : np.ndarray | torch.Tensor
            Convolution kernel.
        sigma : float
            Gaussian kernel sigma.
        size : int
            Median kernel size.
        mode : str
            Convolution mode.

        Returns
        -------
        result : np.ndarray | torch.Tensor
            Convolved image.
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
        Build a Torch-compatible convolution closure for ND slices, supporting:
        - 2D image (H,W)  -> conv2d with [1,1,Kh,Kw]
        - 3D no-channel (D,H,W) -> conv3d with [1,1,Kd,Kh,Kw]
        - 3D with channels (C,H,W) -> conv2d(groups=C)
        - 4D with channels (C,D,H,W) -> conv3d(groups=C)

        Channel is internally moved to dim=0 when present, and moved back before return.

        Parameters
        ----------
        kernel : np.ndarray | torch.Tensor
            Kernel to apply.

        Returns
        -------
        convolve_fn : callable
            Callable function that takes a slice and applies convolution.
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
        Compute convolution padding from kernel shape and selected padding mode.

        Parameters
        ----------
        kernel_shape : tuple[int]
            Shape of the kernel in spatial dimensions (e.g., (3, 3) or (5, 5, 5)).

        Returns
        -------
        padding : tuple[int]
            Padding per dimension.
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
    Generate Gaussian kernels in ND (or 2D with rotation), compatible with NumPy and Torch.

    Notes
    -----
    - Normalization optional via GlobalConfig.normalize.
    - 2D rotation uses an angle in degrees; symmetry averaging available.
    """

    def __init__(
        self,
        *,
        ndconvolver_cfg: NDConvolverConfig = NDConvolverConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize a GaussianKernelGenerator for ND kernel construction.

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
    )-> ArrayLike:
        """
        Generate a Gaussian kernel in arbitrary dimension.

        Parameters
        ----------
        dim : int
            Dimension of the kernel (e.g., 2 for 2D).
        size : int or list[int]
            Kernel size per dimension. If None, inferred from sigma and truncate.
        sigma : float or list[float]
            Standard deviation(s) of the Gaussian.
        truncate : float
            Controls the spatial extent of the kernel (in std dev units).
        symmetry : bool
            Enforce symmetry by averaging with its mirror.
        angle : float
            Angle (in degrees) for rotation in 2D.
        visualize : bool
            If True, plot the kernel (only if dim==2).
        return_grid : bool
            If True, return the coordinate grid used (2D only).
        return_numpy : bool
            If True, return kernel as NumPy array.

        Returns
        -------
        kernel : torch.Tensor | np.ndarray
            The generated Gaussian kernel.
        grid : tuple (X_rot, Y_rot), optional
            Only for dim=2 and return_grid=True.
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

    def _generate_2d(self, size: int, sigma: float, symmetry: bool = True, angle: float = 0.0)-> ArrayLike:

        """
        Generate a 2D Gaussian kernel, optionally rotated and symmetrized.

        Parameters
        ----------
        size : int
            Kernel size (must be odd).
        sigma : float
            Standard deviation of the Gaussian.
        symmetry : bool
            If True, average the kernel with its mirror.
        angle : float
            Rotation angle in degrees (applied to 2D grid).

        Returns
        -------
        kernel : tensor or ndarray
            The resulting 2D Gaussian kernel.
        grid : tuple
            Rotated coordinate grid (X_rot, Y_rot)
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

    def _generate_nd(self, size: List[int], sigma: List[float])-> ArrayLike:
        """
        Generate a normalized ND Gaussian kernel.

        Parameters
        ----------
        size : list[int]
            Size of the kernel in each dimension.
        sigma : list[float]
            Standard deviation per dimension.

        Returns
        -------
        kernel : torch.Tensor or np.ndarray
            ND Gaussian kernel.
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

        Parameters
        ----------
        kernel : torch.Tensor | np.ndarray
            Kernel to visualize. Must be 2D.
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
    Convenience front-end for generating a Gaussian kernel and convolving an image.

    Notes
    -----
    - `conv_strategy` defaults to 'fft' for NumPy, 'torch' for Torch.
    - `processor_strategy` defaults to 'vectorized' for NumPy, 'torch' for Torch.
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