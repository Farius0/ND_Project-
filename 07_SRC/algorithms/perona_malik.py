# ==================================================
# =============  MODULE: perona_malik ==============
# ==================================================
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple, Union
import numpy as np, torch
from tqdm import tqdm

from core.operator_core import OperatorCore
from operators.diff_operator import DiffOperator
from operators.image_processor import ImageProcessor
from filters.edge_aware_filter import EdgeAwareFilter
from operators.gaussian import (
    GaussianKernelGenerator as kernel_generator,
    NDConvolver as convolver,
)
from core.config import (
    LayoutConfig,
    GlobalConfig,
    FilterConfig,
    NDConvolverConfig,
    ImageProcessorConfig,
    DiffOperatorConfig,
    AlgorithmConfig,
)

ArrayLike = Union[np.ndarray, torch.Tensor]
Framework = Literal["numpy", "torch"]
AlgoName = Literal["pm", "enhanced"]

# ==================================================
# ================== PERONA-MALIK ==================
# ==================================================

class PeronaMalikDenoiser(OperatorCore):
    """
    Perona–Malik anisotropic diffusion (classic and enhanced variants).

    Notes
    -----
    - Dual-backend: NumPy & Torch.
    - ND-ready; kernel generator adapts to 2D ('HW') or 3D ('DHW') smoothing for the enhanced variant.
    - Tag propagation preserved via OperatorCore.track().
    """

    def __init__(
        self,
        algo_cfg: AlgorithmConfig = AlgorithmConfig(),
        filter_cfg: FilterConfig = FilterConfig(),
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
        self.filter_cfg: FilterConfig = filter_cfg
        self.algo_cfg: AlgorithmConfig = algo_cfg
        
         # === Layout: resolved axes, layout name, layout dict ===
        self.axes: Dict[str, Any] = self.layout_cfg.resolve(include_meta=True)
        self.layout_name: str = self.axes.get("layout_name")
        self.layout: Dict[str, Any] = self.axes.get("layout")
        self.layout_framework: str = self.axes.get("layout_framework")

        self.channel_axis: Optional[int] = self.axes.get("channel_axis")
        self.batch_axis: Optional[int] = self.axes.get("batch_axis")
        self.direction_axis: Optional[int] = self.axes.get("direction_axis")
        self.height_axis: Optional[int] = self.axes.get("height_axis")
        self.width_axis: Optional[int] = self.axes.get("width_axis")
        self.depth_axis: Optional[int] = self.axes.get("depth_axis")
        
        # ====[ Store pm-specific parameters ]====
        self.dt: float = float(self.algo_cfg.dt)
        self.steps: int = int(self.algo_cfg.steps)
        self.algorithm: str = str(self.algo_cfg.algorithm_strategy)
        self.return_evolution: bool = bool(self.algo_cfg.return_evolution)
        self.disable_tqdm: bool = bool(self.algo_cfg.disable_tqdm)
        self.clip: bool = bool(self.algo_cfg.clip)
        
        # ====[ Mirror inherited params locally for easy access ]====
        self.framework: Framework = self.global_cfg.framework.lower()
        self.output_format: Framework = self.global_cfg.output_format.lower()
        self.add_batch_dim: Optional[bool] = self.global_cfg.add_batch_dim
        self.add_channel_dim: Optional[bool] = self.global_cfg.add_channel_dim
        self.normalize: bool = bool(self.global_cfg.normalize)        
        self.device: str = "cuda" if (torch.cuda.is_available() and self.framework == "torch") else self.global_cfg.device
        self.filter_mode: str = self.filter_cfg.mode
        self.alpha: float = float(self.filter_cfg.alpha)
        self.as_float: bool = bool(self.filter_cfg.as_float)  # kept for consistency
        self.dim: int = int(self.conv_cfg.dim)
        self.sigma: Union[float, Tuple[float, float]] = self.conv_cfg.sigma
        self.size: Union[int, Tuple[int, int]] = self.conv_cfg.size
        self.angle: Optional[float] = self.conv_cfg.angle      

        if self.dt <= 0 or self.steps <= 0 or self.alpha <= 0:
            raise ValueError("dt, steps and alpha must be positive.")
        
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
        
        self.kernel = self.kernel_gen.generate(
            dim=self.dim,
            size=self.size, 
            sigma=self.sigma,  
            angle=self.angle, 
            symmetry=False
            )         
        
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
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg,
        )
        
        # ====[ Create EdgeAwareFilter ]====
        self.edge_filter: EdgeAwareFilter = EdgeAwareFilter(
            filter_cfg = self.filter_cfg,
            img_process_cfg = self.img_process_cfg,
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg,
        )

    def __call__(self, image: ArrayLike) -> ArrayLike:
        """
        Run the selected Perona–Malik algorithm on `image` and convert to output format.
        """
        img = self.convert_once(image)
        result = self._detect(img)
        return self.to_output(result, tag_as="algorithm")

    # ---- internal helpers -------------------------------------------------
        
    def _detect(self, image: ArrayLike) -> ArrayLike:
        """
        Dispatch to the requested algorithm variant.
        """
        algorithms = {
            "pm": self._denoise_pm,
            "enhanced": self._denoise_enhanced_pm,
        }
        if self.algorithm not in algorithms:
            raise ValueError(f"Unknown algorithm method: {self.algorithm!r}")
        return algorithms[self.algorithm](image)

    def _convolve(self, image: ArrayLike) -> ArrayLike:
        """Apply ND convolution with the precomputed kernel."""
        return self.convolve(image, self.kernel)
    
    def _convolve(self, image):
        return self.convolve(image, self.kernel)

    # ---- classic PM ------------------------------------------------------- 
    def _denoise_pm(self, image: ArrayLike) -> ArrayLike:
        """
        Classic Perona–Malik diffusion with edge-aware conductivity g(|∇u|).
        """
        # ====[ Torch version ]====
        def diffuse_torch(channel: torch.Tensor) -> Union[torch.Tensor, list[torch.Tensor]]:
            tagger = self.track(channel)
            u = tagger.copy_to(channel.clone().float()).image
            evolution = [u.clone()] if self.return_evolution else None

            for _ in tqdm(range(self.steps), desc="Perona-Malik (torch)", disable=self.disable_tqdm):
                grad, grad_norm = self.diff.gradient(u, to_return=["gradient", "magnitude"])
                weight = self.edge_filter(grad_norm)
                div = self.diff.divergence(grad, weight=weight)
                u = u + self.dt * div
                u = torch.clamp(u, 0.0, 1.0) if self.clip else u
                u = tagger.copy_to(u).get()
                if self.return_evolution:
                    evolution.append(u.clone())

            return evolution if self.return_evolution else u

        # ====[ NumPy version ]====
        def diffuse_numpy(channel: np.ndarray) -> Union[np.ndarray, list[np.ndarray]]:
            tagger = self.track(channel)
            u = tagger.copy_to(channel.astype(np.float32, copy=False)).image
            evolution = [u.copy()] if self.return_evolution else None

            for _ in tqdm(range(self.steps), desc="Perona-Malik (numpy)", disable=self.disable_tqdm):
                grad, grad_norm = self.diff.gradient(u, to_return=["gradient", "magnitude"])
                weight = self.edge_filter(grad_norm)
                u = u + self.dt * self.diff.divergence(grad, weight=weight)
                u = np.clip(u, 0.0, 1.0) if self.clip else u
                u = tagger.copy_to(u).get()
                if self.return_evolution:
                    evolution.append(u.copy())

            return evolution if self.return_evolution else u

        self.processor.function = diffuse_torch if self.framework == "torch" else diffuse_numpy
        result = self.processor(image)
        
        return result

    # ---- enhanced PM ------------------------------------------------------
    
    def _denoise_enhanced_pm(self, image: ArrayLike) -> ArrayLike:
        """
        Enhanced PM: conductivity uses |∇(Gσ * u)| while gradient/divergence applied on u.
        """
        
        # ====[ Torch version ]====
        def enhanced_torch(channel: torch.Tensor) -> Union[torch.Tensor, list[torch.Tensor]]:
            tagger = self.track(channel)
            u = tagger.copy_to(channel.clone().float()).image
            evolution = [u.clone()] if self.return_evolution else None

            for _ in tqdm(range(self.steps), desc="Enhanced Perona-Malik (torch)", disable=self.disable_tqdm):
                grad_u = self.diff.gradient(u)  # default: returns gradient field
                conv = self._convolve(u)
                grad_norm = self.diff.gradient(conv, to_return=["magnitude"])
                weight = self.edge_filter(grad_norm)
                u = u + self.dt * self.diff.divergence(grad_u, weight=weight)
                u = torch.clamp(u, 0.0, 1.0) if self.clip else u
                u = tagger.copy_to(u).get()
                if self.return_evolution:
                    evolution.append(u.clone())

            return evolution if self.return_evolution else u

        # ====[ NumPy version ]====
        def enhanced_numpy(channel: np.ndarray) -> Union[np.ndarray, list[np.ndarray]]:
            tagger = self.track(channel)
            u = tagger.copy_to(channel.astype(np.float32, copy=False)).image
            evolution = [u.copy()] if self.return_evolution else None

            for _ in tqdm(range(self.steps), desc="Enhanced Perona-Malik (numpy)", disable=self.disable_tqdm):
                grad_u = self.diff.gradient(u)
                conv = self._convolve(u)
                grad_norm = self.diff.gradient(conv, to_return=["magnitude"])
                weight = self.edge_filter(grad_norm)
                u = u + self.dt * self.diff.divergence(grad_u, weight=weight)
                u = np.clip(u, 0.0, 1.0) if self.clip else u
                u = tagger.copy_to(u).get()
                if self.return_evolution:
                    evolution.append(u.copy())

            return evolution if self.return_evolution else u

        self.processor.function = enhanced_torch if self.framework == "torch" else enhanced_numpy
        result = self.processor(image)
        return result

# ======================================================================
#                      Convenience wrapper
# ======================================================================
def pm(
    img: ArrayLike,
    alpha: float = 1e-1,
    dt: float = 4e-2,
    steps: int = 20,
    sigma: Union[float, Tuple[float, float]] = 1.0,
    algorithm: AlgoName = "pm",
    filter_mode: str = "pm",
    framework: Framework = "numpy",
    output_format: Framework = "numpy",
    layout_name: str = "HWC",
    layout_framework: Framework = "numpy",
    processor_strategy: Optional[str] = "classic",
    diff_strategy: Optional[str] = "classic",
    conv_strategy: Optional[str] = "convolve2d",
    disable_tqdm: bool = False,
) -> ArrayLike:
    """
    Convenience wrapper around `PeronaMalikDenoiser` with reasonable defaults.

    Parameters
    ----------
    img
        Input image (NumPy array or Torch tensor).
    alpha, dt, steps, sigma
        Filter, time step, iteration count, and smoothing scale.
    algorithm
        'pm' (classic) or 'enhanced'.
    framework, output_format
        Backend and output format ('numpy' | 'torch').
    layout_name, layout_framework
        Layout/axes configuration passed to LayoutConfig.
    processor_strategy, diff_strategy, conv_strategy
        Low-level strategy hints; fallbacks depend on the chosen framework.
    disable_tqdm
        Hide progress bars if True.
    """
    
    # ====[ Fallback ]====
    diff_strategy=diff_strategy or "vectorized" if framework == "numpy" else "torch"
    conv_strategy=conv_strategy or "fft" if framework == "numpy" else "torch"
    processor_strategy=processor_strategy or "parallel" if framework == "numpy" else "torch"   
        
    # ====[ Configuration ]====
    algorithm_params = {"dt": dt, "steps": steps, "clip": False, "algorithm_strategy": algorithm, "disable_tqdm":disable_tqdm}
    filter_params = {"mode": filter_mode, "alpha": alpha}
    conv_params = {"conv_strategy": conv_strategy, "sigma": sigma}
    proc_params = {"processor_strategy": processor_strategy,}
    layout_params = {"layout_name": layout_name,"layout_framework": layout_framework}
    global_params = {"framework": framework, "output_format": output_format,}
    diff_params = {"diff_strategy": diff_strategy,"boundary_mode":{"gradient": "dirichlet", "divergence": "dirichlet"},                             
                        "diff_mode": {"gradient": "forward","divergence": "backward"},}

    pm_denoise = PeronaMalikDenoiser(
                    algo_cfg=AlgorithmConfig(**algorithm_params),
                    filter_cfg=FilterConfig(**filter_params),
                    diff_operator_cfg=DiffOperatorConfig(**diff_params),
                    ndconvolver_cfg=NDConvolverConfig(**conv_params),
                    img_process_cfg=ImageProcessorConfig(**proc_params),
                    layout_cfg=LayoutConfig(**layout_params),
                    global_cfg=GlobalConfig(**global_params),
                    )
    
    img_copy = pm_denoise.safe_copy(img)

    return pm_denoise(img_copy)