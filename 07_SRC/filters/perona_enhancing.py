# ==================================================
# =========== MODULE: perona_enhancing =============
# ==================================================
from __future__ import annotations

from typing import Union, Literal, Optional, Dict, Any

from pathlib import Path
import numpy as np, torch
# from utils.decorators import timer
from operators.image_io import ImageIO
from algorithms.perona_malik import PeronaMalikDenoiser
from core.config import (LayoutConfig, GlobalConfig, FilterConfig, 
    NDConvolverConfig, ImageProcessorConfig, DiffOperatorConfig, AlgorithmConfig,)

# Public API
__all__ = ["PeronaEnhancer"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Framework = Literal["numpy", "torch"]
PathLike = Union[str, Path]

class PeronaEnhancer:
    """
    Lightweight façade around `PeronaMalikDenoiser`.

    Notes
    -----
    - Fixed algorithm: 'pm' (classic more fast), no 'enhanced' option.
    - Sigma set to 1.0 for the convolution used by internal operators.
    - Dual backend: 'numpy' or 'torch'. Output in the same backend.
    """
    def __init__(self,
                 framework: Framework="numpy",
                 layout_name: str="DHW",
                 layout_framework: Framework="numpy",
                 layout_ensured_name: str="DHW",
                 alpha: float=1e-1,
                 dt: float=4e-2,
                 steps: int=20,
                 clip: bool=False,    
                 disable_tqdm: bool=True                             
                 ) -> None:
        
        # ====[ Configuration ]====
        self._algorithm_params: Dict[str, Any] = {"dt": dt, "steps": steps, "clip": clip, 
                                  "algorithm_strategy": "pm", "disable_tqdm":disable_tqdm}
        self._filter_params: Dict[str, Any] = {"mode": "pm", "alpha": alpha}
        self._conv_params: Dict[str, Any] = {"conv_strategy": "fft" if framework == "numpy" else "torch", "sigma": 1.0}
        self._proc_params: Dict[str, Any] = {"processor_strategy": "parallel" if framework == "numpy" else "torch",}
        self._layout_params: Dict[str, Any] = {"layout_name": layout_name, "layout_ensured_name": layout_ensured_name,
                               "layout_framework": layout_framework}
        self._global_params: Dict[str, Any] = {"framework": framework, "output_format": framework,}
        self._diff_params: Dict[str, Any] = {"diff_strategy": "vectorized" if framework == "numpy" else "torch",
                            "boundary_mode":{"gradient": "dirichlet","divergence": "dirichlet"},                             
                            "diff_mode": {"gradient": "forward","divergence": "backward"},}
        
        self.framework: Framework = framework
        
    def perona_algo(self, img: ArrayLike) -> ArrayLike:
        """Build a `PeronaMalikDenoiser` with stored configs and run it."""
        denoiser = PeronaMalikDenoiser(
            algo_cfg=AlgorithmConfig(**self._algorithm_params),
            filter_cfg=FilterConfig(**self._filter_params),
            diff_operator_cfg=DiffOperatorConfig(**self._diff_params),
            ndconvolver_cfg=NDConvolverConfig(**self._conv_params),
            img_process_cfg=ImageProcessorConfig(**self._proc_params),
            layout_cfg=LayoutConfig(**self._layout_params),
            global_cfg=GlobalConfig(**self._global_params),
        )
        return denoiser(img)   
        
    # @log_exceptions(logger_name="error_logger")
    # @timer(return_result=True, return_elapsed=False, name ="perona_enhancing")
    def __call__(self, image: Union[PathLike, ArrayLike]) -> ArrayLike:
        """
        Read `image` (path/array/tensor), run Perona–Malik, return same backend as `framework`.
        """
        io: ImageIO = ImageIO(
            layout_cfg=LayoutConfig(**self._layout_params),
            global_cfg=GlobalConfig(**self._global_params),
        )
        img: ArrayLike = io.read_image(image, framework=self.framework)
        return self.perona_algo(img)



