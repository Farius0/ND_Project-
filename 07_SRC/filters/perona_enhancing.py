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
    - Fixed algorithm: 'pm' (classic, faster version).
    - Gaussian smoothing uses sigma = 1.0.
    - Dual backend: supports both NumPy and PyTorch.
    - Output is returned in the same framework as the input.
    """

    def __init__(
        self,
        framework: Framework = "numpy",
        layout_name: str = "DHW",
        layout_framework: Framework = "numpy",
        layout_ensured_name: str = "DHW",
        alpha: float = 1e-1,
        dt: float = 4e-2,
        steps: int = 20,
        clip: bool = False,
        disable_tqdm: bool = True
    ) -> None:
        """
        Initialize the Perona enhancer with layout and denoising parameters.

        Parameters
        ----------
        framework : {"numpy", "torch"}, default "numpy"
            Backend used for processing and output format.
        layout_name : str, default "DHW"
            Input layout to be assumed for the image (e.g., DHW, HWC).
        layout_framework : {"numpy", "torch"}, default "numpy"
            Framework used to resolve layout axes.
        layout_ensured_name : str, default "DHW"
            Layout to enforce before applying the denoising algorithm.
        alpha : float, default 1e-1
            Regularization strength for Perona–Malik diffusion.
        dt : float, default 4e-2
            Time step size for the numerical diffusion process.
        steps : int, default 20
            Number of diffusion steps (iterations).
        clip : bool, default False
            Whether to clip output values to the original data range.
        disable_tqdm : bool, default True
            Disable the progress bar during processing.
        """
        
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
        """
        Run Perona–Malik diffusion on the input image using stored configuration parameters.

        Builds a `PeronaMalikDenoiser` instance with internal config dictionaries,
        then applies it to the input image.

        Parameters
        ----------
        img : ArrayLike
            Input image or volume to denoise (NumPy array or PyTorch tensor).

        Returns
        -------
        ArrayLike
            Denoised image, in the same backend as the input.
        """

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
        Apply Perona–Malik diffusion to an image or image path.

        The input can be a file path, a NumPy array, or a PyTorch tensor.
        The image is read (if needed), processed using the Perona–Malik algorithm,
        and returned in the same backend as the configured `framework`.

        Parameters
        ----------
        image : PathLike or ArrayLike
            Path to an image file or in-memory image data.

        Returns
        -------
        ArrayLike
            Denoised image using Perona–Malik diffusion.
        """

        io: ImageIO = ImageIO(
            layout_cfg=LayoutConfig(**self._layout_params),
            global_cfg=GlobalConfig(**self._global_params),
        )
        img: ArrayLike = io.read_image(image, framework=self.framework)
        return self.perona_algo(img)



