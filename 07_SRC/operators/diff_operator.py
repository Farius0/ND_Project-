# ==================================================
# === MODULE: diff_operator (Gradient, Div, Lap) ===
# ==================================================
from __future__ import annotations

from typing import Any, List, Sequence, Tuple, Union, Dict, Optional, Literal
from joblib import Parallel, delayed
from scipy.ndimage import sobel
import numpy as np, torch
# from tqdm import tqdm

from core.operator_core import OperatorCore
from core.layout_axes import get_layout_axes, resolve_and_clean_layout_tags
from core.config import LayoutConfig, GlobalConfig, DiffOperatorConfig, NDConvolverConfig

# Public API
__all__ = ["DiffOperator", "diffop"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Framework = Literal["numpy", "torch"]

# ==================================================
# ================== DiffOperator ==================
# ==================================================

class DiffOperator(OperatorCore):
    """
    Differential operator for computing gradients, divergences, and Laplacians
    on ND images using various strategies (NumPy, Torch, Parallel).
    """

    def __init__(
        self,
        diff_operator_cfg: DiffOperatorConfig = DiffOperatorConfig(),
        ndconvolver_cfg: NDConvolverConfig = NDConvolverConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize a differential operator (gradient, divergence, laplacian)
        compatible with ND tensors and multiple computation strategies.

        """
        # ====[ Configuration ]====
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg
        self.conv_cfg: NDConvolverConfig = ndconvolver_cfg
        self.diff_cfg: DiffOperatorConfig = diff_operator_cfg
        
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
        self.spacing: Optional[Union[List[float], float]] = self.diff_cfg.spacing
        self.diff_mode: Optional[Dict[str, str]] = self.diff_cfg.diff_mode
        self.diff_fallback: Optional[Union[str, bool]]  = self.diff_cfg.diff_fallback
        self.boundary_mode: Optional[Dict[str, str]] = self.diff_cfg.boundary_mode
        self.include_channel_gradient: Optional[bool] = self.diff_cfg.include_channel_gradient
        self.max_flux: Optional[float] = self.diff_cfg.max_flux
        self.strategy: str = self.diff_cfg.diff_strategy.lower()

        STRATEGIES = ['auto', 'torch', 'parallel', 'classic', 'vectorized']

        if self.strategy not in STRATEGIES:
            raise ValueError(f"Unsupported strategy '{self.strategy}'. Use one of: {STRATEGIES}")
        
        if not isinstance(self.diff_mode, dict):
            raise TypeError(f"`diff_mode` must be a dict, got {type(self.diff_mode)}")
        
        if not isinstance(self.boundary_mode, dict):
            raise TypeError(f"`boundary_mode` must be a dict, got {type(self.boundary_mode)}")
        
        # ====[ Mirror inherited params locally for easy access ]====
        self._framework_auto: bool = self.global_cfg.framework == 'auto'
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
        self._log(f"[DiffOperator] Strategy = {self.strategy}, Backend = {self.backend}, Spacing = {self.diff_cfg.spacing}")
                
        # ====[ Initialize OperatorCore with all axes ]====
        super().__init__(
            layout_cfg = self.layout_cfg,
            global_cfg = self.global_cfg, 
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ---------- Helpers: backend/strategy routing ----------

    def _strategy_expects(self, fw_current: str) -> str:
        """
        Return the backend expected by the chosen strategy.
        """
        if self.strategy == "torch":
            return "torch"
        if self.strategy in ("parallel", "classic", "vectorized"):
            return "numpy"
        # auto → keep current
        return fw_current

    def _ensure_backend(self, x: ArrayLike, target_fw: str) -> ArrayLike:
        """
        Convert `x` to the target backend when needed, preserving tags/layout.
        """
        if target_fw == "torch" and isinstance(x, np.ndarray):
            return self.to_framework(x, framework="torch", status="input")
        if target_fw == "numpy" and isinstance(x, torch.Tensor):
            return self.to_framework(x, framework="numpy", status="input")
        return x            

# ==================================================
# =================== Public API ===================
# ==================================================

    def gradient(
        self,
        u: ArrayLike,
        enable_uid: bool = False,
        op_params: dict | None = None,
        framework: Framework | None = None,
        output_format: Framework | None = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: bool | None = None,
        to_return: Sequence[str] = ("gradient",),
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        """
        Compute the ND gradient of an input tensor/image.

        Parameters
        ----------
        u : Tensor | ndarray
            Input image.
        enable_uid : bool
            Enable tagging with UID.
        op_params : dict | None
            Optional metadata for tagging.
        framework : str | None
            Force backend (torch, numpy).
        output_format : str | None
            Output format.
        track : bool
            Enable AxisTracker propagation.
        trace_limit : int
            Tag history limit.
        normalize_override : bool | None
            Override normalize behavior locally.
        to_return : sequence of {'gradient','magnitude'}
            Choose outputs to return.

        Returns
        -------
        grad or (grad, magnitude)
            Tagged outputs in requested format.
        """
        # === Convert input image and assign framework ===
        u = self.convert_once(
            image=u,
            tag_as="input",
            framework=framework or self.framework,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )

        # Infer backend if auto
        if self._framework_auto: 
            if isinstance(u, torch.Tensor): 
                self.framework = "torch" 
            elif isinstance(u, np.ndarray): 
                self.framework = "numpy" 
            else: 
                raise TypeError("[DiffOperator] Cannot infer framework for input type.")

        # Ensure backend matches strategy needs
        expected_fw = self._strategy_expects(self.framework)
        u = self._ensure_backend(u, expected_fw)
        self.framework = "torch" if isinstance(u, torch.Tensor) else "numpy"

        # Dispatch
        if self.strategy == "torch" or (self.strategy == "auto" and self.framework == "torch"):
            grad = self._gradient_torch(u)
        elif self.strategy == "parallel":
            grad = self._gradient_numpy_parallel(u)
        elif self.strategy == "classic":
            grad = self._gradient_numpy_classic(u)
        else:
            grad = self._gradient_numpy(u)
            
        if "magnitude" in to_return:
            # ====[ Layout Handling ]=====     
            tagger = self.track(u)
            grad_mag = self.safe_l2_norm(grad, axis=0)
            tracker = tagger.clone_to(grad_mag, updates = {"status": "magnitude", "shape_after": grad_mag.shape})

        # === Tag and return output ===
        grad_result = self.to_output(
                grad,
                tag_as="output",
                framework=output_format or self.output_format,
                enable_uid=enable_uid,
                op_params=op_params,
                track=track,
                trace_limit=trace_limit,
                normalize_override=normalize_override
            )
        
        if all(grad in to_return for grad in ("magnitude", "gradient")):
            return grad_result, tracker.get()
        elif "magnitude" in to_return:
            return tracker.get()
        elif "gradient" in to_return:
            return grad_result  

    def divergence(
        self,
        v: ArrayLike,
        weight: ArrayLike | None = None,
        enable_uid: bool = False,
        op_params: dict | None = None,
        framework: Framework | None = None,
        output_format: Framework | None = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: bool | None = None,
    ) -> ArrayLike:
        """
        Compute the divergence of a vector field v (e.g., gradient output).

        Parameters
        ----------
        v : torch.Tensor | np.ndarray
            Input ND vector field (first axis = direction axis).
        weight : torch.Tensor | np.ndarray | None
            Optional weight tensor.
        enable_uid : bool
            Enable tagging with UID.
        op_params : dict | None
            Optional metadata for tagging.
        framework : str | None
            Force backend (torch, numpy).
        output_format : str | None
            Output format.
        track : bool
            Enable AxisTracker propagation.
        trace_limit : int
            Tag history limit.
        normalize_override : bool | None
            Override normalize behavior locally.

        Returns
        -------
        div : torch.Tensor | np.ndarray
            ND divergence image.
        """
        # === Preprocess and tag input ===
        v = self.convert_once(
            image=v,
            tag_as="input",
            framework=framework or self.framework,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )

        if self._framework_auto: 
            if isinstance(v, torch.Tensor): 
                self.framework = "torch" 
            elif isinstance(v, np.ndarray): 
                self.framework = "numpy" 
            else: 
                raise TypeError("[DiffOperator] Cannot infer framework for input type.")

        expected_fw = self._strategy_expects(self.framework)
        v = self._ensure_backend(v, expected_fw)
        self.framework = "torch" if isinstance(v, torch.Tensor) else "numpy"

        if self.strategy == "torch" or (self.strategy == "auto" and self.framework == "torch"):
            div = self._divergence_torch(v, weight=weight)
        elif self.strategy == "parallel":
            div = self._divergence_numpy_parallel(v, weight=weight)
        elif self.strategy == "classic":
            div = self._divergence_numpy_classic(v, weight=weight)
        else:
            div = self._divergence_numpy(v, weight=weight)

        # === Tag and return output ===
        return self.to_output(
            div,
            tag_as="output",
            framework=output_format or self.output_format,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )

    def laplacian(
        self,
        u: ArrayLike,
        enable_uid: bool = False,
        op_params: dict | None = None,
        framework: Framework | None = None,
        output_format: Framework | None = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: bool | None = None,
    ) -> ArrayLike:
        """
        Compute the ND Laplacian of input `u` as div(grad(u)).

        Parameters
        ----------
        u : torch.Tensor | np.ndarray
            Input ND scalar image.
        enable_uid : bool
            Enable tagging and UID tracking.
        op_params : dict | None
            Optional metadata for tagging.
        framework : str | None
            Backend to use ('torch' or 'numpy').
        output_format : str | None
            Desired output format.
        track : bool
            Enable AxisTracker tagging.
        trace_limit : int
            Tag memory depth.
        normalize_override : bool | None
            Override global normalization.

        Returns
        -------
        laplacian : torch.Tensor | np.ndarray
            The Laplacian of `u`, tagged and formatted.
        """
        # Step 1 — Gradient
        grad_u = self.gradient(
            u,
            enable_uid=enable_uid,
            op_params=op_params,
            framework=framework or self.framework,
            output_format=framework or self.framework,  # Important for divergence
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )

        # Step 2 — Divergence of Gradient
        lap = self.divergence(
            grad_u,
            enable_uid=enable_uid,
            op_params=op_params,
            framework=framework or self.framework,
            output_format=output_format or self.output_format,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )

        # Step 3 — Final output
        return self.to_output(
            lap,
            tag_as="output",
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )
    
    def hessian(        
        self,
        u: ArrayLike,
        enable_uid: bool = False,
        op_params: dict | None = None,
        framework: Framework | None = None,
        output_format: Framework | None = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: bool | None = None,
    ) -> ArrayLike:
        """
        Compute the full ND Hessian matrix H(u) directly from scalar input u.

        Returns
        -------
        hess : torch.Tensor | np.ndarray
            ND Hessian matrix of shape (D, D, ...)
        """
        u = self.convert_once(
            image=u,
            tag_as="input",
            framework=framework or self.framework,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override,
        )
        
        # Auto dispatch
        if self._framework_auto: 
            if isinstance(u, torch.Tensor): 
                self.framework = "torch" 
            elif isinstance(u, np.ndarray): 
                self.framework = "numpy" 
            else: 
                raise TypeError("[DiffOperator] Unknown input type.")

        expected_fw = self._strategy_expects(self.framework)
        u = self._ensure_backend(u, expected_fw)
        self.framework = "torch" if isinstance(u, torch.Tensor) else "numpy"

        if self.strategy == "torch" or (self.strategy == "auto" and self.framework == "torch"):
            hess = self._hessian_torch(u)
        elif self.strategy == "parallel":
            hess = self._hessian_numpy_parallel(u)
        elif self.strategy == "classic":
            hess = self._hessian_numpy_classic(u)
        else:
            hess = self._hessian_numpy(u)

        return self.to_output(
            hess,
            tag_as="output",
            framework=output_format or self.output_format,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )

# =====================================================
# =================== Other methods ===================
# =====================================================
    
    def sobel(
        self,
        u: ArrayLike,
        enable_uid: bool = False,
        op_params: dict | None = None,
        framework: Framework | None = None,
        output_format: Framework | None = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: bool | None = None,
    ) -> ArrayLike:
        """
        Compute the Sobel gradient magnitude of an ND image.

        Parameters
        ----------
        u : torch.Tensor | np.ndarray
            Input image or volume.
        Same parameters as gradient().

        Returns
        -------
        sobel_mag : torch.Tensor | np.ndarray
            Sobel magnitude map.
        """
        # === Convert input ===
        u = self.convert_once(
            image=u,
            tag_as="input",
            framework=framework or self.framework,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )

        # === Detect backend ===
        if self._framework_auto:
            if isinstance(u, torch.Tensor):
                self.framework = "torch"
            elif isinstance(u, np.ndarray):
                self.framework = "numpy"
            else:
                raise TypeError("[DiffOperator] Cannot infer framework for Sobel input.")

        # === Dispatch backend ===
        if self.framework == "torch":
            if self.diff_fallback:                
                u = self.to_output(u, tag_as="input", framework="numpy",)
                self.framework = "numpy"
                sobel_mag = self._sobel_numpy(u)
                self.framework = "torch"
            else:    
                sobel_mag = self._sobel_torch(u)
        elif self.framework == "numpy":
            sobel_mag = self._sobel_numpy(u)
        else:
            raise ValueError(f"[DiffOperator] Unsupported framework '{self.framework}' for Sobel.")
        
        # === Output ===
        return self.to_output(
            sobel_mag,
            tag_as="output",
            framework=output_format or self.output_format,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )
        
    def sobel_gradient(
        self,
        u: ArrayLike,
        enable_uid: bool = False,
        op_params: dict | None = None,
        framework: Framework | None = None,
        output_format: Framework | None = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: bool | None = None,
    ) -> ArrayLike:
        """
        ND Sobel gradient (Torch or NumPy).
        """
        # === Convert input ===
        u = self.convert_once(
            image=u,
            tag_as="input",
            framework=framework or self.framework,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )

        # === Detect backend ===
        if self._framework_auto:
            if isinstance(u, torch.Tensor):
                self.framework = "torch"
            elif isinstance(u, np.ndarray):
                self.framework = "numpy"
            else:
                raise TypeError("[DiffOperator] Cannot infer framework for Sobel input.")

        # === Dispatch backend ===
        if self.framework == "torch":
            if self.diff_fallback:
                u = self.to_output(u, tag_as="input", framework="numpy",)
                self.framework = "numpy"
                grad = self._sobel_numpy(u, gradient=True)
                self.framework = "torch"
            else:    
                grad = self._sobel_torch(u, gradient=True)
        elif self.framework == "numpy":
            grad = self._sobel_numpy(u, gradient=True)
        else:
            raise ValueError(f"[DiffOperator] Unsupported framework '{self.framework}' for Sobel.")

        # === Output ===
        return self.to_output(
            grad,
            tag_as="output",
            framework=output_format or self.output_format,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )
        
    def sobel_hessian(
        self,
        u: ArrayLike,
        enable_uid: bool = False,
        op_params: dict | None = None,
        framework: Framework | None = None,
        output_format: Framework | None = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: bool | None = None,
    ) -> ArrayLike:
        """
        Compute the ND Hessian matrix using Sobel second derivatives (Torch or NumPy).
        """
        # Convert and assign backend
        u = self.convert_once(
            image=u,
            tag_as="input",
            framework=framework or self.framework,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )

        # Detect framework automatically
        if self._framework_auto:
            if isinstance(u, torch.Tensor):
                self.framework = "torch"
            elif isinstance(u, np.ndarray):
                self.framework = "numpy"
            else:
                raise TypeError("[DiffOperator] Cannot infer framework for Sobel Hessian input.")

        # Dispatch to correct backend
        if self.framework == "torch":
            if self.diff_fallback:
                u_np = self.to_output(u, tag_as="input", framework="numpy",)
                self.framework = "numpy"
                hess = self._sobel_hessian_numpy(u_np)
                self.framework = "torch"
            else:
                hess = self._sobel_hessian_torch(u)
        elif self.framework == "numpy":
            hess = self._sobel_hessian_numpy(u)
        else:
            raise ValueError(f"[DiffOperator] Unsupported framework '{self.framework}' for Sobel Hessian.")

        # Return tagged output
        return self.to_output(
            hess,
            tag_as="output",
            framework=output_format or self.output_format,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )        
        
    def scharr(
        self,
        u: ArrayLike,
        enable_uid: bool = False,
        op_params: dict | None = None,
        framework: Framework | None = None,
        output_format: Framework | None = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: bool | None = None,
    ) -> ArrayLike:
        """
        ND Scharr gradient (Torch or NumPy).
        """
        from skimage.filters import scharr
        
        fw = framework or self.framework
        
        # === Convert input ===
        u = self.convert_once(
            image=u,
            tag_as="input",
            framework=fw,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )
        
        axes = self._get_axes(u)

        if fw == "torch":
            grad = []
            for axis in axes:
                g = torch.from_numpy(
                    scharr(u.detach().cpu().numpy(), axis=axis, mode='reflect')
                ).to(u.device)
                grad.append(g)
            output = torch.stack(grad, dim=0)
        
        elif fw == "numpy":
            grad = [scharr(u, axis=axis, mode='reflect') for axis in axes]
            output = np.stack(grad, axis=0)
        
        else:
            raise ValueError(f"Scharr not supported for framework {self.framework}")
        
        tagger = self.track(u)
        
        # ====[ Layout Handling ]=====
        layout_name, axes_tags = resolve_and_clean_layout_tags(tagger, self.framework, self.layout_name, 
                                                                prefix="G", remove_prefix=False,)

        result = tagger.stack_from(grad, axis=0, update_tags={
                                                            "status": "gradient",
                                                            "layout_name": layout_name,
                                                            "shape_after": output.shape,
                                                            **axes_tags
                                                            }).get()
        return self.to_output(
            result,
            tag_as="output",
            framework=output_format or self.output_format,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )
    
    def gradient_directional(
        self,
        u: ArrayLike,
        angle: float = 0.0,
        enable_uid: bool = False,
        op_params: dict | None = None,
        framework: Framework | None = None,
        output_format: Framework | None = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: bool | None = None,
    ) -> ArrayLike:
        """
        Compute directional derivative along a given angle (2D only for now).

        Parameters
        ----------
        u : torch.Tensor | np.ndarray
            2D input image.
        angle : float
            Angle in radians (0 = x-axis, pi/2 = y-axis).

        Returns
        -------
        grad : torch.Tensor | np.ndarray
            Directional gradient.
        """
        dims = len(self._get_axes(u))
        
        if dims != 2:
            raise NotImplementedError("Directional gradient only implemented for 2D images.")

        fw = framework or self.framework

        grad = self.gradient(u, output_format= self.framework,)
            
        gx, gy = grad[0], grad[1]

        if fw == "torch":
            cos_a = torch.cos(torch.tensor(angle, device=u.device))
            sin_a = torch.sin(torch.tensor(angle, device=u.device))
            dir_grad = cos_a * gx + sin_a * gy
            result = dir_grad

        elif fw == "numpy":
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            dir_grad = cos_a * gx + sin_a * gy
            result = dir_grad

        else:
            raise ValueError(f"Directional gradient not supported for framework {self.framework}")
        
        tagger = self.track(u)
        tracker = tagger.copy_to(result)
        tracker.update_tags({
            "status": "directional_gradient",
            "framework": fw,
            "shape_after": result.shape,
        })
    
        return self.to_output(
            tracker.get(),
            tag_as="output",
            framework=output_format or self.output_format,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )
    
    def gradient_total_variation(
        self,
        u: ArrayLike,
        enable_uid: bool = False,
        op_params: dict | None = None,
        framework: Framework | None = None,
        output_format: Framework | None = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: bool | None = None,
    ) -> ArrayLike:
        """
        Compute total variation gradient approximation (ND).

        Parameters
        ----------
        u : torch.Tensor | np.ndarray
            Input ND image.

        Returns
        -------
        tv : torch.Tensor | np.ndarray
            Approximate total variation per pixel.
        """
        fw = framework or self.framework
        
        grad = self.gradient(u, output_format=fw,)

        if self.framework == "torch":
            tv = torch.sqrt(torch.sum(grad ** 2, dim=0) + 1e-8)
        elif self.framework == "numpy":
            tv = np.sqrt(np.sum(grad ** 2, axis=0) + 1e-8)
        else:
            raise ValueError(f"TV gradient not supported for framework {self.framework}")

        tagger = self.track(u)
        tracker = tagger.copy_to(tv)
        tracker.update_tags({
            "status": "tv",
            "framework": fw,
            "shape_after": tv.shape,
        })
    
        return self.to_output(
            tracker.get(),
            tag_as="output",
            framework=output_format or self.output_format,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override
        )

# ==================================================
# ================= Internal Logic =================
# ==================================================

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

        def to_positive(axis):
            return axis if axis is None or axis >= 0 else axis + ndim

        channel_ax = to_positive(tag.get("channel_axis", self.axes.get("channel_axis")))
        batch_ax = to_positive(tag.get("batch_axis", self.axes.get("batch_axis")))
        direction_ax = to_positive(tag.get("direction_axis", self.axes.get("direction_axis")))

        # Remove non-spatial axes
        if channel_ax is not None and not self.include_channel_gradient and channel_ax in axes:
            axes.remove(channel_ax)
        if batch_ax is not None and batch_ax in axes:
            axes.remove(batch_ax)
        if direction_ax is not None and direction_ax in axes:
            axes.remove(direction_ax)

        if self.verbose:
            print(f"[DiffOperator] Spatial axes selected: {axes}")

        return axes

    def _get_spacing(self, n: int) -> List[float]:
        """
        Retrieve the spacing for each spatial dimension.

        Parameters
        ----------
        n : int
            The number of spatial dimensions for which spacing is required.

        Returns
        -------
        List[float]
            A list of spacing values, one per spatial dimension.
            If self.spacing is None, defaults to 1.0 per dimension.
        
        Raises
        ------
        ValueError
            If the provided spacing does not have the expected length.
        """
        # Default spacing: 1.0 for each dimension
        if self.spacing is None:
            return [1.0] * n

        spacing = self.spacing

        # If spacing is given as a single numerical value
        if isinstance(spacing, (int, float)):
            return [float(spacing)] * n

        # Ensure spacing is in array form, type float32
        spacing = np.atleast_1d(spacing).astype(np.float32, copy=False)

        # If spacing is a single value in an array, replicate it
        if spacing.size == 1:
            return [float(spacing[0])] * n

        # Check if the spacing length matches the expected dimensions
        if spacing.size != n:
            raise ValueError(f"Expected spacing of length {n}, got {spacing.size}")

        return spacing.tolist()
    
    @torch.no_grad()
    def _get_sobel_kernels_torch(self, u: torch.Tensor, ndim: int, device: str):
        """
        Generate Sobel kernels for each spatial axis (Torch tensors).
        """
        if ndim == 2:
            kx = torch.tensor([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]], dtype=torch.float32, device=device)
            
            ky = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=device)

            result = [kx, ky]

        elif ndim == 3:
            base = torch.tensor([1, 2, 1], dtype=torch.float32, device=device)
            diff = torch.tensor([-1, 0, 1], dtype=torch.float32, device=device)

            kx = torch.einsum('i,j,k->ijk', diff, base, base)
            ky = torch.einsum('i,j,k->ijk', base, diff, base)
            kz = torch.einsum('i,j,k->ijk', base, base, diff)
            result = [kx, ky, kz]

        else:
            raise NotImplementedError(f"[DiffOperator] Sobel only for 2D or 3D (got {ndim}D)")
        
        layout_name = "HW" if ndim == 2 else "DHW"
        axes_tags = get_layout_axes(self.framework, layout_name.upper())
        axes_tags.pop("name", None)
        axes_tags.pop("description", None)
        
        tagger = self.track(u)
        
        output = []
        
        for res in result:
            tracker = tagger.copy_to(res)
            tracker.update_tags({
                "status": "Kernel",
                "layout_name": layout_name,
                "shape_after": res.shape,
                **axes_tags
            })
            output.append(tracker.get())
            
        return output
    
    def finite_difference(self, u: ArrayLike, axis: int, h: float = 1.0, mode: str | None = None, boundary: str | None = None) -> ArrayLike:
        """
        Finite difference operator (forward/backward/centered) with boundary handling.
        Preserves input shape. Torch and NumPy compatible.
        """
        mode = mode or self.diff_mode.get("default")
        boundary = boundary or self.boundary_mode.get("default")

        is_torch = isinstance(u, torch.Tensor)
        lib = torch if is_torch else np

        roll_neg = {"shifts": -1, "dims": axis} if is_torch else {"shift": -1, "axis": axis}
        roll_pos = {"shifts": 1, "dims": axis} if is_torch else {"shift": 1, "axis": axis}

        if mode == "centered":
            u_plus = lib.roll(u, **roll_neg)
            u_minus = lib.roll(u, **roll_pos)
            return (u_plus - u_minus) / (2 * h)

        if mode == "forward":
            diff = (lib.roll(u, **roll_neg) - u) / h
            edge_idx = [slice(None)] * u.ndim
            ref_idx = [slice(None)] * u.ndim
            edge_idx[axis] = slice(-1, None)
            ref_idx[axis] = slice(-2, -1)

        elif mode == "backward":
            diff = (u - lib.roll(u, **roll_pos)) / h
            edge_idx = [slice(None)] * u.ndim
            ref_idx = [slice(None)] * u.ndim
            ref_idx[axis] = slice(1, 2)
            edge_idx[axis] = slice(0, 1)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if boundary == "periodic":
            return diff
        elif boundary == "neumann":
            diff[tuple(edge_idx)] = diff[tuple(ref_idx)].clone() if is_torch else diff[tuple(ref_idx)].copy()
        elif boundary == "dirichlet":
            diff[tuple(edge_idx)] = 0.0
        else:
            raise ValueError(f"Unsupported boundary: {boundary}")

        return diff
    
    def safe_l2_norm(self, u: ArrayLike, axis: int | None = None, eps: float = 1e-8, max_val: float = 1e10, clip_val: float = 1e5) -> ArrayLike:
        """
        Compute the L2 norm of a tensor or array, with clipping and safety checks.
        """
        if self.framework == "torch":
            return self.safe_l2_norm_torch(u, axis=axis, eps=eps, max_val=max_val, clip_val=clip_val)
        elif self.framework == "numpy":
            return self.safe_l2_norm_np(u, axis=axis, eps=eps, max_val=max_val, clip_val=clip_val)
    
    @staticmethod
    def safe_l2_norm_np(u: np.ndarray, axis: int | None = None, eps: float = 1e-8, max_val: float = 1e6, clip_val: float = 1e5) -> np.ndarray:
        u_clipped = np.clip(u, -clip_val, clip_val)
        norm_sq = np.sum(u_clipped** 2, axis=axis)
        norm_sq = np.clip(norm_sq, eps, max_val)
        return np.sqrt(norm_sq)
    
    @staticmethod
    def safe_l2_norm_torch(u: torch.Tensor, axis: int | None = None, eps: float = 1e-8, max_val: float = 1e6, clip_val: float = 1e5) -> torch.Tensor:
        u_clipped = torch.clamp(u, min=-clip_val, max=clip_val)
        norm_sq = torch.sum(u_clipped** 2, dim=axis)
        norm_sq = torch.clamp(norm_sq, min=eps, max=max_val)
        return torch.sqrt(norm_sq)    

# ==================================================
# ================= Torch Backend ==================
# ==================================================
    @torch.no_grad()
    def _gradient_torch(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute the ND gradient of a torch tensor using finite differences.
        Applies correct boundary condition and ND axis handling.

        Parameters
        ----------
        u : torch.Tensor
            ND tensor, tracked or raw.

        Returns
        -------
        grad : torch.Tensor
            ND gradient, shape = [D, ...]
        """
        axes = self._get_axes(u)
        spacing = self._get_spacing(len(axes))
        grads = []

        for i, axis in enumerate(axes):
            h = spacing[i]
            diff = self.finite_difference(u, axis, h=h, mode=self.diff_mode.get("gradient"), 
                                          boundary=self.boundary_mode.get("gradient"))
            grads.append(diff)

        # Stack gradients and propagate tags
        tagger = self.track(u)
        output = torch.stack(grads, dim=0)

        # ====[ Layout Handling ]=====
        layout_name, axes_tags = resolve_and_clean_layout_tags(tagger, self.framework, self.layout_name, 
                                                                prefix="G", remove_prefix=False,)

        return tagger.stack_from(grads, axis=0, update_tags={
            "status": "gradient",
            "layout_name": layout_name,
            "shape_after": output.shape,
            **axes_tags
        }).get()

    @torch.no_grad()
    def _divergence_torch(self, v: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:

        """
        Compute the ND divergence of a torch vector field (D, ...).
        Handles boundary conditions and spacing.

        Parameters
        ----------
        v : torch.Tensor
            Stacked vector field, where axis 0 = direction_axis.
            
        weight : torch.Tensor | None
            Optional weights for each direction.

        Returns
        -------
        div : torch.Tensor
            ND scalar field.
        """
        tagger = self.track(v)
        axes = self._get_axes(v)
        spacing = self._get_spacing(len(axes))
        div = torch.zeros_like(v[0])
        
        if weight is not None:
            v = v * weight
            if getattr(self, "max_flux", None) is not None:
                v = torch.clamp(v, -self.max_flux, self.max_flux)

        for i, axis in enumerate(axes):
            h = spacing[i]
            vi = v[i]
            diff = self.finite_difference(vi, axis-1, h=h, mode=self.diff_mode.get("divergence"), 
                                          boundary=self.boundary_mode.get("divergence"))
            div += diff

        # === [ Layout Handling ] ===
        layout_name, axes_tags = resolve_and_clean_layout_tags(tagger, self.framework, self.layout_name, 
                                                               prefix="G", remove_prefix=True)

        tracker = tagger.copy_to(div)
        tracker.update_tags({
            "status": "divergence",
            "layout_name": layout_name,
            "shape_after": div.shape,
            **axes_tags
        })
        return tracker.get()
    
    @torch.no_grad()
    def _sobel_torch(self, u: torch.Tensor, gradient: bool = False, normalize: bool = True, scale_factor: float | None = 15
    ) -> torch.Tensor:  
        """
        Compute Sobel gradient magnitude (Torch backend).
        """
        from operators.gaussian import NDConvolver as convolver
        
        dims = len(self._get_axes(u))
        sobel_kernels = self._get_sobel_kernels_torch(u, dims, u.device)
        grads = []
        
        # ====[ Create Convolver ]====
        convolve = convolver(
        ndconvolver_cfg = self.conv_cfg,
        layout_cfg = self.layout_cfg,
        global_cfg = self.global_cfg.update_config(output_format=self.framework), #same as framework
        )

        for kernel in sobel_kernels:
            g = convolve(u, kernel) 
            grads.append(g.squeeze())

        grad_mag = torch.sqrt(sum(g**2 for g in grads) + 1e-8)
        
        if normalize and scale_factor is not None:
            grad_mag = grad_mag / grad_mag.max() * scale_factor
            grads = [g/grad_mag.max() * (scale_factor*4) for g in grads]
        
        # === Tagging ND auto via layout ===
        tagger = self.track(u)
        
        if not gradient:
            tracker = tagger.clone_to(grad_mag, updates = 
                                  {"status": "magnitude", 
                                   "shape_after": grad_mag.shape})
            return tracker.get()        

        output = torch.stack(grads, dim=0)
        
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=False
        )

        return tagger.stack_from(grads, axis=0, update_tags={
            "status": "gradient",
            "layout_name": layout_name,
            "shape_after": output.shape,
            **axes_tags
        }).get()
        
    @torch.no_grad()
    def _hessian_torch(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute ND Hessian using Torch finite differences.
        Returns a Hessian tensor of shape [D, D, ...].
        """
        tagger = self.track(u)
        axes = self._get_axes(u)
        spacing = self._get_spacing(len(axes))
        D = len(axes)

        hess = [[None for _ in range(D)] for _ in range(D)]

        for i in range(D):
            for j in range(D):
                h = spacing[j]
                first_diff = self.finite_difference(u, axis=axes[j], h=h,
                                                    mode=self.diff_mode.get("gradient"),
                                                    boundary=self.boundary_mode.get("gradient"))
                second_diff = self.finite_difference(first_diff, axis=axes[i], h=spacing[i],
                                                    mode=self.diff_mode.get("gradient"),
                                                    boundary=self.boundary_mode.get("gradient"))
                hess[i][j] = second_diff

        output = torch.stack([torch.stack(row, dim=0) for row in hess], dim=0)

        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=False
        )

        tracker = tagger.copy_to(output)
        tracker.update_tags({
            "status": "hessian",
            "layout_name": layout_name,
            "shape_after": output.shape,
            **axes_tags
        })
        
        tracker.update_tag("direction_axis", (0, 1))

        return tracker.get()
    
    @torch.no_grad()
    def _sobel_hessian_torch(self, u: torch.Tensor, normalize: bool = True, scale_factor: float | None = 15) -> torch.Tensor:

        """
        Compute the ND Sobel Hessian matrix (second derivatives) using torch backend and NDConvolver.
        Output shape: [D, D, ...]
        """
        tagger = self.track(u)
        device = u.device
        axes = self._get_axes(u)
        D = len(axes)

        # === Get first-order Sobel kernels ===
        sobel_kernels = self._get_sobel_kernels_torch(u, D, device)  # list[D] of shape [1, 1, ...]
        
        # === Create NDConvolver instance ===
        from operators.gaussian import NDConvolver as convolver
        convolve = convolver(
            ndconvolver_cfg=self.conv_cfg,
            layout_cfg=self.layout_cfg,
            global_cfg=self.global_cfg.update_config(output_format=self.framework),
        )

        hess = [[None for _ in range(D)] for _ in range(D)]

        # === Compute second derivatives ===
        for i in range(D):
            grad_i = convolve(u, sobel_kernels[i])  # First derivative
            for j in range(D):
                grad_ij = convolve(grad_i, sobel_kernels[j])  # Second derivative
                hess[i][j] = grad_ij.squeeze()

        # === Normalize if needed ===
        if normalize and scale_factor is not None:
            max_val = max(h.abs().max().item() for row in hess for h in row) + 1e-8
            hess = [[(h / max_val) * scale_factor for h in row] for row in hess]

        # === Stack: first along axis 1 (columns), then 0 (rows)
        hess_j = [torch.stack(row, dim=0) for row in hess]
        hess_final = torch.stack(hess_j, dim=0)  # shape: [D, D, ...]

        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=False
        )

        tracker = tagger.copy_to(hess_final)
        tracker.update_tags({
            "status": "hessian",
            "layout_name": layout_name,
            "shape_after": hess_final.shape,
            **axes_tags
        })
        
        tracker.update_tag("direction_axis", (0, 1))  

        return tracker.get()    

# ==================================================
# ================= NumPy Backend ==================
# ==================================================
    def _gradient_numpy(self, u: np.ndarray) -> np.ndarray:
        """
        Compute the ND gradient using NumPy finite differences.
        Supports spacing and boundary conditions. Fully ND-aware.

        Parameters
        ----------
        u : np.ndarray
            ND array to differentiate.

        Returns
        -------
        grad : np.ndarray
            Stacked gradient array with shape [D, ...]
        """
        tagger = self.track(u)
        u = tagger.apply_to_all(np.asarray, dtype=np.float32).get() # Convert to float32 safely (across all slices)
        axes = self._get_axes(u)
        spacing = self._get_spacing(len(axes))
        grads = []

        for i, axis in enumerate(axes):
            h = spacing[i]
            diff = self.finite_difference(u, axis, h=h, mode=self.diff_mode.get("gradient"), 
                                          boundary=self.boundary_mode.get("gradient"))
            grads.append(diff)

        output = np.stack(grads, axis=0)

        # === Tagging ND auto via layout ===
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=False
        )

        return tagger.stack_from(grads, axis=0, update_tags={
            "status": "gradient",
            "layout_name": layout_name,
            "shape_after": output.shape,
            **axes_tags
        }).get()

    def _divergence_numpy(self, v: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:

        """
        Compute the ND divergence using NumPy finite differences.
        Input must be a stacked vector field with shape [D, ...].

        Parameters
        ----------
        v : np.ndarray
            ND vector field with direction_axis = 0.
            
        weight : np.ndarray | None
            ND weight array with shape [D, ...].

        Returns
        -------
        div : np.ndarray
            Scalar ND divergence output.
        """
        tagger = self.track(v)
        v = tagger.apply_to_all(np.asarray, dtype=np.float32).get() # Ensure float32 (across all slices)
        axes = self._get_axes(v)
        spacing = self._get_spacing(len(axes))
        div = np.zeros_like(v[0], dtype=np.float32)
        
        if weight is not None:
            v = v * weight
            if getattr(self, "max_flux", None) is not None:
                v = np.clip(v, -self.max_flux, self.max_flux)

        for i, axis in enumerate(axes):
            h = spacing[i]
            vi = v[i]
            diff = self.finite_difference(vi, axis-1, h=h, mode=self.diff_mode.get("divergence"), 
                                          boundary=self.boundary_mode.get("divergence"))
            div += diff
            
        # === Layout tag restoration ===
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=True
        )

        tracker = tagger.copy_to(div)
        tracker.update_tags({
            "status": "divergence",
            "layout_name": layout_name,
            "shape_after": div.shape,
            **axes_tags
        })
        return tracker.get()

    def _sobel_numpy(self, u: np.ndarray, gradient: bool = False) -> np.ndarray:
        """
        Compute Sobel gradient magnitude (NumPy backend).
        """
        # === Tagging ND auto via layout ===
        tagger = self.track(u)
        
        axes = self._get_axes(u)

        grads = [sobel(u, axis=axis, mode="reflect") for axis in axes]
        
        if not gradient:
            grad_mag = np.sqrt(sum(g**2 for g in grads) + 1e-8)
            tracker = tagger.clone_to(grad_mag, updates = 
                                  {"status": "magnitude", 
                                   "shape_after": grad_mag.shape})
            return tracker.get()

        output = np.stack(grads, axis=0)
        
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=False
        )

        return tagger.stack_from(grads, axis=0, update_tags={
            "status": "gradient",
            "layout_name": layout_name,
            "shape_after": output.shape,
            **axes_tags
        }).get()
        
    def _hessian_numpy(self, u: np.ndarray) -> np.ndarray:
        
        """
        Compute ND Hessian using NumPy finite differences.
        Returns a Hessian tensor of shape [D, D, ...].
        """
        
        tagger = self.track(u)
        u = tagger.apply_to_all(np.asarray, dtype=np.float32).get()
        axes = self._get_axes(u)
        spacing = self._get_spacing(len(axes))
        D = len(axes)

        hess = [[None for _ in range(D)] for _ in range(D)]

        for i in range(D):
            for j in range(D):
                h = spacing[j]
                first_diff = self.finite_difference(u, axis=axes[j], h=h,
                                                    mode=self.diff_mode.get("gradient"),
                                                    boundary=self.boundary_mode.get("gradient"))
                second_diff = self.finite_difference(first_diff, axis=axes[i], h=spacing[i],
                                                    mode=self.diff_mode.get("gradient"),
                                                    boundary=self.boundary_mode.get("gradient"))
                hess[i][j] = second_diff

        output = np.stack([np.stack(row, axis=0) for row in hess], axis=0)

        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=False
        )

        tracker = tagger.copy_to(output)
        tracker.update_tags({
            "status": "hessian",
            "layout_name": layout_name,
            "shape_after": output.shape,
            **axes_tags
        })
        
        tracker.update_tag("direction_axis", (0, 1))        

        return tracker.get()
    
    def _sobel_hessian_numpy(self, u: np.ndarray) -> np.ndarray:
        """
        Compute ND Hessian using Sobel second-order derivatives in NumPy.
        """
        tagger = self.track(u)
        axes = self._get_axes(u)
        D = len(axes)

        hess = [[None for _ in range(D)] for _ in range(D)]

        for i in range(D):
            for j in range(D):
                g = sobel(u, axis=axes[j], mode="reflect")
                hess_ij = sobel(g, axis=axes[i], mode="reflect")
                hess[i][j] = hess_ij

        # Stack (j) then (i) → shape (D, D, ...)
        hess_j = [np.stack(row, axis=0) for row in hess]
        hess_final = np.stack(hess_j, axis=0)

        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=False
        )

        tracker = tagger.copy_to(hess_final)
        tracker.update_tags({
            "status": "hessian",
            "layout_name": layout_name,
            "shape_after": hess_final.shape,
            **axes_tags
        })
        
        tracker.update_tag("direction_axis", (0, 1))  

        return tracker.get()

# ==================================================
# ================= Parallel Mode ==================
# ==================================================

    def _gradient_numpy_parallel(self, u: np.ndarray) -> np.ndarray:
        """
        Compute ND gradient using NumPy + joblib parallelism.
        Supports spacing and boundary handling.

        Parameters
        ----------
        u : np.ndarray
            ND input image.

        Returns
        -------
        grad : np.ndarray
            Gradient stacked on axis 0.
        """
        tagger = self.track(u)
        u = tagger.apply_to_all(np.asarray, dtype=np.float32).get()
        axes = self._get_axes(u)
        spacing = self._get_spacing(len(axes))
        
        grads = Parallel(n_jobs=-1, backend=self.backend)(
            delayed(self.finite_difference)(u, axis, h=h, mode=self.diff_mode.get("gradient"), boundary=self.boundary_mode.get("gradient")) for axis, h in zip(axes, spacing)
        )

        output = np.stack(grads, axis=0)

        # === Tagging ND auto via layout ===
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=False
        )

        return tagger.stack_from(grads, axis=0, update_tags={
            "status": "gradient",
            "layout_name": layout_name,
            "shape_after": output.shape,
            **axes_tags
        }).get()

    def _divergence_numpy_parallel(self, v: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:

        """
        Compute ND divergence using NumPy + joblib parallelism.
        Vector field must be stacked on axis 0.

        Parameters
        ----------
        v : np.ndarray
            Vector field with shape [D, ...].
            
        weight : np.ndarray | None
            Optional weights for each direction.

        Returns
        -------
        div : np.ndarray
            Scalar field after divergence.
        """
        tagger = self.track(v)
        v = tagger.apply_to_all(np.asarray, dtype=np.float32).get()
        axes = self._get_axes(v)
        spacing = self._get_spacing(len(axes))
        
        if weight is not None:
            v = v * weight
            if getattr(self, "max_flux", None) is not None:
                v = np.clip(v, -self.max_flux, self.max_flux)
                
        diffs = Parallel(n_jobs=-1, backend=self.backend)(
            delayed(self.finite_difference)(v[i], axis-1, h=h, mode=self.diff_mode.get("divergence"), 
                                            boundary=self.boundary_mode.get("divergence")) for i, (axis, h) in enumerate(zip(axes, spacing))
        )

        div = np.sum(diffs, axis=0)

        # === Layout ND : restoration ===
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=True
        )

        tracker = tagger.copy_to(div)
        tracker.update_tags({
            "status": "divergence",
            "layout_name": layout_name,
            "shape_after": div.shape,
            **axes_tags
        })
        return tracker.get()
    
    def _hessian_numpy_parallel(self, u: np.ndarray) -> np.ndarray:
        """
        Compute ND Hessian using NumPy + joblib parallelism.
        Returns a Hessian tensor of shape [D, D, ...].
        """
        tagger = self.track(u)
        u = tagger.apply_to_all(np.asarray, dtype=np.float32).get()
        axes = self._get_axes(u)
        spacing = self._get_spacing(len(axes))
        D = len(axes)

        def second_derivative(i, j):
            first_diff = self.finite_difference(u, axes[j], h=spacing[j],
                                                mode=self.diff_mode.get("gradient"),
                                                boundary=self.boundary_mode.get("gradient"))
            second_diff = self.finite_difference(first_diff, axes[i], h=spacing[i],
                                                mode=self.diff_mode.get("gradient"),
                                                boundary=self.boundary_mode.get("gradient"))
            return (i, j, second_diff)

        results = Parallel(n_jobs=-1, backend=self.backend)(
            delayed(second_derivative)(i, j) for i in range(D) for j in range(D)
        )

        hess = [[None for _ in range(D)] for _ in range(D)]
        for i, j, val in results:
            hess[i][j] = val

        hess_j = [np.stack(row, axis=0) for row in hess]
        hess_final = np.stack(hess_j, axis=0)

        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=False
        )

        tracker = tagger.copy_to(hess_final)
        tracker.update_tags({
            "status": "hessian",
            "layout_name": layout_name,
            "shape_after": hess_final.shape,
            **axes_tags
        })
        
        tracker.update_tag("direction_axis", (0, 1))        

        return tracker.get()

# ==================================================
# ================= Classic Fallback ===============
# ==================================================

    def _gradient_numpy_classic(self, u: np.ndarray) -> np.ndarray:
        """
        Compute the ND gradient using classic slicing (non-vectorized).
        Supports spacing and boundary conditions.

        Parameters
        ----------
        u : np.ndarray
            ND input image.

        Returns
        -------
        grad : np.ndarray
            Stacked gradients with shape [D, ...].
        """
        tagger = self.track(u)
        u = tagger.apply_to_all(np.asarray, dtype=np.float32).get()
        ndim = u.ndim
        axes = self._get_axes(u)
        spacing = self._get_spacing(len(axes))
        grads = []

        for i, axis in enumerate(axes): 
            diff = self.finite_difference(u, axis, h=spacing[i], mode=self.diff_mode.get("gradient"), 
                                          boundary=self.boundary_mode.get("gradient")) 
            grads.append(diff)

        output = np.stack(grads, axis=0)

        # === ND layout tagging ===
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=False
        )

        return tagger.stack_from(grads, axis=0, update_tags={
            "status": "gradient",
            "layout_name": layout_name,
            "shape_after": output.shape,
            **axes_tags
        }).get()


    def _divergence_numpy_classic(self, v: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
        """
        Compute ND divergence using classic slicing (non-vectorized).

        Parameters
        ----------
        v : np.ndarray
            Vector field of shape [D, ...] (direction_axis=0).
        
        weight : np.ndarray | None
            Optional weights for each direction.

        Returns
        -------
        div : np.ndarray
            Scalar field after divergence.
        
        """
        tagger = self.track(v)
        v = tagger.apply_to_all(np.asarray, dtype=np.float32).get()
        axes = self._get_axes(v)
        spacing = self._get_spacing(len(axes))
        div = np.zeros_like(v[0], dtype=np.float32)
        
        if weight is not None:
            v = v * weight
            if getattr(self, "max_flux", None) is not None:
                v = np.clip(v, -self.max_flux, self.max_flux)            

        for i, axis in enumerate(axes):
            vi, h = v[i], spacing[i]
            diff = self.finite_difference(vi, axis-1, h=h, mode=self.diff_mode.get("divergence"), boundary=self.boundary_mode.get("divergence"))
            div += diff

        # === ND layout tagging (remove G)
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=True
        )

        tracker = tagger.copy_to(div)
        tracker.update_tags({
            "status": "divergence",
            "layout_name": layout_name,
            "shape_after": div.shape,
            **axes_tags
        })
        return tracker.get()
    
    def _hessian_numpy_classic(self, u: np.ndarray) -> np.ndarray:
        """
        Compute ND Hessian using nested loops.
        Fully ND-aware, independent of gradient().

        Returns a Hessian tensor of shape [D, D, ...].
        """
        tagger = self.track(u)
        u = tagger.apply_to_all(np.asarray, dtype=np.float32).get()
        axes = self._get_axes(u)
        spacing = self._get_spacing(len(axes))
        D = len(axes)

        hess = [[None for _ in range(D)] for _ in range(D)]

        for i in range(D):
            for j in range(D):
                h = spacing[j]
                first_diff = self.finite_difference(u, axis=axes[j], h=h,
                                                    mode=self.diff_mode.get("gradient"),
                                                    boundary=self.boundary_mode.get("gradient"))
                second_diff = self.finite_difference(first_diff, axis=axes[i], h=spacing[i],
                                                    mode=self.diff_mode.get("gradient"),
                                                    boundary=self.boundary_mode.get("gradient"))
                hess[i][j] = second_diff

        hess_j = [np.stack(row, axis=0) for row in hess]
        hess_final = np.stack(hess_j, axis=0)

        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tagger, self.framework, self.layout_name, prefix="G", remove_prefix=False
        )

        tracker = tagger.copy_to(hess_final)
        tracker.update_tags({
            "status": "hessian",
            "layout_name": layout_name,
            "shape_after": hess_final.shape,
            **axes_tags
        })

        tracker.update_tag("direction_axis", (0, 1))

        return tracker.get()

    
# ======================================================================
#                      Convenience wrapper
# ======================================================================

def diffop(
        img: ArrayLike,
        func: str = "gradient",
        diff_strategy: str = "vectorized",
        framework: Framework = "numpy",
        output_format: Framework = "numpy",
        layout_name: str = "HWC",
        layout_framework: Framework = "numpy",
        backend: str = "sequential",
        ):
    """
    Convenience wrapper to build and run a DiffOperator instance.
    Parameters
    ----------
    img : ArrayLike
        Input image.
    func : str, optional
        Method to call on DiffOperator. Options: "gradient", "divergence", "    hessian", "sobel", "sobel_hessian".
        Default is "gradient".
    diff_strategy : str, optional
        Differentiation strategy. Options: "vectorized", "torch", "numpy". Default is "vectorized".
    framework : str, optional
        Framework to use. Options: "numpy", "torch". Default is "numpy".
    output_format : str, optional
        Output format. Options: "numpy", "torch". Default is "numpy".
    layout_name : str, optional
        Layout name. Default is "HWC".
    layout_framework : str, optional        
        Layout framework. Default is "numpy".
    backend : str, optional
        Backend to use. Options: "sequential", "parallel". Default is "sequential".
    """    
    # ====[ Fallback ]====
    diff_strategy=diff_strategy or "vectorized" if framework == "numpy" else "torch"
    
    # ====[ Configuration ]====
    diff_params: Dict[str, Any] = {"spacing": None, "diff_strategy":diff_strategy, "diff_fallback":True}
    layout_params: Dict[str, Any] = {"layout_name": layout_name, "layout_framework": layout_framework}
    global_params: Dict[str, Any] = {"framework": framework, "output_format": output_format, "backend": backend}
    
    diff = DiffOperator(
                        diff_operator_cfg=DiffOperatorConfig(**diff_params),
                        layout_cfg=LayoutConfig(**layout_params),
                        global_cfg=GlobalConfig(**global_params),
                        )
    
    if not hasattr(diff, func):
        raise ValueError(f"'{func}' is not a valid method of DiffOperator.")

    method = getattr(diff, func)
    img_copy = diff.safe_copy(img)    
    return method(img_copy)

