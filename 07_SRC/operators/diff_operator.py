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
    N-dimensional differential operator for computing gradients, divergences, and Laplacians.

    Supports multiple computation strategies (NumPy, Torch, Parallel) and
    handles layout-aware axis tagging, spacing, and backend consistency.

    Notes
    -----
    - Gradient, divergence, and Laplacian are computed via finite differences.
    - Axes are auto-detected or configured via `LayoutConfig`.
    - Compatible with both NumPy and Torch inputs.
    - Strategy and fallback behavior are defined in `DiffOperatorConfig`.
    """

    def __init__(
        self,
        diff_operator_cfg: DiffOperatorConfig = DiffOperatorConfig(),
        ndconvolver_cfg: NDConvolverConfig = NDConvolverConfig(),
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize the DiffOperator for ND differential computations.

        Parameters
        ----------
        diff_operator_cfg : DiffOperatorConfig
            Configuration for gradient, divergence, and Laplacian computation strategies,
            including difference mode, boundary conditions, and fallback handling.
        ndconvolver_cfg : NDConvolverConfig
            Optional configuration used when differential ops require convolution fallback.
        layout_cfg : LayoutConfig
            Axis layout configuration (e.g., spatial, batch, channel).
        global_cfg : GlobalConfig
            Framework preferences (NumPy or Torch), device, output format, and spacing behavior.
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
        """
        Print a message if verbose mode is enabled.

        Parameters
        ----------
        msg : str
            Message to print to stdout.
        """
        if self.verbose:
            print(msg)

    # ---------- Helpers: backend/strategy routing ----------

    def _strategy_expects(self, fw_current: str) -> str:
        """
        Return the backend expected by the selected computation strategy.

        Parameters
        ----------
        fw_current : str
            Current framework in use ('numpy' or 'torch').

        Returns
        -------
        str
            Expected backend for the configured strategy. Returns:
            - 'torch'   → if strategy is 'torch'
            - 'numpy'   → if strategy is 'parallel', 'classic', or 'vectorized'
            - fw_current → for strategy 'auto' (no override)
        """
        if self.strategy == "torch":
            return "torch"
        if self.strategy in ("parallel", "classic", "vectorized"):
            return "numpy"
        # auto → keep current
        return fw_current

    def _ensure_backend(self, x: ArrayLike, target_fw: str) -> ArrayLike:
        """
        Ensure that the input array uses the target computation backend.

        Converts between NumPy and Torch when necessary, preserving layout, tags,
        and status information for traceability.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input array or tensor to verify or convert.
        target_fw : {'numpy', 'torch'}
            Desired target backend for downstream operations.

        Returns
        -------
        np.ndarray or torch.Tensor
            Input converted to the target backend, or returned as-is if already compatible.

        Notes
        -----
        - Conversion is handled via `to_framework`, which applies proper tagging.
        - No conversion is performed if the input already matches the target framework.
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
        Compute the N-dimensional gradient of an image or tensor.

        Supports multiple computation strategies (Torch, NumPy, parallel, classic)
        with automatic backend adaptation and tag propagation.

        Parameters
        ----------
        u : np.ndarray or torch.Tensor
            Input image or tensor. May be 2D, 3D, or ND.
        enable_uid : bool, default False
            Whether to assign a unique ID in the tag.
        op_params : dict, optional
            Metadata dictionary to embed in the tag (e.g., {'method': 'gradient'}).
        framework : {'torch', 'numpy'}, optional
            Force the input backend. If None, inferred from input or config.
        output_format : {'torch', 'numpy'}, optional
            Format of the returned gradient. Defaults to `self.output_format`.
        track : bool, default True
            If True, preserve tagging and axis metadata in the output.
        trace_limit : int, default 10
            Max number of trace entries to keep in tag history.
        normalize_override : bool, optional
            Override global normalization behavior locally.
        to_return : Sequence[str], default ('gradient',)
            Choose which outputs to return. Valid values: 'gradient', 'magnitude', or both.

        Returns
        -------
        grad : ArrayLike
            The computed gradient (tensor with shape: (D, ...) where D = spatial dims).
        magnitude : ArrayLike, optional
            Gradient magnitude (L2 norm across spatial dimensions), if requested.

        Raises
        ------
        TypeError
            If the input format is unsupported or framework cannot be inferred.

        Notes
        -----
        - The computation strategy is controlled by `self.strategy` ('torch', 'parallel', 'classic', etc.).
        - Axes are automatically inferred from the layout config or tags.
        - Output tensors are tagged, and their status is updated to reflect the operation performed.
        - This method is ND-compatible and supports both CPU and GPU backends.
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
        Compute the N-dimensional divergence of a vector field.

        This is the divergence ∇·v where `v` is a vector field (first axis = direction),
        typically the output of `gradient()`. Supports optional per-dimension weighting.

        Parameters
        ----------
        v : np.ndarray or torch.Tensor
            Input vector field. Shape must be (D, ...) where D is the number of spatial axes.
        weight : np.ndarray or torch.Tensor, optional
            Optional weighting array (same shape as `v`) to modulate the divergence.
        enable_uid : bool, default False
            Whether to assign a UID to the output tag.
        op_params : dict, optional
            Metadata dictionary to embed in the output tag.
        framework : {'torch', 'numpy'}, optional
            Force backend. If None, inferred from input or configuration.
        output_format : {'torch', 'numpy'}, optional
            Output format. If None, uses `self.output_format`.
        track : bool, default True
            If True, maintain tag and axis tracking in the output.
        trace_limit : int, default 10
            Maximum length of the tag's operation history.
        normalize_override : bool, optional
            Override normalization behavior for this operation.

        Returns
        -------
        div : np.ndarray or torch.Tensor
            The computed divergence map, in the requested backend format.

        Raises
        ------
        TypeError
            If the input type is not supported or the backend cannot be inferred.

        Notes
        -----
        - The computation strategy is selected via `self.strategy`.
        - Weighting is applied per direction before summing if `weight` is provided.
        - Tagging and axis semantics are preserved and updated via `to_output()`.
        - This method supports both 2D and ND fields (3D, 4D, etc.).
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
        Compute the Laplacian of a scalar ND image using div(grad(u)).

        This operation computes the divergence of the gradient of the input,
        combining two finite-difference operators in sequence. Supports both
        NumPy and Torch backends and preserves tags and layout information.

        Parameters
        ----------
        u : np.ndarray or torch.Tensor
            Input scalar image (ND). Should be float and spatially structured.
        enable_uid : bool, default False
            Whether to assign a unique UID in the output tag.
        op_params : dict, optional
            Optional metadata to include in the output tag.
        framework : {'torch', 'numpy'}, optional
            Force computation in a specific backend. If None, auto-inferred.
        output_format : {'torch', 'numpy'}, optional
            Desired format of the output. If None, uses `self.output_format`.
        track : bool, default True
            Enable propagation of AxisTracker and tagging information.
        trace_limit : int, default 10
            Maximum number of operations to track in tag history.
        normalize_override : bool, optional
            Override global normalization setting locally.

        Returns
        -------
        laplacian : np.ndarray or torch.Tensor
            The Laplacian of `u`, in the desired backend and layout.

        Notes
        -----
        - Computed as ∇·(∇u) using sequential calls to `gradient()` and `divergence()`.
        - Axes and layout are inferred from tags or `LayoutConfig`.
        - Tagging is preserved throughout the operation pipeline.
        - Compatible with arbitrary dimensions (2D, 3D, ND).
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
        Compute the full N-dimensional Hessian matrix of a scalar field u.

        Returns the second-order derivatives of u with respect to all spatial
        directions, forming a symmetric matrix of shape (D, D, ...), where D
        is the number of spatial axes.

        Parameters
        ----------
        u : np.ndarray or torch.Tensor
            Input scalar image (ND), assumed to be float and differentiable.
        enable_uid : bool, default False
            Whether to assign a unique UID to the output tag.
        op_params : dict, optional
            Optional metadata to embed in the output tag.
        framework : {'torch', 'numpy'}, optional
            Force the backend used for computation. If None, inferred automatically.
        output_format : {'torch', 'numpy'}, optional
            Desired output backend for the result.
        track : bool, default True
            Whether to preserve and propagate AxisTracker metadata.
        trace_limit : int, default 10
            Limit the number of operations stored in the tag trace.
        normalize_override : bool, optional
            Locally override normalization behavior.

        Returns
        -------
        hess : np.ndarray or torch.Tensor
            The Hessian matrix of shape (D, D, ...) where D is the number of spatial axes.
            Each (i, j, ...) component corresponds to ∂²u / ∂x_i∂x_j.

        Notes
        -----
        - Strategy dispatch is handled based on `self.strategy` and input backend.
        - The input is converted and tagged before processing.
        - Output is passed through `to_output()` to preserve tags and format.
        - ND-compatible for 2D, 3D, and higher dimensions.
        - Layout is respected and spatial axes are inferred via tag or config.
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
        Compute the Sobel gradient magnitude of an N-dimensional image.

        Applies the Sobel operator along each spatial axis, computes directional
        gradients, and combines them into a single magnitude map (L2 norm).
        Automatically dispatches to the appropriate backend (Torch or NumPy),
        with optional fallback behavior.

        Parameters
        ----------
        u : np.ndarray or torch.Tensor
            Input image or volume. Can be 2D, 3D, or ND.
        enable_uid : bool, default False
            Assign a unique identifier in the output tag.
        op_params : dict, optional
            Optional metadata for tagging and trace history.
        framework : {'torch', 'numpy'}, optional
            Force a specific backend. If None, inferred automatically.
        output_format : {'torch', 'numpy'}, optional
            Desired output format.
        track : bool, default True
            Enable AxisTracker tagging and propagation.
        trace_limit : int, default 10
            Maximum tag history length.
        normalize_override : bool, optional
            Local override of normalization behavior.

        Returns
        -------
        sobel_mag : np.ndarray or torch.Tensor
            Magnitude of the Sobel gradient, with backend and format defined by output configuration.

        Notes
        -----
        - If the Torch backend is selected but unsupported, the operator falls back to NumPy.
        - The output is tagged and layout-aware, and preserves spatial shape.
        - ND versions of the Sobel operator apply directional filters per axis.
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
        Compute the N-dimensional Sobel gradient vector of an image or volume.

        Applies directional Sobel filters along each spatial axis and returns the
        full gradient tensor (shape: D × ...), where D is the number of spatial dimensions.

        Parameters
        ----------
        u : np.ndarray or torch.Tensor
            Input scalar image or volume. Can be 2D, 3D, or higher.
        enable_uid : bool, default False
            Assign a unique identifier to the output tag.
        op_params : dict, optional
            Additional metadata for tagging.
        framework : {'torch', 'numpy'}, optional
            Force backend to use. If None, inferred from input.
        output_format : {'torch', 'numpy'}, optional
            Format for the output. If None, uses `self.output_format`.
        track : bool, default True
            Enable AxisTracker propagation and tagging.
        trace_limit : int, default 10
            Max number of operations to store in the trace history.
        normalize_override : bool, optional
            Locally override the normalization behavior.

        Returns
        -------
        grad : np.ndarray or torch.Tensor
            Sobel gradient tensor of shape (D, ...) in the specified format.

        Notes
        -----
        - The gradient is computed using directional Sobel filters along each spatial axis.
        - If the Torch backend is selected but unsupported, fallback to NumPy is used.
        - The result is ND-compatible and layout-aware, preserving tags and axis roles.
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
        Compute the N-dimensional Hessian matrix using Sobel second derivatives.

        This method uses directional Sobel filters to approximate second-order
        partial derivatives, resulting in a full symmetric Hessian tensor.

        Parameters
        ----------
        u : np.ndarray or torch.Tensor
            Input scalar image or volume (ND).
        enable_uid : bool, default False
            Whether to assign a unique UID to the output tag.
        op_params : dict, optional
            Additional metadata to embed in the output tag.
        framework : {'torch', 'numpy'}, optional
            Force the backend to use. If None, inferred from input.
        output_format : {'torch', 'numpy'}, optional
            Desired format for the returned tensor.
        track : bool, default True
            Enable propagation of AxisTracker metadata.
        trace_limit : int, default 10
            Maximum number of trace operations stored.
        normalize_override : bool, optional
            Override normalization behavior locally.

        Returns
        -------
        hessian : np.ndarray or torch.Tensor
            ND Hessian matrix of shape (D, D, ...) where D is the number of spatial axes.

        Notes
        -----
        - Backend is selected dynamically or via parameter.
        - If `diff_fallback` is True and backend is Torch, fallback to NumPy is used.
        - Each entry H[i, j, ...] corresponds to the second-order derivative ∂²u/∂x_i∂x_j.
        - Output is tagged with layout and operation trace via `to_output()`.
        - Compatible with ND images: 2D, 3D, or higher.
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
        Compute the N-dimensional Scharr gradient of an image.

        Applies the Scharr operator along each spatial axis to estimate directional gradients.
        Produces a stacked tensor of shape (D, ...), where D is the number of spatial dimensions.

        Parameters
        ----------
        u : np.ndarray or torch.Tensor
            Input scalar image (2D, 3D, or ND).
        enable_uid : bool, default False
            Whether to assign a UID to the output tag.
        op_params : dict, optional
            Optional metadata to embed in the tag.
        framework : {'torch', 'numpy'}, optional
            Backend to use for computation. If None, inferred automatically.
        output_format : {'torch', 'numpy'}, optional
            Format of the returned result.
        track : bool, default True
            Whether to propagate tagging and layout tracking.
        trace_limit : int, default 10
            Maximum length of tag history.
        normalize_override : bool, optional
            Override normalization behavior locally.

        Returns
        -------
        gradient : np.ndarray or torch.Tensor
            Scharr gradient vector with shape (D, ...), tagged and formatted.

        Notes
        -----
        - Uses `skimage.filters.scharr` under the hood (NumPy-based).
        - For Torch inputs, the image is converted to NumPy for filtering,
        then returned to the original device.
        - Tagging and layout metadata are preserved via AxisTracker.
        - Especially useful for edge detection with higher accuracy than Sobel.
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
        Compute the directional gradient of a 2D image along a specified angle.

        Uses the dot product of the gradient vector (gx, gy) with a unit vector
        oriented along the provided angle to extract the derivative in that direction.

        Parameters
        ----------
        u : np.ndarray or torch.Tensor
            2D scalar image.
        angle : float, default 0.0
            Angle in radians specifying the direction of the derivative.
            - 0   → horizontal (x-axis),
            - π/2 → vertical (y-axis),
            - π/4 → diagonal.
        enable_uid : bool, default False
            Whether to assign a UID to the output.
        op_params : dict, optional
            Optional metadata to embed in the output tag.
        framework : {'torch', 'numpy'}, optional
            Backend to use. If None, inferred automatically.
        output_format : {'torch', 'numpy'}, optional
            Output format.
        track : bool, default True
            Whether to propagate tags and layout tracking.
        trace_limit : int, default 10
            Maximum tag history depth.
        normalize_override : bool, optional
            Override normalization behavior locally.

        Returns
        -------
        directional_gradient : np.ndarray or torch.Tensor
            Scalar directional derivative image, tagged and formatted.

        Raises
        ------
        NotImplementedError
            If the input is not 2D.

        Notes
        -----
        - Only implemented for 2D inputs.
        - Relies on standard gradient() followed by projection onto the given angle.
        - The angle must be expressed in **radians**.
        - Preserves all tagging and metadata via AxisTracker.
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
        Compute the total variation (TV) gradient approximation of an ND image.

        Computes the L2-norm of the spatial gradient vector at each pixel/voxel.
        This quantity reflects local image variations and is widely used in
        denoising and regularization tasks.

        Parameters
        ----------
        u : np.ndarray or torch.Tensor
            Input scalar image of arbitrary dimensions.
        enable_uid : bool, default False
            Whether to assign a UID to the output.
        op_params : dict, optional
            Optional metadata to include in output tag.
        framework : {'torch', 'numpy'}, optional
            Backend to use. If None, inferred automatically.
        output_format : {'torch', 'numpy'}, optional
            Desired output format.
        track : bool, default True
            Whether to propagate tags and layout tracking.
        trace_limit : int, default 10
            Maximum depth of tag history.
        normalize_override : bool, optional
            Override default normalization behavior.

        Returns
        -------
        tv : np.ndarray or torch.Tensor
            Per-pixel total variation (scalar map), tagged and formatted.

        Notes
        -----
        - Computes √(∑(∂u/∂xᵢ)² + ε) per voxel (ε=1e-8 for stability).
        - Relies internally on `gradient()` to compute spatial derivatives.
        - Layout tracking and UID propagation are preserved via AxisTracker.
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
        Determine spatial axes for applying differential operators.

        Parameters
        ----------
        arr : ArrayLike
            Input array (NumPy or Torch), typically an image or volume.

        Returns
        -------
        List[int]
            List of axis indices corresponding to spatial dimensions, excluding
            batch, channel, and direction axes unless explicitly allowed.
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
        Retrieve spacing values for each spatial dimension.

        Parameters
        ----------
        n : int
            Number of spatial dimensions (e.g., 2 for 2D, 3 for 3D).

        Returns
        -------
        List[float]
            Spacing values along each spatial axis. If `self.spacing` is None,
            a uniform spacing of 1.0 is returned for all dimensions.

        Raises
        ------
        ValueError
            If `self.spacing` is a list but its length does not match `n`.
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
        Generate Sobel kernels for each spatial axis in 2D or 3D (Torch version).

        Parameters
        ----------
        u : torch.Tensor
            Reference tensor used for tagging and layout propagation.
        ndim : int
            Number of spatial dimensions (must be 2 or 3).
        device : str
            Torch device to allocate the kernels on (e.g., "cpu", "cuda").

        Returns
        -------
        List[torch.Tensor]
            List of Sobel kernels, one per spatial axis, with proper tags.

        Raises
        ------
        NotImplementedError
            If `ndim` is not 2 or 3.
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
    
    def finite_difference(
        self,
        u: ArrayLike,
        axis: int,
        h: float = 1.0,
        mode: str | None = None,
        boundary: str | None = None
    ) -> ArrayLike:
        """
        Apply a finite difference operator along a given axis with boundary handling.

        Supports NumPy and PyTorch tensors. Preserves input shape.

        Parameters
        ----------
        u : ArrayLike
            Input array (NumPy or Torch tensor).
        axis : int
            Axis along which to compute the derivative.
        h : float, optional
            Spacing between points (default: 1.0).
        mode : {'forward', 'backward', 'centered'}, optional
            Type of finite difference to apply. If None, uses the default from config.
        boundary : {'neumann', 'dirichlet', 'periodic'}, optional
            Boundary condition strategy. If None, uses the default from config.

        Returns
        -------
        ArrayLike
            Array of the same shape as input `u`, containing finite differences.

        Raises
        ------
        ValueError
            If an unsupported `mode` or `boundary` is provided.
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
    
    def safe_l2_norm(
        self,
        u: ArrayLike,
        axis: int | None = None,
        eps: float = 1e-8,
        max_val: float = 1e10,
        clip_val: float = 1e5
    ) -> ArrayLike:
        """
        Compute the L2 norm with numerical safeguards and optional clipping.

        Dispatches to the appropriate backend (NumPy or Torch) based on configuration.

        Parameters
        ----------
        u : ArrayLike
            Input array or tensor.
        axis : int or None, optional
            Axis along which to compute the norm. If None, use the flattened input.
        eps : float, optional
            Small constant added to prevent division by zero (default: 1e-8).
        max_val : float, optional
            Maximum value before clipping is enforced.
        clip_val : float, optional
            Value to clip the norm to, if it exceeds `max_val`.

        Returns
        -------
        ArrayLike
            Safe L2 norm of the input, same backend and shape as expected.

        Raises
        ------
        NotImplementedError
            If the selected backend is not supported.
        """
        if self.framework == "torch":
            return self.safe_l2_norm_torch(u, axis=axis, eps=eps, max_val=max_val, clip_val=clip_val)
        elif self.framework == "numpy":
            return self.safe_l2_norm_np(u, axis=axis, eps=eps, max_val=max_val, clip_val=clip_val)
    
    @staticmethod
    def safe_l2_norm_np(
        u: np.ndarray,
        axis: int | None = None,
        eps: float = 1e-8,
        max_val: float = 1e6,
        clip_val: float = 1e5
    ) -> np.ndarray:
        """
        Compute a stable L2 norm of a NumPy array, with clipping and lower bound safeguards.

        Parameters
        ----------
        u : np.ndarray
            Input array.
        axis : int or None, optional
            Axis along which to compute the norm. If None, the norm is computed over the entire array.
        eps : float, optional
            Minimum squared norm value to avoid sqrt(0) (default: 1e-8).
        max_val : float, optional
            Maximum squared norm value before clamping (default: 1e6).
        clip_val : float, optional
            Absolute value used to clip the input tensor before norm computation (default: 1e5).

        Returns
        -------
        np.ndarray
            Stable L2 norm with the same shape as `u` minus the reduced axis.
        """
        u_clipped = np.clip(u, -clip_val, clip_val)
        norm_sq = np.sum(u_clipped** 2, axis=axis)
        norm_sq = np.clip(norm_sq, eps, max_val)
        return np.sqrt(norm_sq)
    
    @staticmethod
    def safe_l2_norm_torch(
        u: torch.Tensor,
        axis: int | None = None,
        eps: float = 1e-8,
        max_val: float = 1e6,
        clip_val: float = 1e5
    ) -> torch.Tensor:
        """
        Compute a stable L2 norm of a Torch tensor, with clipping and safety bounds.

        Parameters
        ----------
        u : torch.Tensor
            Input tensor.
        axis : int or None, optional
            Axis along which to compute the norm. If None, the norm is computed over all elements.
        eps : float, optional
            Minimum squared norm value to prevent sqrt(0) (default: 1e-8).
        max_val : float, optional
            Maximum squared norm value before clamping (default: 1e6).
        clip_val : float, optional
            Value used to clamp input values before norm computation (default: 1e5).

        Returns
        -------
        torch.Tensor
            Stable L2 norm with the same shape as `u` minus the reduced axis.
        """
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
        Compute the N-dimensional gradient of a Torch tensor using finite differences.

        Applies spacing-aware finite differences along spatial axes, with support
        for boundary conditions and layout-aware tagging.

        Parameters
        ----------
        u : torch.Tensor
            Input tensor, either raw or tracked with axis metadata.

        Returns
        -------
        torch.Tensor
            Stacked gradient tensor with shape [D, ...], where D is the number
            of spatial axes. Layout tags and metadata are preserved.
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
        Compute the N-dimensional divergence of a Torch vector field using finite differences.

        Supports spatial weighting, optional flux clipping, and layout-aware metadata tracking.
        Handles boundary conditions and spacing along each spatial axis.

        Parameters
        ----------
        v : torch.Tensor
            Input vector field, stacked along axis 0 (shape: [D, ...]), where D is the number
            of spatial directions.
        weight : torch.Tensor or None, optional
            Optional per-direction weighting tensor (same shape as `v`).

        Returns
        -------
        torch.Tensor
            Divergence scalar field with same shape as a single vector component.
            Includes updated layout and tag metadata.
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
    def _sobel_torch(
        self,
        u: torch.Tensor,
        gradient: bool = False,
        normalize: bool = True,
        scale_factor: float | None = 15
    ) -> torch.Tensor:
        """
        Compute the Sobel gradient magnitude or full gradient field of a Torch tensor.

        Applies dimension-aware Sobel filtering using convolution. Supports optional
        normalization and ND gradient output with tag propagation.

        Parameters
        ----------
        u : torch.Tensor
            Input image or volume (tracked or raw).
        gradient : bool, optional
            If True, return the full ND gradient as a stacked tensor [D, ...].
            If False (default), return the scalar gradient magnitude.
        normalize : bool, optional
            Whether to normalize the gradient magnitude (default: True).
        scale_factor : float or None, optional
            Scale to apply after normalization (default: 15). Ignored if normalize is False.

        Returns
        -------
        torch.Tensor
            Sobel-filtered output, either a scalar gradient magnitude or
            a stacked tensor of per-axis gradients with full tagging support.
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
        Compute the N-dimensional Hessian matrix of a Torch tensor using finite differences.

        Uses second-order partial derivatives computed via repeated finite difference
        operations along each pair of spatial axes.

        Parameters
        ----------
        u : torch.Tensor
            Input tensor (tracked or raw), representing a scalar field.

        Returns
        -------
        torch.Tensor
            Hessian tensor of shape [D, D, ...], where D is the number of spatial axes.
            Layout tags are updated accordingly, with direction axes set to (0, 1).

        Notes
        -----
        - Handles spacing and boundary conditions based on config.
        - Gradient mode is used for both partial derivatives.
        - Output preserves tagging and layout semantics via tracker propagation.
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
    def _sobel_hessian_torch(
        self,
        u: torch.Tensor,
        normalize: bool = True,
        scale_factor: float | None = 15
    ) -> torch.Tensor:
        """
        Compute the N-dimensional Hessian matrix using Sobel filters and Torch backend.

        Applies two successive Sobel convolutions along each pair of spatial axes
        to estimate second-order derivatives. Optionally normalizes the Hessian
        components to a fixed scale.

        Parameters
        ----------
        u : torch.Tensor
            Input tensor (2D or 3D), raw or tracked.
        normalize : bool, optional
            Whether to normalize the Hessian values (default: True).
        scale_factor : float or None, optional
            Target range for normalization (default: 15). Ignored if `normalize` is False.

        Returns
        -------
        torch.Tensor
            Hessian tensor of shape [D, D, ...], where D is the number of spatial axes.
            Includes full tagging and layout metadata.

        Notes
        -----
        - Uses Sobel kernels generated with `_get_sobel_kernels_torch`.
        - Convolution performed using `NDConvolver`.
        - Result is layout-aware and includes 'direction_axis': (0, 1).
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
        Compute the N-dimensional gradient of a NumPy array using finite differences.

        Applies spacing-aware finite differences along spatial axes, with support
        for boundary conditions and layout-aware tagging.

        Parameters
        ----------
        u : np.ndarray
            Input ND array representing a scalar field.

        Returns
        -------
        np.ndarray
            Stacked gradient array of shape [D, ...], where D is the number of spatial axes.
            Includes tag propagation with updated layout and metadata.

        Notes
        -----
        - Input is converted to float32 before computation for consistency.
        - Tag tracking is preserved and updated automatically.
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
        Compute the N-dimensional divergence of a NumPy vector field using finite differences.

        Uses per-axis gradients with optional spatial weighting and flux clipping.
        Handles boundary conditions and spacing along each spatial direction.

        Parameters
        ----------
        v : np.ndarray
            Input vector field with shape [D, ...], where D is the number of spatial axes.
            Axis 0 must correspond to the direction_axis.
        weight : np.ndarray or None, optional
            Optional weight array of the same shape as `v`.

        Returns
        -------
        np.ndarray
            Scalar divergence field with same spatial shape as a single vector component.
            Includes updated tags and layout metadata.

        Notes
        -----
        - Input is cast to float32 before processing.
        - Tagging and axis layout are preserved and updated automatically.
        - Uses config-defined spacing, finite difference mode, and boundary policy.
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
        Compute the N-dimensional Sobel gradient (NumPy backend).

        Applies 1D Sobel filters along each spatial axis and returns either the
        scalar gradient magnitude or the full ND gradient field. Preserves layout
        and tagging metadata.

        Parameters
        ----------
        u : np.ndarray
            Input ND image or volume (tracked or raw).
        gradient : bool, optional
            If True, returns the full stacked gradient field [D, ...].
            If False (default), returns scalar gradient magnitude.

        Returns
        -------
        np.ndarray
            Either a scalar gradient magnitude or a stacked gradient array with shape [D, ...],
            including layout-aware tags and metadata.

        Notes
        -----
        - Uses `scipy.ndimage.sobel` along each spatial axis.
        - Adds epsilon (1e-8) to prevent sqrt(0) during magnitude computation.
        - Tagging and layout handling are consistent with other ND methods.
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
        Compute the N-dimensional Hessian matrix using NumPy finite differences.

        Applies two successive partial derivatives along each pair of spatial axes
        using spacing-aware finite differences and boundary handling.

        Parameters
        ----------
        u : np.ndarray
            Input scalar field (2D or ND NumPy array, tracked or raw).

        Returns
        -------
        np.ndarray
            Hessian tensor of shape [D, D, ...], where D is the number of spatial axes.
            Includes updated tags and layout metadata with 'direction_axis': (0, 1).

        Notes
        -----
        - Input is safely cast to float32.
        - Tagging and layout handling are preserved via tracker logic.
        - Uses config-defined finite difference and boundary modes.
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
        Compute the N-dimensional Hessian matrix using Sobel filters (NumPy backend).

        Applies two successive Sobel operations along each pair of spatial axes
        to estimate second-order partial derivatives.

        Parameters
        ----------
        u : np.ndarray
            Input scalar field (ND array, tracked or raw).

        Returns
        -------
        np.ndarray
            Hessian tensor of shape [D, D, ...], where D is the number of spatial axes.
            Includes updated layout and tagging metadata with 'direction_axis': (0, 1).

        Notes
        -----
        - Uses `scipy.ndimage.sobel` with 'reflect' mode.
        - Result includes auto-tagging and layout propagation.
        - No normalization or scaling is applied.
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
        Compute the N-dimensional gradient of a NumPy array using joblib-based parallelism.

        Applies finite difference operations along each spatial axis in parallel,
        using spacing and boundary conditions defined in the configuration.

        Parameters
        ----------
        u : np.ndarray
            Input scalar field (ND array, tracked or raw).

        Returns
        -------
        np.ndarray
            Stacked gradient array of shape [D, ...], where D is the number of spatial axes.
            Includes updated layout and tagging metadata.

        Notes
        -----
        - Uses `joblib.Parallel` for parallel axis-wise computation.
        - Input is cast to float32 before processing.
        - Layout and tags are automatically propagated.
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
        Compute the N-dimensional divergence of a NumPy vector field using joblib-based parallelism.

        Applies finite difference operations along each spatial axis in parallel,
        with optional per-direction weighting and maximum flux clipping.

        Parameters
        ----------
        v : np.ndarray
            Input vector field of shape [D, ...], where D is the number of spatial axes.
            Axis 0 corresponds to the direction_axis.
        weight : np.ndarray or None, optional
            Optional per-direction weight array (same shape as `v`).

        Returns
        -------
        np.ndarray
            Scalar divergence field with same spatial shape as a single vector component.
            Includes updated layout and tagging metadata.

        Notes
        -----
        - Uses `joblib.Parallel` to compute per-axis divergence terms concurrently.
        - Supports spacing and boundary modes as defined in config.
        - Tags and layout are propagated from the input field.
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
        Compute the N-dimensional Hessian matrix using NumPy with joblib-based parallelism.

        Applies two successive finite difference operations for each pair of spatial axes
        in parallel, using spacing and boundary rules from config.

        Parameters
        ----------
        u : np.ndarray
            Input scalar field (ND array), raw or tracked.

        Returns
        -------
        np.ndarray
            Hessian tensor of shape [D, D, ...], where D is the number of spatial axes.
            Includes updated tags and layout metadata, with 'direction_axis': (0, 1).

        Notes
        -----
        - Uses `joblib.Parallel` to compute each second-order derivative in parallel.
        - Preserves tags and ND layout using tracker logic.
        - Input is cast to float32 before processing.
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
        Compute the N-dimensional gradient using classic NumPy slicing (non-vectorized).

        Applies spacing-aware finite differences along each spatial axis using
        standard slicing operations, without vectorized backends or joblib parallelism.

        Parameters
        ----------
        u : np.ndarray
            Input ND array representing a scalar field (tracked or raw).

        Returns
        -------
        np.ndarray
            Stacked gradient array of shape [D, ...], where D is the number of spatial axes.
            Includes updated layout and tagging metadata.

        Notes
        -----
        - Intended as a simple and readable baseline implementation.
        - Supports spacing and boundary modes as defined in config.
        - Uses layout-aware tagging via tracker logic.
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
        Compute the N-dimensional divergence using classic NumPy slicing (non-vectorized).

        Computes the divergence from a stacked vector field using sequential
        finite differences along each spatial axis. Intended as a simple baseline
        without vectorization or parallelism.

        Parameters
        ----------
        v : np.ndarray
            Input vector field of shape [D, ...], where D is the number of spatial axes.
            Axis 0 must be the direction_axis.
        weight : np.ndarray or None, optional
            Optional per-direction weights. If provided, applied before differentiation.

        Returns
        -------
        np.ndarray
            Scalar divergence field with same spatial shape as a single vector component.
            Includes updated layout and tagging metadata.

        Notes
        -----
        - Input is safely cast to float32.
        - Spacing, boundary mode, and clipping are applied per axis.
        - Uses tracker logic to maintain ND metadata and layout consistency.
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
        Compute the N-dimensional Hessian matrix using classic nested finite difference loops.

        Computes all second-order partial derivatives via two successive directional
        finite differences, without relying on external gradient calls. This method
        serves as a baseline implementation that is fully ND-aware.

        Parameters
        ----------
        u : np.ndarray
            Input scalar field (ND array), tracked or raw.

        Returns
        -------
        np.ndarray
            Hessian tensor of shape [D, D, ...], where D is the number of spatial axes.
            Includes updated tagging and layout metadata, with 'direction_axis': (0, 1).

        Notes
        -----
        - Uses simple nested loops instead of vectorized or parallel computation.
        - Applies spacing and boundary conditions per axis.
        - Preserves layout and ND metadata through tracker propagation.
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
    Universal wrapper to configure and apply a DiffOperator method on a given image.

    Allows dynamic selection of differentiation method, backend strategy,
    layout handling, and output format.

    Parameters
    ----------
    img : ArrayLike
        Input image or volume (NumPy or Torch array).
    func : str, optional
        Name of the method to call on the DiffOperator instance.
        Options: "gradient", "divergence", "hessian", "sobel", "sobel_hessian".
        Default is "gradient".
    diff_strategy : str, optional
        Differentiation backend strategy.
        Options: "vectorized", "torch", "numpy", "parallel", "classic".
        Default is "vectorized".
    framework : {"numpy", "torch"}, optional
        Framework used to process the image and execute operations.
        Default is "numpy".
    output_format : {"numpy", "torch"}, optional
        Format of the returned result. Default is "numpy".
    layout_name : str, optional
        Layout string describing axis order (e.g., "HWC", "ZYX"). Default is "HWC".
    layout_framework : {"numpy", "torch"}, optional
        Framework used to resolve layout conventions. Default is "numpy".
    backend : {"sequential", "parallel"}, optional
        Execution mode for NumPy processing. Default is "sequential".

    Returns
    -------
    ArrayLike
        Output of the requested operation, formatted according to the selected framework
        and output_format. Includes updated tags and layout metadata if supported.

    Raises
    ------
    ValueError
        If the requested method does not exist in the DiffOperator instance.
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

