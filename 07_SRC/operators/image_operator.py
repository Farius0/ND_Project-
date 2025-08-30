# ==================================================
# =============  MODULE: image_operator  ===========
# ==================================================
from __future__ import annotations

from typing import Optional, Union, Tuple, Dict, Any, Literal
import torch, numpy as np

from core.operator_core import OperatorCore
from core.layout_axes import resolve_and_clean_layout_tags
from core.config import LayoutConfig, GlobalConfig, ImageProcessorConfig
from degradations.noise import apply_noise
from degradations.blur import apply_spatial_blur
from degradations.inpaint import apply_inpaint

ArrayLike = Union[np.ndarray, torch.Tensor]
Framework = Literal["numpy", "torch"]

# Public API
__all__ = ["Operator", "DeepOperator", "clip_image", "operator"]

# ====[ Clip image between 0 and 1 if required ]====
def clip_image(img: ArrayLike, framework: Framework, enabled: bool = True) -> ArrayLike:
    """
    Clip the image/tensor to [0, 1] if `enabled=True`.

    Parameters
    ----------
    img : ndarray | Tensor
        Input image.
    framework : {'numpy','torch'}
        Backend flag to choose the proper clipping op.
    enabled : bool, default True
        Whether to clip.

    Returns
    -------
    ndarray | Tensor
        Clipped image/tensor or original if disabled.
    """
    if not enabled:
        return img
    if hasattr(img, "get"):  # AxisTracker-compatible
        img = img.get()
    return torch.clamp(img, 0, 1) if framework == "torch" else np.clip(img, 0, 1)


# ==================================================
# ==================== Operator ====================
# ==================================================

# ====[ Operator: Classical Degradation Simulator ]====
class Operator(OperatorCore):
    """
    Classical degradation operators: noise, blur, inpaint.
    Inherits from OperatorCore for conversion, tagging and output formatting.

    Notes
    -----
    - Dual-backend: keeps NumPy↔Torch symmetry.
    - ND-ready: 2D/3D first-class; higher D when meaningful.
    - Layout-aware tagging through OperatorCore facilities.
    """

    def __init__(
        self,
        image: ArrayLike,
        clip: bool = False,
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize the classical operator wrapper.

        Parameters
        ----------
        image : ndarray | Tensor
            Input image/volume.
        clip : bool, default False
            Clip outputs to [0, 1] when True.
        layout_cfg : LayoutConfig
            Axis/layout configuration.
        global_cfg : GlobalConfig
            Global behavior (framework, output_format, device, etc.).
        """  
        # ====[ Configuration ]====
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg
        
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
        self.clip: bool = bool(clip)
        self.verbose: bool = bool(self.global_cfg.verbose)  
            
        # ====[ Mirror inherited params locally for easy access ]====
        self.framework: Framework = self.global_cfg.framework.lower()
        self.output_format: Framework = self.global_cfg.output_format.lower()
        self.add_batch_dim: Optional[bool] = self.global_cfg.add_batch_dim
        self.add_channel_dim: Optional[bool] = self.global_cfg.add_channel_dim
        self.normalize: bool = bool(self.global_cfg.normalize)
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
        
        if image is None:
            raise ValueError("Input image must not be None")

        self.image = self.convert_once(image, tag_as="original")

        if self.verbose:
            print(f"[Operator] Image loaded: shape = {getattr(self.image, 'shape', None)}")

    # ====[ Apply noise degradation ]====
    def noise(self, sigma: float = 0.2) -> ArrayLike:
        """
        Apply additive Gaussian noise to the input image.

        Parameters
        ----------
        sigma : float
            Standard deviation of the Gaussian noise.

        Returns
        -------
        noised : np.ndarray | torch.Tensor
            Noised image, tagged and optionally clipped.
        """
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")

        degraded = apply_noise(
            self.image,
            sigma=sigma,
            framework=self.framework,
        )
        degraded = clip_image(degraded, self.framework, self.clip)

        tracker = self.track(self.image).copy_to(degraded)

        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tracker, self.framework, self.layout_name, prefix="", remove_prefix=False
        )

        tracker.update_tags({
            "status": "noised",
            "layout_name": layout_name,
            **axes_tags
        })

        return self.to_output(
            tracker.get(),
            tag_as="output",
            enable_uid=True,
            op_params={"op": "noise", "sigma": sigma}
        )

    # ====[ Apply spatial blur degradation ]====
    def blur(self, sigma: float = 1.0, return_kernel: bool = False) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        """
        Apply Gaussian blur to the input image.

        Parameters
        ----------
        sigma : float
            Standard deviation of the Gaussian kernel.
        return_kernel : bool
            If True, also returns the blur kernel.

        Returns
        -------
        blurred : np.ndarray | torch.Tensor
            Blurred image.
        kernel : optional
            Kernel used for blurring.
        """
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")

        degraded, kernel = apply_spatial_blur(
            self.image,
            sigma=sigma,
            framework=self.framework,
            channel_axis=self.get_axis(self.image, "channel_axis"),
            return_kernel=True
        )

        degraded = clip_image(degraded, self.framework, self.clip)

        tracker = self.track(self.image).copy_to(degraded)
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tracker, self.framework, self.layout_name, prefix="", remove_prefix=False
        )

        tracker.update_tags({
            "status": "blurred",
            "layout_name": layout_name,
            **axes_tags
        })

        degraded = self.to_output(
            tracker.get(),
            tag_as="output",
            enable_uid=True,
            op_params={"op": "blur", "sigma": sigma}
        )

        if return_kernel:
            kernel_out = self.to_output(
                kernel,
                tag_as="output",
                enable_uid=True,
                op_params={"op": "blur", "type": "kernel", "sigma": sigma}
            )
            return degraded, kernel_out

        return degraded

    # ====[ Apply inpainting degradation ]====
    def inpaint(
        self,
        mask: Optional[ArrayLike] = None,
        sigma: float = 0.05,
        threshold: float = 0.4,
        mode: str = "replace",
        seed: Optional[int] = None,
        return_mask: bool = False,
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        """
        Apply inpainting to an image using a mask and optional noise.

        Parameters
        ----------
        mask : np.ndarray | torch.Tensor | None
            Binary mask to use for inpainting. If None, will be generated.
        sigma : float
            Standard deviation of added noise inside the mask.
        threshold : float
            Threshold for mask generation (if mask is None).
        mode : str
            'replace' or 'keep' strategy inside mask.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        inpainted : np.ndarray | torch.Tensor
        used_mask : np.ndarray | torch.Tensor
        """
        degraded, used_mask = apply_inpaint(
            self.image,
            mask=mask,
            sigma=sigma,
            framework=self.framework,
            threshold=threshold,
            mode=mode,
            seed=seed
        )

        degraded = clip_image(degraded, self.framework, self.clip)

        # === Track & tag degraded image
        tracker = self.track(self.image).copy_to(degraded)
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tracker, self.framework, self.layout_name, prefix="", remove_prefix=False
        )

        tracker.update_tags({
            "status": "inpainted",
            "layout_name": layout_name,
            **axes_tags
        })

        degraded = self.to_output(
            tracker.get(),
            tag_as="output",
            enable_uid=True,
            op_params={"op": "inpaint", "sigma": sigma, "threshold": threshold}
        )

        # === Track & tag used_mask
        mask_tracker = self.track(self.image).copy_to(used_mask)
        mask_tracker.update_tag("status", "mask")

        used_mask = self.to_output(
            mask_tracker.get(),
            tag_as="output",
            enable_uid=True,
            op_params={"op": "inpaint", "type": "mask"}
        )
        if return_mask:
            return degraded, used_mask
        return degraded

    # ====[ Operator summary ]====
    def summary(self) -> None:
        """
        Print a summary of the Operator configuration and input image metadata.
        """
        print("====[ Operator Summary ]====")
        print(f"Framework     : {self.framework}")
        print(f"Output Format : {self.output_format}")
        print(f"Device        : {self.device}")
        print(f"Normalize     : {self.normalize}")
        print(f"Clipping      : {self.clip}")
        print(f"Add Batch Dim : {self.add_batch_dim}")
        print(f"Verbose       : {self.verbose}")

        print(f"Image Type    : {type(self.image)}")
        print(f"Image Shape   : {getattr(self.image, 'shape', None)}")

        # === Axes
        print("Axes ND:")
        for k, v in self.axes.items():
            print(f"  {k:<16}: {v}")

        # === Tag info
        tag = self.get_tag(self.image, self.framework) if self.has_tag(self.image, self.framework) else None
        if tag:
            print("Tag Metadata:")
            for key, value in tag.items():
                print(f"  {key:<16}: {value}")
        else:
            print("Tag Metadata  : None")


# ==================================================
# ================== DeepOperator ==================
# ==================================================

class DeepOperator(OperatorCore):
    """
    DeepInv-based degradations (noise, blur, inpainting).
    Assumes Torch backend and (B, C, H, W) tensors.
    """

    def __init__(
        self,
        image: ArrayLike,
        clip: bool = False,
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize the deep operator wrapper.

        Parameters
        ----------
        image : ndarray | Tensor
            Input image. Will be converted to the configured framework.
        clip : bool, default False
            Clip outputs to [0, 1].
        layout_cfg : LayoutConfig
            Axis/layout configuration.
        global_cfg : GlobalConfig
            Global behavior (framework, output_format, device, etc.).
        """ 
        # ====[ Configuration ]====
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg
        
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
        self.clip: bool = bool(clip)
        self.verbose: bool = bool(self.global_cfg.verbose)  
            
        # ====[ Mirror inherited params locally for easy access ]====
        self.framework: Framework = self.global_cfg.framework.lower()
        self.output_format: Framework = self.global_cfg.output_format.lower()
        self.add_batch_dim: Optional[bool] = self.global_cfg.add_batch_dim
        self.add_channel_dim: Optional[bool] = self.global_cfg.add_channel_dim
        self.normalize: bool = bool(self.global_cfg.normalize)
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
        
        if image is None:
            raise ValueError("Input image must not be None")

        self.image = self.convert_once(image, tag_as="original")

        if self.verbose:
            print(f"[DeepOperator] Image loaded: shape = {getattr(self.image, 'shape', None)}")

    # ====[ Internal utility to apply a DeepInv operator ]====
    def _apply_operator(self, operator) -> torch.Tensor:
        """
        Apply a deepinv.physics operator to the tracked image.

        Parameters
        ----------
        operator : callable
            A DeepInv operator (e.g., Blur, Denoising, Inpainting)

        Returns
        -------
        result : torch.Tensor
            Degraded image (B, C, H, W), potentially clipped.
        """
        if self.verbose:
            print(f"[DeepOperator] Applying: {operator.__class__.__name__}")

        with torch.no_grad():
            result = operator(self.image)

        result = clip_image(result, self.framework, self.clip)
        return result

    # ====[ Add Gaussian Noise ]====
    def noise(self, sigma: float = 0.2) -> torch.Tensor:
        """
        Apply additive Gaussian noise using DeepInv's Denoising operator.

        Parameters
        ----------
        sigma : float
            Standard deviation of the noise.

        Returns
        -------
        noised : torch.Tensor
            Noised image with full ND tracking and tags.
        """
        import deepinv as dinv
        
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")

        op = dinv.physics.Denoising(
            noise_model=dinv.physics.GaussianNoise(sigma=sigma)
        ).to(self.device)

        degraded = self._apply_operator(op)

        tracker = self.track(self.image).copy_to(degraded)
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tracker, self.framework, self.layout_name, prefix="", remove_prefix=False
        )

        tracker.update_tags({
            "status": "noised",
            "layout_name": layout_name,
            **axes_tags
        })

        return self.to_output(
            tracker.get(),
            tag_as="output",
            enable_uid=True,
            op_params={"op": "noise", "sigma": sigma}
        )

    # ====[ Apply Gaussian Blur ]====
    def blur(self, sigma: Union[Tuple[float, float], float] = (2.0, 2.0), angle: float = 0.0, return_kernel: bool = False):
        """
        Apply a DeepInv Gaussian blur operator.

        Parameters
        ----------
        sigma : tuple of 2 floats
            Standard deviations of the Gaussian kernel in (x, y).
        angle : float
            Rotation angle in degrees.

        Returns
        -------
        blurred : torch.Tensor
        kernel  : torch.Tensor
        """
        import deepinv as dinv
        
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        
        if not (isinstance(sigma, tuple) and len(sigma) == 2 and all(s > 0 for s in sigma)):
            raise ValueError("Sigma must be a tuple of two positive values.")

        filt = dinv.physics.blur.gaussian_blur(sigma=sigma, angle=angle).to(self.device)
        blur_op = dinv.physics.Blur(filt, padding='circular').to(self.device)

        degraded = self._apply_operator(blur_op)

        tracker = self.track(self.image).copy_to(degraded)
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tracker, self.framework, self.layout_name, prefix="", remove_prefix=False
        )

        tracker.update_tags({
            "status": "blurred",
            "layout_name": layout_name,
            **axes_tags
        })

        degraded = self.to_output(
            tracker.get(),
            tag_as="output",
            enable_uid=True,
            op_params={"op": "blur", "sigma": sigma}
        )

        kernel = self.track(self.image).copy_to(filt)
        kernel = self.to_output(
            kernel.get(),
            tag_as="output",
            enable_uid=True,
            op_params={"op": "blur", "type": "kernel", "sigma": sigma}
        )
        if not return_kernel:
            return degraded
        return degraded, kernel

    # ====[ Inpainting with Optional Mask ]====
    def inpaint(self, mask: Optional[ArrayLike] = None, sigma: float = 0.05, threshold: float = 0.4, return_mask: bool = False):
        """
        Apply DeepInv inpainting using a binary mask and additive noise.

        Parameters
        ----------
        mask : np.ndarray | torch.Tensor | None
            Binary mask where True means missing data.
        sigma : float
            Noise standard deviation applied inside the mask.
        threshold : float
            Used to generate mask if none is provided.

        Returns
        -------
        inpainted : torch.Tensor
        mask      : torch.Tensor
        """
        import deepinv as dinv
        
        if self.image.ndim != 4:
            raise ValueError("DeepOperator expects image of shape (B, C, H, W)")

        _, _, H, W = self.image.shape

        if sigma < 0:
            raise ValueError("Sigma must be non-negative.")

        if mask is None:
            mask = torch.rand(1, 1, H, W, device=self.device) > threshold
        else:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = mask.to(self.device).bool()
            if mask.shape[-2:] != (H, W):
                raise ValueError(f"Mask dimensions must match image size: {(H, W)}")

        op = dinv.physics.Inpainting(
            mask=mask,
            tensor_size=(H, W),
            noise_model=dinv.physics.GaussianNoise(sigma=sigma)
        ).to(self.device)

        inpainted = self._apply_operator(op)

        tracker = self.track(self.image).copy_to(inpainted)
        layout_name, axes_tags = resolve_and_clean_layout_tags(
            tracker, self.framework, self.layout_name, prefix="", remove_prefix=False
        )

        tracker.update_tags({
            "status": "inpainted",
            "layout_name": layout_name,
            **axes_tags
        })

        inpainted = self.to_output(
            tracker.get(),
            tag_as="output",
            enable_uid=True,
            op_params={"op": "inpaint", "sigma": sigma, "threshold": threshold}
        )

        # === Tag du masque (sans layout)
        mask_tracker = self.track(self.image).copy_to(mask)
        mask_tracker.update_tags({
            "status": "mask",
            "layout_name": layout_name,
        })

        mask = self.to_output(
            mask_tracker.get(),
            tag_as="output",
            enable_uid=True,
            op_params={"op": "inpaint", "type": "mask"}
        )

        if not return_mask:
            return inpainted
        return inpainted, mask

    # ====[ Summary Printer ]====
    def summary(self) -> None:
        """
        Print the current configuration and input tag metadata.
        """
        print("====[ DeepOperator Summary ]====")
        print(f"Device         : {self.device}")
        print(f"Framework      : {self.framework}")
        print(f"Output Format  : {self.output_format}")
        print(f"Normalize      : {self.normalize}")
        print(f"Clipping       : {self.clip}")
        print(f"Add Batch Dim  : {self.add_batch_dim}")
        print(f"Verbose        : {self.verbose}")

        print(f"Image Type     : {type(self.image)}")
        print(f"Image Shape    : {getattr(self.image, 'shape', None)}")

        # === Axes ND
        print("Axes ND:")
        for k, v in self.axes.items():
            print(f"  {k:<16}: {v}")

        # === Tag
        tag = self.get_tag(self.image, self.framework) if self.has_tag(self.image, self.framework) else None
        if tag:
            print("Tag Metadata:")
            for key, value in tag.items():
                print(f"  {key:<16}: {value}")
        else:
            print("Tag Metadata  : None")

# ======================================================================
#                      Convenience wrapper
# ======================================================================

def operator(
    img: ArrayLike,
    operator: str = "noise",
    operator_class=Operator,
    noise_level: float = 0.2,
    blur_level: float = 0.5,
    threshold: float = 0.7,
    mask_mode: str = "grid_noised",
    framework: Framework = "numpy",
    output_format: Framework = "numpy",
    layout_framework: Framework = "numpy",
    layout_name: str = "HWC",
    processor_strategy: Optional[str] = None,
    add_batch_dim: Optional[bool] = None,
    backend: str = "sequential",
) -> ArrayLike:
    """
    Convenience entrypoint to run a chosen operator through ImageProcessor.

    Notes
    -----
    - If operator_class is DeepOperator, framework is forced to 'torch' and
      add_batch_dim defaults to True.
    """    
    
    from operators.image_processor import ImageProcessor
    
    # ====[ Fallback ]====
    processor_strategy=processor_strategy or "vectorized" if framework == "numpy" else "torch"   
        
    # ====[ Configuration ]====
    proc_params: Dict[str, Any] = {"processor_strategy": processor_strategy, "return_tuple":True}
    layout_params: Dict[str, Any] = {"layout_name": layout_name, "layout_framework": layout_framework}
    global_params: Dict[str, Any] = {"framework": "torch" if operator_class == DeepOperator else framework, "output_format": output_format, 
                     "backend": backend, "add_batch_dim":True if operator_class == DeepOperator else add_batch_dim} 
    operator_params: Dict[str, Any] = {"clip":True, "layout_cfg":LayoutConfig(**layout_params), "global_cfg":GlobalConfig(**global_params)}   
    
    if operator_class == Operator:
        if operator == "noise":
            func = lambda x: operator_class(x, **operator_params).noise(sigma=noise_level)
        elif operator == "blur":
            func = lambda x: operator_class(x, **operator_params).blur(sigma=blur_level)
        elif operator == "inpaint":
            func = lambda x: operator_class(x, **operator_params).inpaint(sigma=noise_level, threshold=threshold, 
                                                                          mode=mask_mode, return_mask=True)
    
    elif operator_class == DeepOperator:
        if operator == "noise":
            func = lambda x: operator_class(x, **operator_params).noise(sigma=noise_level)
        elif operator == "blur":
            func = lambda x: operator_class(x, **operator_params).blur(sigma=blur_level)
        elif operator == "inpaint":
            func = lambda x: operator_class(x, **operator_params).inpaint(sigma=noise_level, threshold=threshold)
    
    processor = ImageProcessor(
                            img_process_cfg = ImageProcessorConfig(function=func, **proc_params,),
                            layout_cfg=LayoutConfig(**layout_params),
                            global_cfg=GlobalConfig(**global_params),
                            )
    
    return processor(img)