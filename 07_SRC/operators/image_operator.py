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
    Clip an image or tensor to the [0, 1] range if enabled.

    Parameters
    ----------
    img : ndarray or Tensor
        Input image. Can also be AxisTracker-compatible (with `.get()` method).
    framework : {'numpy', 'torch'}
        Backend to use for clipping operation.
    enabled : bool, default True
        If False, returns the input unchanged.

    Returns
    -------
    ndarray or Tensor
        Clipped image/tensor if enabled; otherwise the original input.
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
    Classical image degradation operators: noise, blur, and inpainting.

    Inherits from OperatorCore to provide layout-aware processing, axis tagging,
    and dual-backend support (NumPy and Torch).

    Notes
    -----
    - Dual-backend ready: consistent behavior with NumPy and Torch inputs.
    - ND-aware: supports 2D and 3D natively; higher dimensions when meaningful.
    - Preserves and updates layout metadata using OperatorCore facilities.
    """

    def __init__(
        self,
        image: ArrayLike,
        clip: bool = False,
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize the classical operator wrapper with layout and backend configuration.

        Parameters
        ----------
        image : ndarray or Tensor
            Input image or volume to process.
        clip : bool, default False
            If True, output is clipped to the [0, 1] range after processing.
        layout_cfg : LayoutConfig
            Configuration for interpreting and tracking axis layout.
        global_cfg : GlobalConfig
            Configuration for backend selection, output format, and device behavior.
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
        Apply additive Gaussian noise to the input image or volume.

        The degraded image is automatically tagged, layout-tracked,
        and clipped to [0, 1] if `self.clip` is enabled.

        Parameters
        ----------
        sigma : float, default 0.2
            Standard deviation of the Gaussian noise. Must be strictly positive.

        Returns
        -------
        ArrayLike
            Noised image or volume (NumPy or Torch), with layout and UID tagging.

        Raises
        ------
        ValueError
            If `sigma` is not strictly positive.

        Notes
        -----
        - The operation preserves shape and dtype.
        - The output is tagged with status='noised' and includes axis layout info.
        - Uses the configured backend (`self.framework`) and clipping policy (`self.clip`).
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
        Apply Gaussian blur to the input image or volume.

        The result is layout-tagged, optionally clipped to [0, 1], and optionally
        returned with the underlying Gaussian kernel used for blurring.

        Parameters
        ----------
        sigma : float, default 1.0
            Standard deviation of the Gaussian kernel. Must be strictly positive.
        return_kernel : bool, default False
            If True, also returns the Gaussian kernel used for blurring.

        Returns
        -------
        blurred : ArrayLike
            Blurred image (NumPy or Torch), with layout and UID tags.
        kernel : ArrayLike, optional
            The Gaussian kernel used (if `return_kernel=True`).

        Raises
        ------
        ValueError
            If `sigma` is not strictly positive.

        Notes
        -----
        - Automatically uses the correct channel axis (if defined in the layout).
        - Output is tagged with `status='blurred'` and updated layout metadata.
        - Uses the configured backend (`self.framework`) and clipping setting (`self.clip`).
        - Returns one or two outputs depending on `return_kernel`.
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
        Apply inpainting to the input image using a binary mask and optional noise.

        The masked regions can be replaced or preserved, and noise can be injected
        inside the masked zones. The mask can be provided or generated automatically.

        Parameters
        ----------
        mask : ndarray or Tensor, optional
            Binary mask to apply. If None, it will be generated based on thresholding.
        sigma : float, default 0.05
            Standard deviation of Gaussian noise added inside the mask.
        threshold : float, default 0.4
            Threshold for automatic mask generation (if `mask` is None).
        mode : str, default 'replace'
            Inpainting strategy. 'replace' fills masked regions with noise,
            'keep' keeps original values outside the mask.
        seed : int or None, optional
            Seed for random number generation (for reproducibility).
        return_mask : bool, default False
            If True, also return the inpainting mask used.

        Returns
        -------
        inpainted : ndarray or Tensor
            Image with masked regions inpainted.
        used_mask : ndarray or Tensor, optional
            The binary mask used, only returned if `return_mask=True`.

        Notes
        -----
        - The inpainted image is tagged with `status='inpainted'`, layout info, and UID.
        - The mask is also tagged (`status='mask'`) and returned as a traceable output.
        - Works with NumPy or Torch, and supports both 2D and 3D images.
        - Layout and axis tags are preserved and updated using `OperatorCore`.
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
    Torch-based degradation operators using DeepInv-style models.

    Assumes Torch tensors with shape (B, C, H, W) or (B, C, D, H, W).
    Inherits from OperatorCore to provide layout-aware processing,
    tagging, and output formatting.

    Notes
    -----
    - Backend is fixed to Torch (conversion is enforced internally).
    - Designed to interface with learned inverse models.
    - LayoutConfig and GlobalConfig control axis handling and output format.
    """
    def __init__(
        self,
        image: ArrayLike,
        clip: bool = False,
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize the deep operator wrapper with layout and backend configuration.

        Parameters
        ----------
        image : ndarray or Tensor
            Input image or batch to be processed (will be converted to Torch if needed).
        clip : bool, default False
            Whether to clip the output to [0, 1] after degradation.
        layout_cfg : LayoutConfig
            Configuration for axis interpretation and layout tracking.
        global_cfg : GlobalConfig
            Global behavior: framework, device, output format, backend strategy, etc.
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
        Apply a DeepInv-style physics operator to the tracked image.

        Parameters
        ----------
        operator : callable
            A DeepInv degradation operator (e.g., Blur, Denoising, Inpainting).
            Must be callable and accept a Torch tensor input.

        Returns
        -------
        result : torch.Tensor
            Degraded image, shaped (B, C, H, W) or (B, C, D, H, W),
            clipped to [0, 1] if `self.clip` is True.

        Notes
        -----
        - Runs the operator in `no_grad` mode to disable gradient tracking.
        - Assumes input image has already been converted to a Torch tensor.
        - Clipping is performed via `clip_image()` using the current backend config.
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
        Apply additive Gaussian noise using DeepInv's denoising operator.

        Parameters
        ----------
        sigma : float, default 0.2
            Standard deviation of the Gaussian noise. Must be strictly positive.

        Returns
        -------
        noised : torch.Tensor
            Noised image or volume (Torch tensor), fully tagged and layout-tracked.

        Raises
        ------
        ValueError
            If `sigma` is not strictly positive.

        Notes
        -----
        - Internally uses `deepinv.physics.Denoising` with a Gaussian noise model.
        - Operates in `no_grad` mode for performance and memory efficiency.
        - The result is tagged with `status='noised'`, layout metadata, and a unique UID.
        - Clipping is applied automatically if `self.clip` is enabled.
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
        Apply anisotropic Gaussian blur using DeepInv's blur operator.

        Parameters
        ----------
        sigma : float or tuple of two floats, default (2.0, 2.0)
            Standard deviations of the Gaussian kernel in (x, y) directions.
            If a single float is provided, the same value is used for both axes.
        angle : float, default 0.0
            Rotation angle (in degrees) applied to the kernel.
        return_kernel : bool, default False
            If True, also return the blur kernel used.

        Returns
        -------
        blurred : torch.Tensor
            Blurred image, layout-tagged and traced.
        kernel : torch.Tensor, optional
            Kernel tensor used for blurring, only returned if `return_kernel=True`.

        Raises
        ------
        ValueError
            If `sigma` is not a positive float or a valid (positive, positive) tuple.

        Notes
        -----
        - Uses `deepinv.physics.gaussian_blur()` and wraps it with `deepinv.physics.Blur`.
        - Applies padding mode `'circular'` to preserve spatial dimensions.
        - The resulting image is tagged with `status='blurred'` and has full UID/axis tracking.
        - The kernel is also tagged if returned.
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
        Apply DeepInv inpainting to the input image using a binary mask and optional Gaussian noise.

        Parameters
        ----------
        mask : np.ndarray or torch.Tensor or None, default None
            Binary mask where True/1 indicates missing regions to inpaint.
            If None, a random mask is generated using the given threshold.
        sigma : float, default 0.05
            Standard deviation of the Gaussian noise applied inside the masked regions.
        threshold : float, default 0.4
            Threshold for generating the random mask (if `mask` is None).
        return_mask : bool, default False
            If True, also return the mask used for inpainting.

        Returns
        -------
        inpainted : torch.Tensor
            Image with masked regions filled, tagged and tracked.
        mask : torch.Tensor, optional
            The binary mask used during inpainting (if `return_mask=True`).

        Raises
        ------
        ValueError
            If input image does not have shape (B, C, H, W) or if the mask shape is invalid.
            If sigma is negative.

        Notes
        -----
        - Uses `deepinv.physics.Inpainting` with a Gaussian noise model.
        - Tags output with `status='inpainted'` and layout metadata.
        - Generated or provided mask is also tagged (`status='mask'`).
        - Layout is resolved via `LayoutConfig`, and UID tracking is applied.
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
    Run a simple image operator (noise, blur, inpaint) using the ImageProcessor pipeline.

    Dynamically builds the appropriate processor configuration for the chosen operator,
    backend, and layout, and executes the operation on the given image.

    Parameters
    ----------
    img : ArrayLike
        Input image as NumPy array or Torch tensor.
    operator : str, default "noise"
        Operation to apply. Supported values: "noise", "blur", "inpaint".
    operator_class : class, default Operator
        Operator class to instantiate. Can be `Operator` or `DeepOperator`.
    noise_level : float, default 0.2
        Standard deviation for noise injection.
    blur_level : float, default 0.5
        Standard deviation for Gaussian blur.
    threshold : float, default 0.7
        Threshold used in inpainting mask.
    mask_mode : str, default "grid_noised"
        Mode for inpainting mask generation (if applicable).
    framework : {"numpy", "torch"}, default "numpy"
        Backend used for processing and layout conversion.
    output_format : {"numpy", "torch"}, default "numpy"
        Format of the returned result.
    layout_framework : {"numpy", "torch"}, default "numpy"
        Framework used to resolve the layout (e.g., "HWC" → axes).
    layout_name : str, default "HWC"
        Layout description of the input image.
    processor_strategy : str, optional
        Strategy used by the processor. Default is "vectorized" for NumPy and "torch" for Torch.
    add_batch_dim : bool, optional
        Whether to force batch dimension. Defaults to True for DeepOperator.
    backend : str, default "sequential"
        Execution backend strategy (e.g., "sequential", "parallel").

    Returns
    -------
    ArrayLike
        Resulting image after applying the selected operator.

    Notes
    -----
    - If `operator_class` is `DeepOperator`, the framework is forced to 'torch' and
      `add_batch_dim` defaults to True.
    - Operator parameters are injected into the corresponding method on the fly.
    - Uses `ImageProcessor` internally, which ensures layout tagging and axis tracking.
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