# ==================================================
# =============  MODULE: operator_core =============
# ==================================================
from __future__ import annotations

from typing import Optional, Literal, Any, Dict

import numpy as np
import torch

from core.base_converter import BaseConverter
from core.config import GlobalConfig, LayoutConfig

# Public API
__all__ = ["OperatorCore"]

ArrayLike = np.ndarray | torch.Tensor
Framework = Literal["numpy", "torch"]

# ==================================================
# ================== OperatorCore ==================
# ==================================================
class OperatorCore(BaseConverter):
    """
    Base class for ND operators, extending BaseConverter with convenience helpers.

    Notes
    -----
    - ND-ready; dual-backend; preserves dtype/device/layout when feasible.
    - Unifies conversion, format control, and traceable tagging for NumPy/Torch.
    """

    # ====[ INITIALIZATION – OperatorCore ]====
    def __init__(
        self,
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Core ND operator initialization (layout-aware, traceable, compatible).

        Parameters
        ----------
        layout_cfg : LayoutConfig
            Provides axis roles, default layout names, and framework mapping.
        global_cfg : GlobalConfig
            Provides normalization policy, default device, and default flags.
        """
        # Mirror config locally (readable attributes)
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg

        # Axis defaults resolved from layout_cfg
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

        # Inherited params exposed locally
        self.framework: str = self.global_cfg.framework.lower()
        self.output_format: str = self.global_cfg.output_format.lower()
        self.add_batch_dim: Optional[bool] = self.global_cfg.add_batch_dim
        self.add_channel_dim: Optional[bool] = self.global_cfg.add_channel_dim
        self.normalize: bool = bool(self.global_cfg.normalize)
        self.device: str = (
            "cuda"
            if (torch.cuda.is_available() and self.framework == "torch")
            else self.global_cfg.device
        )

        # Initialize BaseConverter
        super().__init__(layout_cfg=self.layout_cfg, global_cfg=self.global_cfg)

    # ====[ RESOLVE AXIS – Get Axis from Image ]====
    def resolve_axis(self, *args, default: Optional[Any]=None) -> Optional[Any]:
        """Return the first non-None value among `args`, otherwise `default`."""
        return super().resolve_axis(*args, default=default)

    # ====[ ALREADY CONVERTED – Check Conversion Status ]====
    def _already_converted(self, arr: ArrayLike, framework: Literal["numpy", "torch"] = "numpy") -> bool:
        """Proxy to BaseConverter.already_converted()."""
        return super().already_converted(arr, framework)

    # ====[ CONVERT ONCE (ND-compatible) – OperatorCore ]====
    def convert_once(
        self,
        image: ArrayLike,
        framework: Optional[str] = None,
        tag_as: str = "processed",
        direction_axis: int | None = None,
        batch_axis: int | None = None,
        add_batch_dim: bool | None = None,
        add_channel_dim: bool | None = None,
        channel_axis: int | None = None,
        height_axis: int | None = None,
        width_axis: int | None = None,
        depth_axis: int | None = None,
        enable_uid: bool = False,
        op_params: dict | None = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: bool | None = None,
        force_reset: bool = False,
        require_shape_match: bool = True,
        require_layout_match: bool = False,
        require_uid: bool = False,
        expected_layout_name: Optional[str] = None,
        require_status: Optional[str] = None,
    ) -> ArrayLike:
        """
        Wrapper around BaseConverter.convert_once(), enriched with all axis controls.

        Parameters
        ----------
        image : ndarray | Tensor
        framework : {'torch','numpy'}, optional
            Target framework (defaults to self.framework).
        tag_as : str, default 'processed'
            Conversion status recorded in the tag.
        ... (axes/flags identical à BaseConverter.convert_once)

        Returns
        -------
        ndarray | Tensor
            Tagged image in the requested framework.
        """
        fw = self.resolve_axis(framework, self.framework)

        if force_reset:
            self.purge_tags_for(image)

        if not force_reset and self._already_converted(image, fw):
            return image

        return super().convert_once(
            image=image,
            framework=fw,
            channel_axis=channel_axis,
            batch_axis=batch_axis,
            direction_axis=direction_axis,
            height_axis=height_axis,
            width_axis=width_axis,
            depth_axis=depth_axis,
            add_batch_dim=add_batch_dim,
            add_channel_dim=add_channel_dim,
            tag_as=tag_as,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override,
            require_shape_match=require_shape_match,
            require_layout_match=require_layout_match,
            require_uid=require_uid,
            expected_layout_name=expected_layout_name,
            require_status=require_status,
        )

    # ====[ TO OUTPUT FORMAT – Framework-aware Return ]====
    def to_output(
        self,
        image: ArrayLike,
        framework: Optional[str] = None,
        tag_as: str = "output",
        direction_axis: int | None = None,
        batch_axis: int | None = None,
        add_batch_dim: bool | None = None,
        add_channel_dim: bool | None = None,
        channel_axis: int | None = None,
        height_axis: int | None = None,
        width_axis: int | None = None,
        depth_axis: int | None = None,
        enable_uid: bool = False,
        op_params: dict | None = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: bool | None = None,
        require_shape_match: bool = True,
        require_layout_match: bool = False,
        require_uid: bool = False,
        expected_layout_name: Optional[str] = None,
        require_status: Optional[str] = None,
    ) -> ArrayLike:
        """
        Convert to the operator's output format (e.g., numpy or torch) and tag.

        Returns
        -------
        ndarray | Tensor
            Converted image with tag.
        """
        return self.convert_once(
            image=image,
            framework=self.resolve_axis(framework, self.output_format),
            channel_axis=channel_axis,
            batch_axis=batch_axis,
            direction_axis=direction_axis,
            height_axis=height_axis,
            width_axis=width_axis,
            depth_axis=depth_axis,
            add_batch_dim=add_batch_dim,
            add_channel_dim=add_channel_dim,
            tag_as=tag_as,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override,
            require_shape_match=require_shape_match,
            require_layout_match=require_layout_match,
            require_uid=require_uid,
            expected_layout_name=expected_layout_name,
            require_status=require_status,
        )

    # ====[ ENSURE FORMAT – Layout-Conformant Conversion ]====
    def ensure_format(
        self,
        image: ArrayLike,
        tag_as: str = "prepared",
        framework: Optional[str] = None,
        layout: Optional[dict] = None,
        enable_uid: bool = False,
        op_params: dict | None = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: bool | None = None,
        force_reset: bool = False,
        require_shape_match: bool = True,
        require_layout_match: bool = False,
        require_uid: bool = False,
        expected_layout_name: Optional[str] = None,
        require_status: Optional[str] = None,
    ):
        """
        Ensure that the image layout matches the expected definition ('NCHW', 'GNCDHW', ...).

        Parameters
        ----------
        image : ndarray | Tensor
        layout : dict
            Target layout dict (e.g., from get_layout_axes()).
        framework : {'torch','numpy'}, optional
        tag_as : str, default 'prepared'

        Returns
        -------
        AxisTracker
            Tracker around the converted image, with updated tag.
        """
        framework = framework or self.framework

        img = self.convert_once(
            image=image,
            framework=framework,
            tag_as=tag_as,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
            normalize_override=normalize_override,
            force_reset=force_reset,
            require_shape_match=require_shape_match,
            require_layout_match=require_layout_match,
            require_uid=require_uid,
            expected_layout_name=expected_layout_name,
            require_status=require_status,
        )
        tracker = self.track(img)
        tag = tracker.get_tag()

        if layout is not None and tag.get("layout_name") != layout.get("name"):
            for key, required in layout.items():
                if key in {"ndim", "name", "description"} or required is None:
                    continue
                current = tag.get(key, None)
                if current is not None and current != required:
                    if getattr(self, "verbose", False):
                        print(f"[ensure_format] Moving {key} axis: {current} → {required}")
                    tracker = tracker.moveaxis(current, required)
                    tag = tracker.get_tag()  # refresh

        return tracker

    def safe_copy(self, obj: np.ndarray | torch.Tensor):
        """
        Create a safe copy of a NumPy array or PyTorch tensor and propagate the tag.

        Returns
        -------
        ndarray | Tensor
            New object with copied tag (if any).
        """
        if isinstance(obj, np.ndarray):
            new_obj = obj.copy()
            fw = "numpy"
        elif isinstance(obj, torch.Tensor):
            new_obj = obj.clone()
            fw = "torch"
        else:
            raise ValueError("Unsupported input type.")

        if self.has_tag(obj, framework=fw):
            tagger = self.track(obj).copy_to(new_obj)
            return tagger.image
        return new_obj

    # ====[ IMAGE SUMMARY – Shape, Tag, Device, etc. ]====
    def summary(self, image: Any, framework: Optional[str] = None, verbose: bool = True, as_tensor: bool = False) -> None:
        """
        Print a summary (shape, dtype, device, framework, tag).

        Parameters
        ----------
        image : ndarray | Tensor
        framework : {'torch','numpy'}, optional
        verbose : bool, default True
        as_tensor : bool, default False
            Force conversion to the current framework before printing.
        """
        fw = framework or self.framework
        img = self.convert_once(image, framework=fw, tag_as="summary")

        if as_tensor:
            img = self.convert_once(img, framework=("torch" if fw == "torch" else "numpy"))

        if isinstance(img, torch.Tensor):
            dtype, shape, device = img.dtype, tuple(img.shape), img.device
        elif isinstance(img, np.ndarray):
            dtype, shape, device = img.dtype, img.shape, "cpu"
        else:
            dtype, shape, device = "unknown", "unknown", "unknown"

        print(f"Shape     : {shape}")
        print(f"Dtype     : {dtype}")
        print(f"Device    : {device}")
        print(f"Framework : {fw}")
        print(f"Output    : {self.output_format}")

        if verbose:
            tag = self.get_tag(img, fw)
            if tag:
                print("Tag Info  :")
                for k, v in tag.items():
                    print(f"  - {k}: {v}")
            else:
                print("No tag found.")

    # ====[ RESET CONVERSION TAG – Robust Tag Removal ]====
    def purge_all_tags(self, verbose: bool = False) -> None:
        """Proxy to BaseConverter.purge_all_tags()."""
        return super().purge_all_tags(verbose=verbose)

    def purge_tags_for(self, image: np.ndarray | torch.Tensor, verbose: bool = False) -> None:
        """Proxy to BaseConverter.purge_tags_for()."""
        return super().purge_tags_for(image, verbose=verbose)

    def reset_conversion_tag(self, image: ArrayLike) -> None:
        """
        Remove the conversion tag for a given image (NumPy or Torch).
        """
        for fw_loop in ["torch", "numpy"]:
            if self.has_tag(image, fw_loop):
                try:
                    self.del_tag(image, fw_loop)
                except AttributeError:
                    pass
                except Exception as e:
                    if self.verbose:
                        print(f"[reset_conversion_tag] Error removing tag from {fw_loop}: {e}")

    # ====[ COPY CONVERSION TAG – Safe Tag Copy ]====
    def copy_conversion_tag(
        self,
        source: ArrayLike,
        target: ArrayLike,
        framework_source: str = "numpy",
        framework_target: str = "torch",
    ) -> ArrayLike:
        """
        Copy the conversion tag from `source` to `target` (frameworks may differ).

        Returns
        -------
        target : ndarray | Tensor
            The target image with the copied tag.
        """
        tag = self.get_tag(source, framework_source)
        if not tag:
            raise ValueError(
                f"No valid tag found for the source image (framework: {framework_source})"
            )

        try:
            self.set_tag(target, framework=framework_target, tag=tag)
        except Exception as e:
            if self.verbose:
                print(f"[copy_conversion_tag] Error copying tag: {e}")

        return target

    # ====[ UPDATE AXIS TAG – Safely Update Axis Information ]====
    def _update_axis_tag(
        self,
        image: ArrayLike,
        key: str,
        new_axis: int | None,
        add_new_axis: bool = False,
    ) -> None:
        """
        Update a specific axis in the conversion tag (e.g., 'channel_axis').

        Parameters
        ----------
        image : ndarray | Tensor
        key : str
            Axis key (e.g., 'channel_axis', 'batch_axis').
        new_axis : int | None
        add_new_axis : bool, default False
            Allow adding the key if missing.
        """
        fw = "numpy" if isinstance(image, np.ndarray) else "torch"
        tag = self.get_tag(image, fw)
        if not tag:
            raise ValueError(f"No tag found for the image (framework: {fw}). Cannot update axis.")

        if key in tag or add_new_axis:
            tag[key] = new_axis
            # persist the update
            self.set_tag(image, fw, tag)
        else:
            raise KeyError(f"Axis key '{key}' not found in the tag.")

    # ====[ TRACK – Attach AxisTracker to Image or return class ]====
    def track(self, image: Optional[ArrayLike] = None):
        """
        Attach an AxisTracker to the image, or return the class when image is None.

        Returns
        -------
        AxisTracker | type[AxisTracker]
        """
        from operators.axis_tracker import AxisTracker

        if image is None:
            return AxisTracker
        return AxisTracker(image, operator=self, framework=self.framework)

    # ====[ Safe Axis Access from Tracked Images ]====
    def get_axis(self, image: ArrayLike, key: str, default: Any = None) -> Optional[int]:
        """
        Retrieve a specific axis from the AxisTracker tag of a given image.

        Returns
        -------
        int | None
        """
        if isinstance(image, np.ndarray):
            fw = "numpy"
        elif isinstance(image, torch.Tensor):
            fw = "torch"
        else:
            raise ValueError("Unsupported input type.")

        if self.has_tag(image, framework=fw):
            tag = self.track(image).get_tag()
            return tag.get(key, default)
        return default

    # ====[ Sync axes from a tagged input ]====
    def sync_axes_from_tag(
        self,
        image: ArrayLike,
        override_axes: bool = False,
        update_layout: bool = True,
        axes=(
            "channel_axis",
            "batch_axis",
            "direction_axis",
            "height_axis",
            "width_axis",
            "depth_axis",
        ),
    ) -> None:
        """
        Sync missing axis configuration from the image tag into this operator.

        Parameters
        ----------
        image : ndarray | Tensor
        override_axes : bool, default False
            If True, override existing values with tag values (when present).
        update_layout : bool, default True
            If True, also copy 'layout_name'.
        """
        fw = "numpy" if isinstance(image, np.ndarray) else "torch"
        tag = self.get_tag(image, fw) if self.has_tag(image, fw) else None
        if not tag:
            return

        for ax in axes:
            if override_axes:
                if ax in tag and tag[ax] is not None:
                    setattr(self, ax, tag[ax])
            else:
                if getattr(self, ax, None) is None and ax in tag:
                    setattr(self, ax, tag.get(ax, None))

        if update_layout and hasattr(self, "layout_name"):
            self.layout_name = tag.get("layout_name", self.layout_name)

    def merge_axes(self, image: ArrayLike, axis1: int, axis2: int) -> ArrayLike:
        """
        Merge two axes into a single axis (e.g., merge direction and batch).

        Returns
        -------
        ndarray | Tensor
            New image with merged axes; tag is updated via AxisTracker.
        """
        tracker = self.track(image)
        shape = list(tracker.image.shape)

        if axis1 >= len(shape) or axis2 >= len(shape):
            raise ValueError("Axis index out of bounds.")

        axis1, axis2 = sorted((axis1, axis2))
        merged_dim = shape[axis1] * shape[axis2]
        new_shape = shape[:axis1] + [merged_dim] + shape[axis2 + 1 :]

        new_img = tracker.image.reshape(new_shape)

        tracker = self.track(new_img)
        tracker.update_tag("merged_axes", [axis1, axis2])
        tracker.update_tag("shape_after", new_shape)

        return tracker.image

    def remove_axis(self, image: ArrayLike, axis: int) -> ArrayLike:
        """
        Remove a specific axis from the image (squeeze operation).

        Returns
        -------
        ndarray | Tensor
        """
        tracker = self.track(image)
        shape = tracker.image.shape

        if shape[axis] != 1:
            raise ValueError(f"Cannot squeeze axis {axis} (dim={shape[axis]}).")

        new_img = (
            np.squeeze(tracker.image, axis=axis)
            if isinstance(image, np.ndarray)
            else torch.squeeze(tracker.image, dim=axis)
        )

        tracker = self.track(new_img)
        tracker.update_tag("removed_axis", axis)
        tracker.update_tag("shape_after", new_img.shape)

        return tracker.image

    def reposition_axis(self, image: ArrayLike, src_axis: int, dst_axis: int) -> ArrayLike:
        """
        Reposition an axis in the image (e.g., move channel axis to first).

        Returns
        -------
        ndarray | Tensor
        """
        return self.track(image).moveaxis(src_axis, dst_axis).image

    # ====[ Tag lock helpers ]====
    def lock_tag(self, image: ArrayLike, framework: Optional[str] = None) -> None:
        """Lock the tag associated with the image to prevent in-place modification."""
        fw = framework or ("torch" if isinstance(image, torch.Tensor) else "numpy")
        tag = self.get_tag(image, fw)
        if tag is not None:
            tag["__locked__"] = True
            self.set_tag(image, fw, tag)

    def unlock_tag(self, image: ArrayLike, framework: Optional[str] = None) -> None:
        """Unlock the tag to allow modifications."""
        fw = framework or ("torch" if isinstance(image, torch.Tensor) else "numpy")
        tag = self.get_tag(image, fw)
        if tag is not None and "__locked__" in tag:
            tag.pop("__locked__", None)
            self.set_tag(image, fw, tag)

    def is_locked(self, image: ArrayLike, framework: Optional[str] = None) -> bool:
        """Return True if the tag is locked (read-only)."""
        fw = framework or ("torch" if isinstance(image, torch.Tensor) else "numpy")
        tag = self.get_tag(image, fw)
        return bool(tag.get("__locked__", False)) if tag else False
        
