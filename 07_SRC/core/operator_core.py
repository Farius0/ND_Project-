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
    def resolve_axis(self, *args, default: Optional[Any] = None) -> Optional[Any]:
        """
        Return the first non-None value among the arguments, falling back to `default`.

        This is a wrapper that delegates to the base class implementation.

        Parameters
        ----------
        *args : Any
            Sequence of candidate values.
        default : Any, optional
            Value to return if all candidates are None.

        Returns
        -------
        Any or None
            First non-None value found, or `default` if none exist.
        """
        return super().resolve_axis(*args, default=default)

    # ====[ ALREADY CONVERTED – Check Conversion Status ]====
    def _already_converted(self, arr: ArrayLike, framework: Literal["numpy", "torch"] = "numpy") -> bool:
        """
        Check if the array has already been converted according to the target framework.

        This method delegates to `BaseConverter.already_converted()`.

        Parameters
        ----------
        arr : ArrayLike
            Input array or tensor to check.
        framework : {'numpy', 'torch'}, optional
            Target framework to check against. Default is 'numpy'.

        Returns
        -------
        bool
            True if the array is already in the expected format, False otherwise.
        """
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
        Convert and tag an image with layout, UID, and axis information.

        This method wraps `BaseConverter.convert_once()` with full control over axes,
        layout validation, tagging, UID handling, and optional normalization.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or Torch tensor) to convert.
        framework : {'numpy', 'torch'}, optional
            Target framework for the converted output. Defaults to self.framework.
        tag_as : str, optional
            Tag to assign as the conversion status (e.g., 'processed').
        direction_axis, batch_axis, channel_axis, height_axis, width_axis, depth_axis : int or None
            Explicit axis positions. Overrides inferred values if provided.
        add_batch_dim : bool or None, optional
            If True, add a batch dimension to the image if missing.
        add_channel_dim : bool or None, optional
            If True, add a channel dimension to the image if missing.
        enable_uid : bool, optional
            If True, assign a unique identifier to the image tag.
        op_params : dict or None, optional
            Additional parameters to include in the tag under 'op_params'.
        track : bool, optional
            If True, attach a tag with layout and status tracking.
        trace_limit : int, optional
            Maximum number of entries in the trace history.
        normalize_override : bool or None, optional
            If True, normalize image values even if already processed.
        force_reset : bool, optional
            If True, ignore existing tags and reprocess the image.
        require_shape_match : bool, optional
            If True, raises an error if image shape does not match expected layout.
        require_layout_match : bool, optional
            If True, raises an error if tag layout does not match expected layout.
        require_uid : bool, optional
            If True, raises an error if the image lacks a UID after processing.
        expected_layout_name : str or None, optional
            Expected layout name (e.g., 'HWC'); used for layout validation.
        require_status : str or None, optional
            Required status tag value (e.g., 'raw') before conversion is allowed.

        Returns
        -------
        ArrayLike
            Converted image (NumPy or Torch) with updated tag.
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
        Convert an image to the configured output format and apply final tagging.

        This is a wrapper around `convert_once()` specifically used for preparing
        the final output of an operator, enforcing tagging, layout validation,
        UID assignment, and output consistency.

        Parameters
        ----------
        image : ArrayLike
            Input image to convert (NumPy array or Torch tensor).
        framework : {'numpy', 'torch'}, optional
            Target backend for the output format. Defaults to self.output_format.
        tag_as : str, optional
            Status tag to assign after conversion (default: "output").
        direction_axis, batch_axis, channel_axis, height_axis, width_axis, depth_axis : int or None
            Optional axis overrides for layout resolution.
        add_batch_dim : bool or None, optional
            Whether to add a batch dimension if missing.
        add_channel_dim : bool or None, optional
            Whether to add a channel dimension if missing.
        enable_uid : bool, optional
            If True, assign a unique identifier to the output image.
        op_params : dict or None, optional
            Additional metadata to attach to the tag.
        track : bool, optional
            Whether to track layout and status using internal tagging.
        trace_limit : int, optional
            Maximum number of recorded trace entries in the tag.
        normalize_override : bool or None, optional
            If True, forces normalization even if already done.
        require_shape_match : bool, optional
            If True, check shape consistency with the expected layout.
        require_layout_match : bool, optional
            If True, enforce layout name match between tag and config.
        require_uid : bool, optional
            If True, raise error if UID is missing after conversion.
        expected_layout_name : str or None, optional
            Expected layout name used for validation.
        require_status : str or None, optional
            Required tag status before allowing conversion (e.g., "processed").

        Returns
        -------
        ArrayLike
            Converted and tagged image in the desired output format.
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
        Convert and tag an image to ensure it matches the target layout and format.

        This method enforces a given layout structure (e.g., 'NCHW', 'GNCDHW'),
        attaches a tag, and wraps the result in an `AxisTracker`.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or Torch tensor) to be formatted.
        tag_as : str, optional
            Status string to assign to the tag (default: "prepared").
        framework : {'numpy', 'torch'}, optional
            Target framework for the output format.
        layout : dict, optional
            Layout dictionary specifying expected axis positions.
        enable_uid : bool, optional
            If True, assign a UID to the tag.
        op_params : dict or None, optional
            Operator-specific parameters to embed in the tag.
        track : bool, optional
            If True, wrap the image in an `AxisTracker`.
        trace_limit : int, optional
            Max length of the trace history recorded in the tag.
        normalize_override : bool or None, optional
            If True, force re-normalization of image values.
        force_reset : bool, optional
            If True, ignore previous tags and reprocess image from scratch.
        require_shape_match : bool, optional
            If True, enforce shape match with the expected layout.
        require_layout_match : bool, optional
            If True, raise error if layout name does not match expected.
        require_uid : bool, optional
            If True, raise error if UID is missing after conversion.
        expected_layout_name : str or None, optional
            Expected layout name to validate against.
        require_status : str or None, optional
            Expected previous status tag before processing.

        Returns
        -------
        AxisTracker
            Image wrapped with an AxisTracker containing layout and tag metadata.
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
        Create a safe copy of a NumPy array or PyTorch tensor while preserving its tag.

        Parameters
        ----------
        obj : ndarray or Tensor
            Input object to copy. Must be a NumPy array or PyTorch tensor.

        Returns
        -------
        ndarray or Tensor
            A cloned copy of the input with its tag (if any) propagated.
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
        Print a summary of the image including shape, dtype, device, framework, and tag information.

        Parameters
        ----------
        image : ndarray or Tensor
            Input image to summarize.
        framework : {'torch', 'numpy'}, optional
            Framework to use for format resolution. Defaults to self.framework.
        verbose : bool, optional
            If True, prints extended information. Default is True.
        as_tensor : bool, optional
            If True, converts image to the target framework before printing.

        Returns
        -------
        None
            The function prints to stdout and does not return a value.
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
        """
        Remove all registered tags from internal storage.

        This is a direct proxy to `BaseConverter.purge_all_tags()`.

        Parameters
        ----------
        verbose : bool, optional
            If True, print a summary of the purge action.

        Returns
        -------
        None
        """
        return super().purge_all_tags(verbose=verbose)

    def purge_tags_for(self, image: np.ndarray | torch.Tensor, verbose: bool = False) -> None:
        """
        Remove the tag associated with a specific image.

        This is a proxy to `BaseConverter.purge_tags_for()`.

        Parameters
        ----------
        image : ndarray or Tensor
            The image whose tag should be removed.
        verbose : bool, optional
            If True, print a message indicating the tag was removed.

        Returns
        -------
        None
        """
        return super().purge_tags_for(image, verbose=verbose)

    def reset_conversion_tag(self, image: ArrayLike) -> None:
        """
        Remove the conversion tag from the given image.

        Parameters
        ----------
        image : ArrayLike
            NumPy array or Torch tensor from which to remove the tag.

        Returns
        -------
        None
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
        Copy the conversion tag from a source image to a target image, possibly across frameworks.

        Parameters
        ----------
        source : ArrayLike
            Source image (NumPy or Torch) from which to copy the tag.
        target : ArrayLike
            Target image (NumPy or Torch) to which the tag will be attached.
        framework_source : {'numpy', 'torch'}, optional
            Framework used to interpret the source image tag.
        framework_target : {'numpy', 'torch'}, optional
            Framework used to attach the tag to the target image.

        Returns
        -------
        ArrayLike
            Target image with the copied tag metadata.
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
        Update the value of a specific axis key in the conversion tag.

        Parameters
        ----------
        image : ndarray or Tensor
            The image whose tag will be updated.
        key : str
            Axis key to update (e.g., 'channel_axis', 'batch_axis').
        new_axis : int or None
            New axis index to assign to the key. If None, the key may be removed or ignored.
        add_new_axis : bool, optional
            If True, allows adding the key if it does not already exist in the tag.

        Returns
        -------
        None
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
        Attach an AxisTracker to the image, or return the AxisTracker class if image is None.

        Parameters
        ----------
        image : ArrayLike or None, optional
            Image to wrap with an AxisTracker. If None, returns the AxisTracker class itself.

        Returns
        -------
        AxisTracker or type[AxisTracker]
            Tracked image instance, or the AxisTracker class object.
        """
        from operators.axis_tracker import AxisTracker

        if image is None:
            return AxisTracker
        return AxisTracker(image, operator=self, framework=self.framework)

    # ====[ Safe Axis Access from Tracked Images ]====
    def get_axis(self, image: ArrayLike, key: str, default: Any = None) -> Optional[int]:
        """
        Retrieve the index of a specific axis from the tag associated with the image.

        Parameters
        ----------
        image : ArrayLike
            Image (NumPy array or Torch tensor) carrying an AxisTracker tag.
        key : str
            Axis key to retrieve (e.g., 'channel_axis', 'height_axis').
        default : Any, optional
            Value to return if the key is not present in the tag.

        Returns
        -------
        int or None
            Axis index if found, otherwise the provided default.
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
        Sync axis configuration from the image tag into the current operator.

        This method copies axis definitions (e.g., 'height_axis') from the image's
        AxisTracker tag into the operator's layout configuration.

        Parameters
        ----------
        image : ndarray or Tensor
            Image from which to read tag metadata.
        override_axes : bool, optional
            If True, existing axis values in the operator will be overwritten by tag values.
        update_layout : bool, optional
            If True, also update the layout name ('layout_name') from the tag.
        axes : tuple of str, optional
            Axis keys to synchronize. Default includes all major semantic axes.

        Returns
        -------
        None
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
        Merge two axes into a single axis in the image (e.g., direction and batch).

        The specified axes are merged by reshaping the image; the corresponding tag
        is automatically updated using AxisTracker.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or Torch tensor) with at least `axis1` and `axis2`.
        axis1 : int
            First axis to merge.
        axis2 : int
            Second axis to merge.

        Returns
        -------
        ArrayLike
            New image with the specified axes merged; the tag is updated accordingly.
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
        Remove a specific axis from the image using a squeeze operation.

        The corresponding tag is updated to reflect the new shape.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or Torch tensor).
        axis : int
            Index of the axis to remove. Must be of size 1.

        Returns
        -------
        ArrayLike
            Image with the specified axis removed and tag updated.
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
        Move a specific axis from one position to another within the image.

        Useful for rearranging axes (e.g., moving the channel axis to the first position).
        The associated tag is updated accordingly.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or Torch tensor).
        src_axis : int
            Index of the axis to move.
        dst_axis : int
            Target position for the axis.

        Returns
        -------
        ArrayLike
            Image with the axis repositioned and tag updated.
        """
        return self.track(image).moveaxis(src_axis, dst_axis).image

    # ====[ Tag lock helpers ]====
    def lock_tag(self, image: ArrayLike, framework: Optional[str] = None) -> None:
        """
        Lock the tag associated with the image to prevent further in-place modifications.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or Torch tensor) whose tag will be locked.
        framework : {'numpy', 'torch'}, optional
            Framework used to resolve the tag location. Defaults to self.framework.

        Returns
        -------
        None
        """
        fw = framework or ("torch" if isinstance(image, torch.Tensor) else "numpy")
        tag = self.get_tag(image, fw)
        if tag is not None:
            tag["__locked__"] = True
            self.set_tag(image, fw, tag)

    def unlock_tag(self, image: ArrayLike, framework: Optional[str] = None) -> None:
        """
        Unlock the tag associated with the image, allowing in-place modifications.

        Parameters
        ----------
        image : ArrayLike
            Input image (NumPy array or Torch tensor) whose tag will be unlocked.
        framework : {'numpy', 'torch'}, optional
            Framework used to resolve the tag location. Defaults to self.framework.

        Returns
        -------
        None
        """
        fw = framework or ("torch" if isinstance(image, torch.Tensor) else "numpy")
        tag = self.get_tag(image, fw)
        if tag is not None and "__locked__" in tag:
            tag.pop("__locked__", None)
            self.set_tag(image, fw, tag)

    def is_locked(self, image: ArrayLike, framework: Optional[str] = None) -> bool:
        """
        Check whether the tag associated with the image is currently locked (read-only).

        Parameters
        ----------
        image : ArrayLike
            Image (NumPy array or Torch tensor) to check.
        framework : {'numpy', 'torch'}, optional
            Framework used to access the tag. Defaults to self.framework.

        Returns
        -------
        bool
            True if the tag is locked, False otherwise.
        """
        fw = framework or ("torch" if isinstance(image, torch.Tensor) else "numpy")
        tag = self.get_tag(image, fw)
        return bool(tag.get("__locked__", False)) if tag else False
        
