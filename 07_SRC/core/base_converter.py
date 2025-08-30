# ==================================================
# =============  MODULE: base_converter  ===========
# ==================================================
from __future__ import annotations

from typing import Optional, Literal, Any, Dict
import uuid

import numpy as np
import torch

from core.tag_registry import get_tag, set_tag, has_tag, del_tag, clear_tags_by_backend
from core.layout_axes import get_layout_axes, LayoutResolver
from core.config import LayoutConfig, GlobalConfig

# Public API
__all__ = ["BaseConverter"]

# Alias for readability
ArrayLike = np.ndarray | torch.Tensor
Framework = Literal["numpy", "torch"]

# ==================================================
# ================ BaseConverter ===================
# ==================================================

class BaseConverter:
    """
    Unified, layout-aware converter with traceable tagging for NumPy/Torch arrays.

    Notes
    -----
    - ND-ready (2D/3D first-class; higher-D when meaningful).
    - Backend-preserving: NumPy in → NumPy out; Torch in → Torch out.
    - Preserves dtype/device/layout when feasible. Normalization only affects
      integer/bool inputs; floating types are returned unchanged.
    """

    def __init__(
        self,
        layout_cfg: LayoutConfig = LayoutConfig(),
        global_cfg: GlobalConfig = GlobalConfig(),
    ) -> None:
        """
        Initialize the converter with layout and global policies.

        Parameters
        ----------
        layout_cfg : LayoutConfig
            Provides axis roles, default layout names, and framework mapping.
        global_cfg : GlobalConfig
            Provides normalization policy, default device, and default flags
            for batch/channel insertion.
        """
        # ====[ Configuration ]====
        self.layout_cfg: LayoutConfig = layout_cfg
        self.global_cfg: GlobalConfig = global_cfg

        # ====[ Axis Resolution – Respect Layout Defaults ]====
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

        # ====[ Framework & Layout Resolution ]====
        self.framework: Framework = self.global_cfg.framework.lower()

        # ====[ General Flags ]====
        self.add_batch_dim: Optional[bool] = self.global_cfg.add_batch_dim
        self.add_channel_dim: Optional[bool] = self.global_cfg.add_channel_dim
        self.normalize: bool = bool(self.global_cfg.normalize)
        self.device: str = (
            "cuda"
            if (torch.cuda.is_available() and self.framework == "torch")
            else self.global_cfg.device
        )

    # ====[ Axis Resolution – Fallback to Defaults ]====
    @staticmethod
    def resolve_axis(*args, default: Optional[Any] = None) -> Optional[Any]:
        """
        Return the first non-None value from the input arguments.

        Parameters
        ----------
        *args : Any
            Sequence of values to check.
        default : Any, optional
            Value to return if all arguments are None.

        Returns
        -------
        Any or None
            The first non-None value in `args`, or `default` if none is found.
        """
        return next((x for x in args if x is not None), default)

    # ===[ NORMALIZATION ]===
    def _normalize(self, arr: ArrayLike) -> ArrayLike:
        """
        Normalize integer/bool inputs to float32 with a safe dynamic range.

        Parameters
        ----------
        arr : ndarray | Tensor
            Integer/bool types are scaled to [0, 1] for unsigned or to [-1, 1] for
            signed types. Floating types are returned unchanged.

        Returns
        -------
        ndarray | Tensor
            Same backend as input, converted to float32 when normalization applies.

        Notes
        -----
        - No-op when `self.normalize` is False.
        - uint8 → /255, uint16 → /65535; bool → {0., 1.}.
        - Signed ints → /max(|x|); if max==0 → zeros float32.
        """
        if not self.normalize:
            return arr

        if isinstance(arr, np.ndarray):
            if arr.dtype == np.bool_:
                return arr.astype(np.float32, copy=False)
            if arr.dtype == np.uint8:
                return arr.astype(np.float32, copy=False) / 255.0
            if arr.dtype == np.uint16:
                return arr.astype(np.float32, copy=False) / 65535.0
            if arr.dtype in (np.int8, np.int16, np.int32, np.int64):
                maxval = np.abs(arr).max()
                return arr.astype(np.float32, copy=False) / maxval if maxval != 0 else arr.astype(
                    np.float32, copy=False
                )

        elif isinstance(arr, torch.Tensor):
            if arr.dtype == torch.bool:
                return arr.to(torch.float32)
            # torch.uint16 exists; getattr handles older torch versions safely
            if arr.dtype in (torch.uint8, getattr(torch, "uint16", torch.uint8)):
                scale = 255.0 if arr.dtype == torch.uint8 else 65535.0
                return arr.to(torch.float32) / scale
            if arr.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                maxval = torch.abs(arr).max()
                max_item = float(maxval.item())
                return arr.to(torch.float32) / max_item if max_item != 0.0 else arr.to(
                    torch.float32
                )

        return arr

    # ====[ Axis Movement – Safe Reordering ]====
    def move_axis(
        self,
        arr: ArrayLike,
        src: int,
        dst: int,
        protected_axes: Optional[list[int]] = None,
    ) -> ArrayLike:
        """
        Safely move one axis while optionally protecting specific positions.

        Parameters
        ----------
        arr : ndarray | Tensor
            Input array or tensor.
        src : int
            Source axis index (supports negatives).
        dst : int
            Destination axis index (supports negatives).
        protected_axes : list[int], optional
            Indices that must keep their position; negative indices allowed.
            Move is rejected if any protected index would change.

        Returns
        -------
        ndarray | Tensor
            Same backend as input, with the axis moved.

        Raises
        ------
        TypeError
            If `arr` is not a NumPy array or Torch tensor.
        ValueError
            If moving the axis would alter a protected index.
        """
        if not isinstance(arr, (np.ndarray, torch.Tensor)):
            raise TypeError("Unsupported array type for axis movement.")

        ndim = arr.ndim
        src = src % ndim
        dst = dst % ndim

        if src == dst:
            return arr

        if protected_axes:
            protected = [(p % ndim) for p in protected_axes]
            axes = list(range(ndim))
            new_axes = axes.copy()
            axis_val = new_axes.pop(src)
            new_axes.insert(dst, axis_val)
            for p in protected:
                if axes[p] != new_axes[p]:
                    raise ValueError(
                        f"Cannot move axis {src} to {dst}: would alter protected axis {p}"
                    )

        return (
            np.moveaxis(arr, src, dst)
            if isinstance(arr, np.ndarray)
            else torch.movedim(arr, src, dst)
        )

    # ====[ Tag Assignment – Apply + Trace ]====
    def tag(
        self,
        arr: ArrayLike,
        framework: Framework,
        status: str = "tagged",
        original_shape: Optional[tuple] = None,
        final_shape: Optional[tuple] = None,
        axis_map: Optional[dict] = None,
        enable_uid: bool = False,
        layout: Optional[dict] = None,
        layout_name: Optional[str] = None,
        op_params: Optional[dict] = None,
        trace_limit: int = 10,
        track: bool = True,
        silent: bool = True,
    ) -> ArrayLike:
        """
        Attach a traceable tag with layout, shapes, axis map, UID (optional), and history.

        Parameters
        ----------
        arr : ndarray | Tensor
            Target array/tensor to annotate.
        framework : {'numpy', 'torch'}
            Backend used for this data (informative only).
        status : str, default 'tagged'
            Stage label to append to the trace (e.g., 'input', 'normalized').
        original_shape : tuple, optional
            Shape before this operation.
        final_shape : tuple, optional
            Shape after this operation (defaults to current `arr.shape`).
        axis_map : dict, optional
            Semantic axis indices (e.g., {'batch': 0, 'channel': 1}).
        enable_uid : bool, default False
            Whether to attach a new short unique identifier.
        layout : dict, optional
            Pre-resolved layout dictionary.
        layout_name : str, optional
            Human-readable layout string (e.g., 'NCHW').
        op_params : dict, optional
            Free-form operator metadata to help tracing/debugging.
        trace_limit : int, default 10
            Maximum number of stored trace events.
        track : bool, default True
            If False, returns `arr` unchanged (no tag applied).
        silent : bool, default True
            If False, re-raises errors from the underlying tag registry.

        Returns
        -------
        ndarray | Tensor
            Same object with its tag updated (if `track=True`).
        """
        if not track:
            return arr

        previous = get_tag(arr) or {}
        prev_trace = previous.get("trace", [])
        trace = (prev_trace + [status])[-trace_limit:] if status else prev_trace

        tag = {
            "status": status,
            "converted": True,
            "framework": framework,
            "device": str(self.device),
            "original_shape": original_shape,
            "shape_after": final_shape or getattr(arr, "shape", None),
            "channel_axis_original": self.channel_axis,
            "layout": layout or self.layout,
            "layout_name": layout_name or self.layout_name,
            "axis_map": axis_map or {},
            "trace": trace,
            "op_params": op_params or {},
        }

        # Enrich + UID BEFORE applying the tag, so UID is persisted
        layout_data = self.tag_from_layout(self.resolve_axis(layout, self.layout))
        tag.update(layout_data)
        if enable_uid:
            tag["uid"] = uuid.uuid4().hex[:8]

        try:
            set_tag(arr, tag)
        except Exception as e:
            if not silent:
                raise e

        return arr

    # ====[ Tag Generation – From Layout Dictionary ]====
    @staticmethod
    def tag_from_layout(layout: dict) -> dict:
        """
        Build a standard axis dictionary from a layout description.

        Parameters
        ----------
        layout : dict
            Output of `get_layout_axes()` / `LayoutResolver`.

        Returns
        -------
        dict
            Keys among {'batch_axis','channel_axis','direction_axis',
                        'height_axis','width_axis','depth_axis','ndim'}.
        """
        if not isinstance(layout, dict):
            raise TypeError("Layout must be a dictionary with axis definitions.")
        keys = [
            "batch_axis",
            "channel_axis",
            "direction_axis",
            "height_axis",
            "width_axis",
            "depth_axis",
            "ndim",
        ]
        return {k: layout.get(k) for k in keys}

    # ====[ Tag Tracing ]====
    @staticmethod
    def trace_tag_event(
        arr: ArrayLike, event: str, trace_limit: int = 10
    ) -> None:
        """
        Append a trace event to the tag and re-apply the tag.

        Parameters
        ----------
        arr : ndarray | Tensor
            Target array/tensor.
        event : str
            Event name to append to the trace.
        trace_limit : int, default 10
            Maximum number of stored trace events.
        """
        if not isinstance(arr, (np.ndarray, torch.Tensor)):
            raise TypeError("Input must be a NumPy array or Torch tensor.")

        tag = get_tag(arr) or {}
        trace = list(tag.get("trace", []))
        if not trace or trace[-1] != event:
            trace.append(event)
        tag["trace"] = trace[-trace_limit:]
        set_tag(arr, tag)

    def assert_valid_tag(
        self,
        arr: ArrayLike,
        framework: Framework = "numpy",
        required_keys: Optional[list] = None,
        check_uid: bool = True,
        check_shape: bool = True,
        check_layout: bool = True,
        verbose: bool = False,
        raise_on_fail: bool = True,
    ) -> bool:
        """
        Validate that the attached tag contains required keys and consistent shapes/layout.

        Parameters
        ----------
        arr : ndarray | Tensor
            Data structure whose tag is validated.
        framework : {'numpy','torch'}, default 'numpy'
            Expected framework in the tag (informative only).
        required_keys : list[str], optional
            Keys to enforce; defaults to a minimal set.
        check_uid : bool, default True
            Require the presence of a UID.
        check_shape : bool, default True
            Verify that recorded `shape_after` matches `arr.shape`.
        check_layout : bool, default True
            Require valid `layout` and `layout_name`.
        verbose : bool, default False
            Print diagnostics on failure.
        raise_on_fail : bool, default True
            Raise ValueError on failure; otherwise return False.

        Returns
        -------
        bool
            True if tag is valid; False (or exception) otherwise.
        """
        tag = self.get_tag(arr, framework)
        shape = getattr(arr, "shape", None)
        errors: list[str] = []

        if not tag:
            msg = "Tag is missing."
            if verbose:
                print("[TagCheck]", msg)
            if raise_on_fail:
                raise ValueError(msg)
            return False

        keys_to_check = required_keys or [
            "status",
            "converted",
            "framework",
            "shape_after",
            "axis_map",
        ]
        for k in keys_to_check:
            if k not in tag:
                errors.append(f"Missing tag key: '{k}'")

        if check_uid and "uid" not in tag:
            errors.append("Missing UID.")

        if check_shape and tag.get("shape_after") != shape:
            errors.append(
                f"Shape mismatch: tag shape_after={tag.get('shape_after')} vs array shape={shape}"
            )

        if check_layout and ("layout" not in tag or "layout_name" not in tag):
            errors.append("Missing layout or layout_name in tag.")

        if errors:
            if verbose:
                print("[TagCheck] Tag validation failed:")
                for e in errors:
                    print("  -", e)
            if raise_on_fail:
                raise ValueError("Invalid tag: " + "; ".join(errors))
            return False

        if verbose:
            print("[TagCheck] Tag is valid.")
        return True

    # ====[ Tag Retrieval ]====
    @staticmethod
    def get_tag(arr: ArrayLike, framework: Framework) -> dict:
        """
        Retrieve the conversion tag from a NumPy array or Torch tensor.

        Parameters
        ----------
        arr : ndarray | Tensor
            Target array/tensor.
        framework : {'numpy','torch'}
            Informative only; used for sanity validation (no routing decision).

        Returns
        -------
        dict
            Tag dictionary (or {} if absent).
        """
        if framework not in {"numpy", "torch"}:
            raise ValueError(f"Unsupported framework '{framework}'.")
        return get_tag(arr) or {}

    # ====[ Tag Existence Check ]====
    @staticmethod
    def has_tag(arr: ArrayLike, framework: Framework) -> bool:
        """
        Check whether the array has a conversion tag.

        Parameters
        ----------
        arr : ndarray | Tensor
            Target array/tensor.
        framework : {'numpy','torch'}
            Informative only; used for sanity validation (no routing decision).

        Returns
        -------
        bool
            True if tag exists, False otherwise.
        """
        if framework not in {"numpy", "torch"}:
            raise ValueError(f"Unsupported framework '{framework}'.")
        return has_tag(arr)

    # ====[ Tag Setting ]====
    @staticmethod
    def set_tag(arr: ArrayLike, framework: Framework, tag: dict) -> None:
        """
        Set a conversion tag on a NumPy array or Torch tensor.

        Parameters
        ----------
        arr : ndarray | Tensor
            Target array/tensor.
        framework : {'numpy','torch'}
            Informative only; used for sanity validation (no routing decision).
        tag : dict
            Tag content.
        """
        if framework not in {"numpy", "torch"}:
            raise ValueError(f"Unsupported framework '{framework}'.")
        set_tag(arr, tag)

    # ====[ Tag Deletion ]====
    @staticmethod
    def del_tag(arr: ArrayLike, framework: Literal["numpy", "torch"]) -> None:
        """
        Delete the conversion tag from a NumPy array or Torch tensor.

        Parameters
        ----------
        arr : ndarray | Tensor
            Target array/tensor.
        framework : {'numpy','torch'}
            Informative only; used for sanity validation (no routing decision).
        """
        if framework not in {"numpy", "torch"}:
            raise ValueError(f"Unsupported framework '{framework}'.")
        del_tag(arr)

    # ====[ UID Retrieval ]====
    def get_uid(
        self, arr: ArrayLike, framework: Framework
    ) -> Optional[str]:
        """
        Return the UID from the tag if available.

        Parameters
        ----------
        arr : ndarray | Tensor
            Target array/tensor.
        framework : {'numpy','torch'}
            Informative only; used for sanity validation (no routing decision).

        Returns
        -------
        str | None
            UID string if present, else None.
        """
        tag = self.get_tag(arr, framework)
        return tag.get("uid", None)

    # ====[ AXIS RETRIEVAL – get_axis() ]====
    def get_axis(
        self,
        arr: ArrayLike,
        role: str,
        framework: Framework,
    ) -> Optional[int]:
        """
        Return the index of a semantic axis (e.g., 'channel', 'batch', 'height').

        Parameters
        ----------
        arr : ndarray | Tensor
            Target array/tensor.
        role : {'channel','batch','direction','height','width','depth'}
            Semantic role to look up.
        framework : {'numpy','torch'}
            Backend expected in the tag (informative only).

        Returns
        -------
        int | None
            Axis index, if available in `axis_map` or in `<role>_axis` fields.
        """
        if role not in {"channel", "batch", "direction", "height", "width", "depth"}:
            raise ValueError(
                "Unsupported role. Expected one of "
                "{'channel','batch','direction','height','width','depth'}."
            )

        if not isinstance(arr, (np.ndarray, torch.Tensor)):
            raise TypeError("Input must be a NumPy array or Torch tensor.")

        tag = self.get_tag(arr, framework)
        axis = tag.get("axis_map", {}).get(role, None)
        if axis is None:
            axis = tag.get(f"{role}_axis", None)
        return axis

    # ====[ TAG SUMMARY – get_tag_summary() ]====
    def get_tag_summary(
        self, arr: ArrayLike, framework: Framework
    ) -> dict:
        """
        Summarize core tag fields for debugging.

        Parameters
        ----------
        arr : ndarray | Tensor
            Target array/tensor.
        framework : {'numpy','torch'}
            Backend expected in the tag (informative only).

        Returns
        -------
        dict
            {shape, framework, status, uid, axes{...}, trace}.
        """
        if framework not in {"numpy", "torch"}:
            raise ValueError(f"Unsupported framework '{framework}'.")
        if not isinstance(arr, (np.ndarray, torch.Tensor)):
            raise TypeError("Input must be a NumPy array or Torch tensor.")

        tag = self.get_tag(arr, framework)
        return {
            "shape": getattr(arr, "shape", None),
            "framework": tag.get("framework"),
            "status": tag.get("status"),
            "uid": tag.get("uid"),
            "axes": {
                "channel": tag.get("channel_axis"),
                "batch": tag.get("batch_axis"),
                "direction": tag.get("direction_axis"),
                "height": tag.get("height_axis"),
                "width": tag.get("width_axis"),
                "depth": tag.get("depth_axis"),
            },
            "trace": tag.get("trace", []),
        }

    # ====[ TAG COPY – copy_tag() ]====
    def copy_tag(
        self,
        source: ArrayLike,
        target: ArrayLike,
        update_dict: Optional[dict] = None,
    ) -> None:
        """
        Copy the tag from one array to another, optionally updating fields.

        Parameters
        ----------
        source : ndarray | Tensor
            Array/tensor to read the tag from.
        target : ndarray | Tensor
            Array/tensor to receive the tag.
        update_dict : dict, optional
            Keys to update/override in the copied tag.
        """
        if not isinstance(source, (np.ndarray, torch.Tensor)) or not isinstance(
            target, (np.ndarray, torch.Tensor)
        ):
            raise TypeError(
                "Both source and target must be NumPy arrays or Torch tensors."
            )

        tag = (get_tag(source) or {}).copy()
        if update_dict:
            tag.update(update_dict)
        set_tag(target, tag)

    # ====[ TAG PURGE – purge_all_tags() ]====
    def purge_all_tags(self, verbose: bool = False) -> None:
        """
        Clear all existing tag records (NumPy and Torch) from the registry.

        Parameters
        ----------
        verbose : bool, default False
            If True, prints a confirmation message.
        """
        for backend in ["torch", "numpy"]:
            clear_tags_by_backend(backend)
        if verbose or getattr(self, "verbose", False):
            print("[purge_all_tags] All tags have been cleared.")

    # ====[ TAG PURGE – purge_tags_for() ]====
    def purge_tags_for(self, image: ArrayLike, verbose: bool = False) -> None:
        """
        Remove all tags associated with a specific image, both NumPy and Torch.

        Parameters
        ----------
        image : ndarray | Tensor
            The image or tensor whose tags should be removed.
        verbose : bool, default False
            If True, prints a confirmation message.
        """
        for fw in ["numpy", "torch"]:
            if self.has_tag(image, fw):
                try:
                    self.del_tag(image, fw)
                    if verbose or getattr(self, "verbose", False):
                        print(f"[purge_tags_for] Tag removed from {fw} image.")
                except Exception as e:
                    if verbose or getattr(self, "verbose", False):
                        print(
                            f"[purge_tags_for] Failed to remove tag from {fw} image: {e}"
                        )

    # ====[ CONVERSION CHECK – already_converted() ]====
    def already_converted(
        self,
        arr: ArrayLike,
        framework: Framework = "numpy",
        require_shape_match: bool = True,
        require_layout_match: bool = False,
        require_uid: bool = False,
        expected_layout_name: Optional[str] = None,
        require_status: Optional[str] = None,
        verbose: bool = False,
    ) -> bool:
        """
        Check if `arr` already matches the expected converted state.

        Parameters
        ----------
        arr : ndarray | Tensor
            Target data.
        framework : {'numpy','torch'}, default 'numpy'
            Expected backend recorded in the tag.
        require_shape_match : bool, default True
            Require that `shape_after == arr.shape`.
        require_layout_match : bool, default False
            Require that `layout_name` matches `expected_layout_name`.
        expected_layout_name : str, optional
            Layout string to enforce (e.g., 'NCHW').
        require_uid : bool, default False
            Require a UID presence.
        require_status : str, optional
            Require an exact status string match (e.g., 'input').
        verbose : bool, default False
            Print detailed mismatch reasons.

        Returns
        -------
        bool
            True if the current array matches the expected converted state.
        """
        if not isinstance(arr, (np.ndarray, torch.Tensor)):
            raise TypeError("Input must be a NumPy array or Torch tensor.")
        if framework not in {"numpy", "torch"}:
            raise ValueError(f"Unsupported framework '{framework}'.")

        tag = self.get_tag(arr, framework)
        shape = getattr(arr, "shape", None)

        if not tag:
            if verbose:
                print("[ConvertCheck] No tag found.")
            return False
        if not tag.get("converted", False):
            if verbose:
                print("[ConvertCheck] Tag exists but not marked as 'converted=True'.")
            return False
        if tag.get("framework") != framework:
            if verbose:
                print(
                    f"[ConvertCheck] Framework mismatch: {tag.get('framework')} ≠ {framework}"
                )
            return False
        if require_shape_match and tag.get("shape_after") != shape:
            if verbose:
                print(
                    f"[ConvertCheck] Shape mismatch: {tag.get('shape_after')} ≠ {shape}"
                )
            return False
        if require_layout_match:
            layout_name = tag.get("layout_name", None)
            if expected_layout_name and layout_name != expected_layout_name:
                if verbose:
                    print(
                        f"[ConvertCheck] Layout mismatch: expected '{expected_layout_name}' ≠ tag '{layout_name}'"
                    )
                return False
        if require_uid and not tag.get("uid"):
            return False

        if require_status:
            status = tag.get("status")
            if status != require_status:
                if verbose:
                    print(
                        f"[ConvertCheck] Status mismatch: got '{status}', expected '{require_status}'"
                    )
                return False

        return True

    # ====[ NUMPY → TORCH CONVERSION – Axis-aware, Tagged ]====
    def numpy_to_tensor(
        self,
        array: np.ndarray,
        direction_axis: Optional[int] = None,
        add_batch_dim: Optional[bool] = None,
        add_channel_dim: Optional[bool] = None,
        batch_axis: Optional[int] = None,
        channel_axis: Optional[int] = None,
        depth_axis: Optional[int] = None,
        height_axis: Optional[int] = None,
        width_axis: Optional[int] = None,
        status: str = "input",
        enable_uid: bool = False,
        op_params: Optional[dict] = None,
        track: bool = True,
        trace_limit: int = 10,
    ) -> torch.Tensor:
        """
        Convert a NumPy array into a Torch tensor with layout-aware axis handling and tagging.

        Parameters
        ----------
        array : ndarray
            Input array.
        direction_axis : int, optional
            Optional index for direction axis.
        add_batch_dim : bool, optional
            If True, adds a singleton batch dimension when absent.
        add_channel_dim : bool, optional
            If True, adds a singleton channel dimension when absent.
        batch_axis, channel_axis, depth_axis, height_axis, width_axis : int, optional
            Axis positions if known; otherwise resolved from layout/tag.
        status : str, default 'input'
            Tag status.
        enable_uid : bool, default False
            Attach a UID to the tag.
        op_params : dict, optional
            Operator metadata for tracing.
        track : bool, default True
            Whether to apply tagging.
        trace_limit : int, default 10
            Max number of stored trace events.

        Returns
        -------
        Tensor
            Tensor on the configured device with an attached tag.

        Notes
        -----
        - Dtype policy:
          - If `normalize=True`, outputs are float32.
          - If `normalize=False`, dtype is preserved when feasible
            (e.g., NumPy uint8 → Torch uint8, float32 ↔ float32);
            otherwise, fallback is float32.
        """
        # ====[ Early Exit: Already Tensor ]====
        if isinstance(array, torch.Tensor):
            return self._normalize(array.to(self.device))

        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a NumPy array.")

        # ====[ Extract or Default Axes from Tag ]====
        if self.has_tag(array, "numpy"):
            tag = self.get_tag(array, "numpy")
            channel_axis = self.resolve_axis(channel_axis, tag.get("channel_axis"))
            direction_axis = self.resolve_axis(
                direction_axis, tag.get("direction_axis")
            )
            batch_axis = self.resolve_axis(batch_axis, tag.get("batch_axis"))
            depth_axis = self.resolve_axis(depth_axis, tag.get("depth_axis"))
            height_axis = self.resolve_axis(height_axis, tag.get("height_axis"))
            width_axis = self.resolve_axis(width_axis, tag.get("width_axis"))
        else:
            channel_axis = self.resolve_axis(channel_axis, self.channel_axis)
            direction_axis = self.resolve_axis(direction_axis, self.direction_axis)
            batch_axis = self.resolve_axis(batch_axis, self.batch_axis)
            depth_axis = self.resolve_axis(depth_axis, self.depth_axis)
            height_axis = self.resolve_axis(height_axis, self.height_axis)
            width_axis = self.resolve_axis(width_axis, self.width_axis)

        add_batch_dim = self.resolve_axis(add_batch_dim, self.add_batch_dim)
        add_channel_dim = self.resolve_axis(add_channel_dim, self.add_channel_dim)

        # ====[ Normalize + Convert to Torch ]====
        array = self._normalize(array)
        original_shape = array.shape

        tensor = torch.from_numpy(array)
        if self.normalize:
            # Normalized outputs are float32 by design.
            tensor = tensor.to(torch.float32)
        else:
            # Preserve dtype when feasible; documented fallback is float32.
            if array.dtype == np.float64:
                tensor = tensor.to(torch.float64)
            elif array.dtype == np.float32:
                tensor = tensor.to(torch.float32)
            elif array.dtype == np.uint8:
                tensor = tensor.to(torch.uint8)
            elif array.dtype == np.bool_:
                tensor = tensor.to(torch.bool)
            else:
                tensor = tensor.to(torch.float32)
        tensor = tensor.to(self.device)

        # ====[ Axis Insertion + Mapping ]====
        axis_map: dict[str, Optional[int]] = {}
        insert_index = 0
        channel_index_update = 0

        if direction_axis is not None:
            axis_map["direction"] = insert_index
            insert_index += 2 if isinstance(direction_axis, tuple) else 1  # Hessian/Grad
            if channel_axis is not None or add_channel_dim is True:
                channel_index_update += 1 if isinstance(direction_axis, tuple) else 0

        if batch_axis is not None:
            axis_map["batch"] = insert_index
            insert_index += 1
        elif add_batch_dim:
            tensor = tensor.unsqueeze(insert_index)
            axis_map["batch"] = insert_index
            insert_index += 1
            if channel_axis is not None or add_channel_dim is True:
                channel_index_update += 1

        if channel_axis is not None:
            src_axis = (channel_axis % array.ndim) + channel_index_update
            if src_axis >= tensor.ndim:
                raise ValueError(
                    f"Invalid channel axis {src_axis} for tensor ndim {tensor.ndim}"
                )
            tensor = self.move_axis(tensor, src_axis, insert_index)
            axis_map["channel"] = insert_index
        elif add_channel_dim:
            tensor = tensor.unsqueeze(insert_index)
            axis_map["channel"] = insert_index

        # ====[ Layout Reconstruction ]====
        layout_str = LayoutResolver.from_flags(
            direction=(direction_axis is not None),
            batch=(batch_axis is not None or add_batch_dim is True),
            channel=(channel_axis is not None or add_channel_dim is True),
            depth=(depth_axis is not None),
            height=(height_axis is not None),
            width=(width_axis is not None),
            channel_position="first",
        )
        layout = get_layout_axes("torch", layout_str)

        # ====[ Final Tag Application ]====
        return self.tag(
            arr=tensor,
            framework="torch",
            status=status,
            original_shape=original_shape,
            final_shape=tensor.shape,
            layout=layout,
            layout_name=layout_str,
            axis_map=axis_map,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
        )

    # ====[ TORCH → NUMPY CONVERSION – Axis-aware, Tagged ]====
    def tensor_to_numpy(
        self,
        tensor: torch.Tensor,
        direction_axis: Optional[int] = None,
        add_batch_dim: Optional[bool] = None,
        add_channel_dim: Optional[bool] = None,
        batch_axis: Optional[int] = None,
        channel_axis: Optional[int] = None,
        height_axis: Optional[int] = None,
        width_axis: Optional[int] = None,
        depth_axis: Optional[int] = None,
        remove_batch_dim: bool = True,
        remove_channel_dim: bool = True,
        status: str = "input",
        enable_uid: bool = False,
        op_params: Optional[dict] = None,
        track: bool = True,
        trace_limit: int = 10,
    ) -> np.ndarray:
        """
        Convert a Torch tensor into a NumPy array, preserving layout and trace.

        Parameters
        ----------
        tensor : Tensor
            Input tensor.
        direction_axis, batch_axis, channel_axis : int, optional
            Axes to restore when reconstructing a layout.
        remove_batch_dim : bool, default True
            If True, remove singleton batch if present.
        remove_channel_dim : bool, default True
            If True, remove singleton channel if present.
        status : str, default 'input'
            Tag status.
        enable_uid : bool, default False
            Attach a new UID.
        track : bool, default True
            Whether to tag.
        trace_limit : int, default 10
            Max trace steps.

        Returns
        -------
        ndarray
            NumPy array with an attached tag.

        Notes
        -----
        - Dtype policy:
          - If `normalize=True`, outputs are float32.
          - If `normalize=False`, dtype is preserved when feasible; otherwise
            fallback is float32 (handled upstream).
        """
        # ====[ Type Check ]====
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor.")

        # ====[ Extract Tag Info ]====
        if self.has_tag(tensor, "torch"):
            tag = self.get_tag(tensor, "torch")
            channel_axis_original = tag.get("channel_axis_original", channel_axis)
            batch_axis = self.resolve_axis(batch_axis, tag.get("batch_axis"))
            direction_axis = self.resolve_axis(
                direction_axis, tag.get("direction_axis")
            )
            depth_axis = self.resolve_axis(depth_axis, tag.get("depth_axis"))
            height_axis = self.resolve_axis(height_axis, tag.get("height_axis"))
            width_axis = self.resolve_axis(width_axis, tag.get("width_axis"))
        else:
            channel_axis_original = self.resolve_axis(channel_axis, self.channel_axis)
            batch_axis = self.resolve_axis(batch_axis, self.batch_axis)
            direction_axis = self.resolve_axis(direction_axis, self.direction_axis)
            depth_axis = self.resolve_axis(depth_axis, self.depth_axis)
            height_axis = self.resolve_axis(height_axis, self.height_axis)
            width_axis = self.resolve_axis(width_axis, self.width_axis)

        add_batch_dim = self.resolve_axis(add_batch_dim, self.add_batch_dim)
        add_channel_dim = self.resolve_axis(add_channel_dim, self.add_channel_dim)

        # ====[ Convert to NumPy ]====
        if tensor.requires_grad:
            tensor = tensor.detach()
        array = tensor.cpu().numpy()
        original_shape = array.shape

        # ====[ Axis Mapping Reconstruction ]====
        axis_map: dict[str, Optional[int]] = {}
        current_axis = 0

        if direction_axis is not None:
            axis_map["direction"] = current_axis
            current_axis += 2 if isinstance(direction_axis, tuple) else 1  # Hessian/Grad

        if batch_axis is not None or add_batch_dim:
            batch_index = current_axis
            if remove_batch_dim and array.shape[batch_index] == 1:
                array = np.squeeze(array, axis=batch_index)
                axis_map["batch"] = None
            else:
                axis_map["batch"] = batch_index
                current_axis += 1

        if channel_axis_original is not None:
            src_axis = current_axis
            dst_axis = (channel_axis_original + current_axis) % array.ndim
            array = self.move_axis(array, src_axis, dst_axis)
            axis_map["channel"] = dst_axis
        elif add_channel_dim:
            channel_index = current_axis
            if remove_channel_dim and array.shape[channel_index] == 1:
                array = np.squeeze(array, axis=channel_index)
                axis_map["channel"] = None
            else:
                axis_map["channel"] = channel_index

        # ====[ Layout Reconstruction ]====
        layout_str = LayoutResolver.from_flags(
            direction=(direction_axis is not None),
            batch=(axis_map.get("batch") is not None),
            channel=(axis_map.get("channel") is not None),
            depth=(depth_axis is not None),
            height=(height_axis is not None),
            width=(width_axis is not None),
            channel_position="last",
        )
        layout = get_layout_axes("numpy", layout_str)

        # ====[ Final Tag Application ]====
        return self.tag(
            arr=array,
            framework="numpy",
            status=status,
            original_shape=original_shape,
            final_shape=array.shape,
            layout=layout,
            layout_name=layout_str,
            axis_map=axis_map,
            enable_uid=enable_uid,
            op_params=op_params,
            track=track,
            trace_limit=trace_limit,
        )

    # ====[ UNIVERSAL CONVERTER – to_framework() ]====
    def to_framework(
        self,
        image: ArrayLike,
        framework: Framework = "numpy",
        status: str = "input",
        direction_axis: Optional[int] = None,
        add_batch_dim: Optional[bool] = None,
        add_channel_dim: Optional[bool] = None,
        batch_axis: Optional[int] = None,
        channel_axis: Optional[int] = None,
        height_axis: Optional[int] = None,
        width_axis: Optional[int] = None,
        depth_axis: Optional[int] = None,
        enable_uid: bool = False,
        op_params: Optional[dict] = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: Optional[bool] = None,
        require_shape_match: bool = True,
        require_layout_match: bool = False,
        require_uid: bool = False,
        expected_layout_name: Optional[str] = None,
        require_status: Optional[str] = None,
    ) -> ArrayLike:
        """
        Convert an image to the desired framework ('numpy' or 'torch'),
        preserving layout and attaching a traceable tag.

        Parameters
        ----------
        image : ndarray | Tensor
            Input image.
        framework : {'torch','numpy'}, default 'numpy'
            Target backend.
        status : str, default 'input'
            Tag status string.
        direction_axis, batch_axis, channel_axis, height_axis, width_axis, depth_axis : int, optional
            Optional overrides; otherwise resolved from tag/layout.
        enable_uid : bool, default False
            Attach UID to the tag.
        op_params : dict, optional
            Operator metadata for tracing.
        track : bool, default True
            Whether to apply tag.
        trace_limit : int, default 10
            Trace history depth.
        normalize_override : bool, optional
            Force/disable normalization regardless of config.
        require_* : see `already_converted()` for semantics.
        expected_layout_name : str, optional
            Required layout name when `require_layout_match=True`.

        Returns
        -------
        ndarray | Tensor
            Data in the target framework with a tag.

        Notes
        -----
        - Dim insertion policy (when flags are True):
          - Torch target: channel-first; batch inserted before channel if both are missing.
          - NumPy target: channel-last; batch inserted before channel if both are missing.
        """
        # ====[ Skip Conversion If Already Done ]====
        if self.already_converted(
            image,
            framework,
            expected_layout_name=expected_layout_name,
            require_shape_match=require_shape_match,
            require_layout_match=require_layout_match,
            require_uid=require_uid,
            require_status=require_status,
        ):
            return image

        fw = "torch" if isinstance(image, torch.Tensor) else "numpy"

        if self.has_tag(image, fw):
            tag = self.get_tag(image, fw)
            direction_axis = tag.get("direction_axis", None)
            batch_axis = tag.get("batch_axis", None)
            channel_axis = tag.get("channel_axis", None)
            height_axis = tag.get("height_axis", None)
            width_axis = tag.get("width_axis", None)
            depth_axis = tag.get("depth_axis", None)
        else:
            # ====[ Axis Fallbacks ]====
            direction_axis = self.direction_axis
            batch_axis = self.batch_axis
            channel_axis = self.channel_axis
            height_axis = self.height_axis
            width_axis = self.width_axis
            depth_axis = self.depth_axis

        add_batch_dim = add_batch_dim or self.add_batch_dim
        add_channel_dim = add_channel_dim or self.add_channel_dim

        # ====[ Layout Reconstruction (shared) ]====
        layout_str = LayoutResolver.from_flags(
            direction=(direction_axis is not None),
            batch=(batch_axis is not None or add_batch_dim is True),
            channel=(channel_axis is not None or add_channel_dim is True),
            depth=(depth_axis is not None),
            height=(height_axis is not None),
            width=(width_axis is not None),
            channel_position="first" if framework == "torch" else "last",
        )
        layout = get_layout_axes(framework, layout_str)

        # ====[ TORCH Target ]====
        if framework == "torch":
            if isinstance(image, np.ndarray):
                return self.numpy_to_tensor(
                    array=image,
                    status=status,
                    direction_axis=direction_axis,
                    batch_axis=batch_axis,
                    channel_axis=channel_axis,
                    height_axis=height_axis,
                    width_axis=width_axis,
                    depth_axis=depth_axis,
                    add_batch_dim=add_batch_dim,
                    add_channel_dim=add_channel_dim,
                    enable_uid=enable_uid,
                    op_params=op_params,
                    track=track,
                    trace_limit=trace_limit,
                )
            elif isinstance(image, torch.Tensor):
                arr = image.to(self.device)
                if normalize_override is True or (
                    normalize_override is None and self.normalize
                ):
                    arr = self._normalize(arr)

                if add_channel_dim is True and len(layout_str) != arr.ndim:
                    arr = arr.unsqueeze(0)

                if add_batch_dim is True and len(layout_str) != arr.ndim:
                    arr = arr.unsqueeze(0)

                return self.tag(
                    arr=arr,
                    framework="torch",
                    status=status,
                    original_shape=image.shape,
                    final_shape=arr.shape,
                    layout=layout,
                    layout_name=layout_str,
                    axis_map={},
                    enable_uid=enable_uid,
                    op_params=op_params,
                    track=track,
                    trace_limit=trace_limit,
                )
            else:
                raise TypeError("Unsupported input type for torch conversion.")

        # ====[ NUMPY Target ]====
        elif framework == "numpy":
            if isinstance(image, torch.Tensor):
                return self.tensor_to_numpy(
                    tensor=image,
                    status=status,
                    direction_axis=direction_axis,
                    batch_axis=batch_axis,
                    channel_axis=channel_axis,
                    height_axis=height_axis,
                    width_axis=width_axis,
                    depth_axis=depth_axis,
                    add_batch_dim=add_batch_dim,
                    add_channel_dim=add_channel_dim,
                    enable_uid=enable_uid,
                    op_params=op_params,
                    track=track,
                    trace_limit=trace_limit,
                )
            elif isinstance(image, np.ndarray):
                arr = image
                if normalize_override is True or (
                    normalize_override is None and self.normalize
                ):
                    arr = self._normalize(arr)

                if add_channel_dim is True and len(layout_str) != arr.ndim:
                    arr = np.expand_dims(arr, axis=0)

                if add_batch_dim is True and len(layout_str) != arr.ndim:
                    arr = np.expand_dims(arr, axis=0)

                return self.tag(
                    arr=arr,
                    framework="numpy",
                    status=status,
                    original_shape=image.shape,
                    final_shape=arr.shape,
                    layout=layout,
                    layout_name=layout_str,
                    axis_map={},
                    enable_uid=enable_uid,
                    op_params=op_params,
                    track=track,
                    trace_limit=trace_limit,
                )
            else:
                raise TypeError("Unsupported input type for NumPy conversion.")

        # ====[ Invalid Framework ]====
        else:
            raise ValueError("Unsupported framework. Use 'numpy' or 'torch'.")

    # ====[ ENTRY POINT – convert_once() ]====
    def convert_once(
        self,
        image: ArrayLike,
        framework: Framework = "torch",
        tag_as: str = "input",
        direction_axis: Optional[int] = None,
        batch_axis: Optional[int] = None,
        add_batch_dim: Optional[bool] = None,
        add_channel_dim: Optional[bool] = None,
        channel_axis: Optional[int] = None,
        height_axis: Optional[int] = None,
        width_axis: Optional[int] = None,
        depth_axis: Optional[int] = None,
        enable_uid: bool = False,
        op_params: Optional[dict] = None,
        track: bool = True,
        trace_limit: int = 10,
        normalize_override: Optional[bool] = None,
        require_shape_match: bool = True,
        require_layout_match: bool = False,
        require_uid: bool = False,
        expected_layout_name: Optional[str] = None,
        require_status: Optional[str] = None,
    ) -> ArrayLike:
        """
        Entry point for single, traceable image conversion.

        Parameters
        ----------
        image : ndarray | Tensor
            Input image.
        framework : {'torch','numpy'}, default 'torch'
            Target framework.
        tag_as : str, default 'input'
            Status string recorded in the tag.
        direction_axis, batch_axis, channel_axis, height_axis, width_axis, depth_axis : int, optional
            Optional axis overrides.
        enable_uid : bool, default False
            Attach UID to the tag.
        op_params : dict, optional
            Extra metadata.
        track : bool, default True
            Whether to attach a tag.
        trace_limit : int, default 10
            Trace history depth.
        normalize_override : bool, optional
            Force/disable normalization.
        require_* / expected_layout_name : see `already_converted()`.

        Returns
        -------
        ndarray | Tensor
            Data converted to the desired framework with an enriched tag.
        """
        return self.to_framework(
            image,
            framework=framework,
            status=tag_as,
            direction_axis=direction_axis,
            batch_axis=batch_axis,
            add_batch_dim=add_batch_dim,
            add_channel_dim=add_channel_dim,
            channel_axis=channel_axis,
            height_axis=height_axis,
            width_axis=width_axis,
            depth_axis=depth_axis,
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
