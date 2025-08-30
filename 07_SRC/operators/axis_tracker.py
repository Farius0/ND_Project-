# ==================================================
# =============  MODULE: axis_tracker  =============
# ==================================================
from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, Callable, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from core.tag_registry import get_tag, set_tag, has_tag
from core.layout_axes import get_layout_axes, LayoutResolver

# Public API
__all__ = ["AxisTracker"]

ArrayLike = Union[np.ndarray, torch.Tensor]

# ==================================================
# ================= AxisTracker ====================
# ==================================================

class AxisTracker:
    """
    Utility class to track axis modifications, tagging, and propagation
    of metadata (conversion, axis_map, uid, etc.) across image operations.
    """

    # ====[ Initialization ]====
    def __init__(self, image: ArrayLike, operator: Optional[Any] = None, framework: Optional[str] = "numpy") -> None:
        """
        Initialize an AxisTracker for the given image, optionally linking it to an operator.

        Parameters
        ----------
        image : ArrayLike
            The image to track, typically a NumPy array or PyTorch tensor.
        operator : Any, optional
            Optional reference to the operator that produced or is managing the image.
        framework : str, optional
            Backend framework used for axis interpretation, either "numpy" or "torch".

        Returns
        -------
        None
        """
        self.image: ArrayLike = image
        self.operator: Optional[Any] = operator
        self.framework: str = framework or ("torch" if torch.is_tensor(image) else "numpy")

        # Pull existing tag (project registry may infer framework).
        self.tag: Dict[str, Any] = deepcopy(get_tag(image)) if has_tag(image) else {}

        # Ensure UID exists and basic trace
        if "uid" not in self.tag:
            self.tag["uid"] = str(uuid.uuid4())
            self.tag["trace"] = [f"INIT:UID={self.tag['uid']}"]
            set_tag(self.image, self.tag)

        # Canonical axis keys we track in tags
        self.axes: List[str] = [
            "batch_axis",
            "channel_axis",
            "direction_axis",
            "depth_axis",
            "height_axis",
            "width_axis",
        ]

    # ====[ Internal tag update via OperatorCore ]====
    def _update_tag(self, key: str, value: Any, add_new_axis: bool = False) -> None:
        """
        Update an internal metadata tag associated with the array, such as axis information.

        This method modifies the internal tag dictionary by setting the given key to the provided value.
        Optionally, a new axis entry can be added if not already present.

        Parameters
        ----------
        key : str
            Name of the metadata tag to update (e.g., "axis_map", "uid", "layout").
        value : Any
            Value to assign to the tag (type depends on the key).
        add_new_axis : bool, optional
            If True, allows adding a new axis entry when updating axis-related tags.

        Returns
        -------
        None
            Updates are made in-place.
        """

        if self.operator:
            # Delegate to OperatorCore to ensure registry consistency
            self.operator._update_axis_tag(self.image, key, value, add_new_axis=add_new_axis)
        else:
            # Fallback: update directly if no operator provided
            tag = deepcopy(get_tag(self.image)) or {}
            if add_new_axis or (key in tag):
                tag[key] = value
                set_tag(self.image, tag)

    def update_tag(self, key: str, value: Any, add_new_axis: bool = False) -> None:
        """
        Public method to update metadata tags associated with the array or object.

        Delegates to the internal `_update_tag` method for applying the update,
        optionally allowing the creation of a new axis tag if needed.

        Parameters
        ----------
        key : str
            Tag key to update (e.g., "layout", "axis_map", "uid").
        value : Any
            Value to assign to the tag. Type depends on the key.
        add_new_axis : bool, optional
            If True, allows creating a new axis entry if the key involves axis metadata.

        Returns
        -------
        None
            The internal tag dictionary is updated in-place.
        """

        self._update_tag(key, value, add_new_axis=add_new_axis)

    def update_tags(self, updates: Mapping[str, Any], add_new_axis: bool = False) -> None:
        """
        Update multiple metadata tags at once.

        Applies a batch of key-value updates to the internal tag dictionary.
        Optionally allows the creation of new axis-related entries if specified.

        Parameters
        ----------
        updates : Mapping[str, Any]
            Dictionary of tag updates, where each key corresponds to a tag name
            and each value is the value to assign.
        add_new_axis : bool, optional
            If True, permits adding new axis entries when relevant.

        Returns
        -------
        None
            All updates are applied in-place to the internal tag structure.
        """
        for k, v in updates.items():
            self._update_tag(k, v, add_new_axis=add_new_axis)

    # ====[ Copy tags to another image with deep update]====
    
    def clone_to(
        self,
        target: Union[ArrayLike, "AxisTracker"],
        keys: Optional[Sequence[str]] = None,
        override: bool = True,
        updates: Optional[Mapping[str, Any]] = None,
        preserve_uid: bool = False,
        force_new_uid: bool = False,
    ) -> "AxisTracker":
        """
        Clone the current tracker’s metadata tags to another image or AxisTracker.

        This method transfers selected metadata keys (tags) from the source tracker
        to a new or existing target (which can be a raw image or another AxisTracker).
        UID is excluded by default unless `force_new_uid` is False.

        Parameters
        ----------
        target : ArrayLike or AxisTracker
            Target object to receive the cloned tags. Can be a NumPy/Torch array
            or another `AxisTracker` instance.
        keys : Sequence[str], optional
            List of tag keys to copy. If None, all tags except "uid" are copied.
        override : bool, optional
            If True, overwrite existing tags on the target. If False, keep existing ones.
        updates : Mapping[str, Any], optional
            Additional key-value pairs to update or override after cloning.
        preserve_uid : bool, optional
            If True, preserve the UID from the source tracker in the target tracker.
        force_new_uid : bool, optional
            If True, generate a new UID for the target tracker, even if one exists.

        Returns
        -------
        AxisTracker
            A new AxisTracker instance wrapping the target image with updated tags.
        """

        
        target_img = target.image if isinstance(target, AxisTracker) else target
        source_tag = self.get_tag() or None
        
        if source_tag is None:
            raise ValueError("No tag found in source tracker.")

        if keys is None:
            keys = list(source_tag.keys())
            if not preserve_uid:
                keys = [k for k in keys if k != "uid"]

        tag_subset = deepcopy({k: source_tag[k] for k in keys if k in source_tag})

        tag = get_tag(target_img) or {}
        
        if override:
            tag.update(tag_subset)
            trace_msg = f"[clone_to] override: keys={list(tag_subset.keys())}"
        else:
            new_items = {k: v for k, v in tag_subset.items() if k not in tag}
            tag.update(new_items)
            trace_msg = f"[clone_to] merge: new_keys={list(new_items.keys())}"

        if updates:
            tag.update(deepcopy(updates))
            trace_msg += f" + updates={list(updates.keys())}"
            
        if force_new_uid:
            tag["uid"] = uuid.uuid4().hex[:8]

        # Add to trace
        tag.setdefault("trace", []).append(trace_msg)
        set_tag(target_img, tag)


        return AxisTracker(target_img, self.operator, self.framework)
 
    def copy_to(
        self,
        target: Union[ArrayLike, "AxisTracker"],
        keys: Optional[Sequence[str]] = None,
        override: bool = True,
        preserve_uid: bool = False,
        force_new_uid: bool = False
    ) -> "AxisTracker":
        """
        Copy selected metadata tags to another image or AxisTracker.

        Unlike `clone_to`, this method does not apply any extra updates
        beyond the selected tags. UID is excluded by default unless
        `force_new_uid` is set to True.

        Parameters
        ----------
        target : ArrayLike or AxisTracker
            Target object to receive the copied tags. Can be a raw array or tracker.
        keys : Sequence[str], optional
            List of tag keys to copy. If None, copy all tags except "uid".
        override : bool, optional
            If True, overwrite existing tags on the target. Default is True.
        preserve_uid : bool, optional
            If True, preserve the UID from the source tracker in the target tracker.
        force_new_uid : bool, optional
            If True, assign a new UID to the target even if one already exists.

        Returns
        -------
        AxisTracker
            A new AxisTracker wrapping the target with copied metadata tags.
        """

        return self.clone_to(target, keys=keys, override=override, preserve_uid=preserve_uid, force_new_uid=force_new_uid)

    def copy_tag_from(self, source_tracker: "AxisTracker", keys: Optional[Sequence[str]] = None) -> "AxisTracker":
        """
        Copy metadata tags from another AxisTracker into the current tracker.

        This method selectively imports tags from `source_tracker` and merges
        them into the current tracker’s internal tag dictionary.

        Parameters
        ----------
        source_tracker : AxisTracker
            Another AxisTracker instance from which to copy tags.
        keys : Sequence[str], optional
            List of tag keys to copy. If None, all tags except "uid" are copied.

        Returns
        -------
        AxisTracker
            The current tracker instance, with updated tags.
        """
        if not isinstance(source_tracker, AxisTracker):
            raise TypeError("source_tracker must be an AxisTracker instance.")
        return source_tracker.copy_to(self.image, keys=keys, override=True)

    def with_updated_tag(self, updates: Mapping[str, Any], force_new_uid: bool = False) -> "AxisTracker":
        """
        Return a new AxisTracker with updated metadata tags.

        This method creates a copy of the current tracker and applies the given
        tag updates. Optionally generates a new UID for the new tracker.

        Parameters
        ----------
        updates : Mapping[str, Any]
            Dictionary of tag key-value pairs to update or add.
        force_new_uid : bool, optional
            If True, force the creation of a new UID in the copied tracker.

        Returns
        -------
        AxisTracker
            A new tracker instance with updated tags and shared data.
        """
        img_copy = deepcopy(self.image)
        return self.clone_to(img_copy, override=True, updates=updates, force_new_uid=force_new_uid)

    def copy_tag_to_many(
        self,
        targets: Sequence[Optional[Union[ArrayLike, "AxisTracker"]]],
        keys: Optional[Sequence[str]] = None,
        override: bool = False,
        update_tags: Optional[Mapping[str, Any]] = None
    ) -> List["AxisTracker"]:
        """
        Copy metadata tags from the current tracker to multiple target images or trackers.

        Each target is wrapped in a new AxisTracker instance, receiving the selected tags
        from the current tracker. Additional updates can be applied uniformly to all outputs.

        Parameters
        ----------
        targets : Sequence of ArrayLike or AxisTracker
            List of target arrays or AxisTracker instances to which the tags will be copied.
        keys : Sequence[str], optional
            List of tag keys to copy. If None, all tags except "uid" are copied.
        override : bool, optional
            If True, overwrite existing tags in the targets. Default is False.
        update_tags : Mapping[str, Any], optional
            Additional tag updates to apply to all targets after copying.

        Returns
        -------
        List[AxisTracker]
            List of new AxisTracker instances with copied and updated tags.
        """

        return [self.clone_to(t, keys=keys, override=override, updates=update_tags) for t in targets if t is not None]

    def copy_and_tag_scalar(
        self,
        scalar: Any,
        keys: Optional[Sequence[str]] = None,
        override: bool = True
    ) -> "AxisTracker":
        """
        Copy selected metadata tags to a scalar-like object wrapped in a new AxisTracker.

        This method is useful for attaching contextual metadata (e.g., layout, UID)
        to scalar results derived from tracked data (e.g., metrics, statistics).

        Parameters
        ----------
        scalar : Any
            Scalar-like value (e.g., float, int, 0D array) to be wrapped and tagged.
        keys : Sequence[str], optional
            List of tag keys to copy. If None, all tags except "uid" are copied.
        override : bool, optional
            If True, overwrite any existing tags in the scalar tracker. Default is True.

        Returns
        -------
        AxisTracker
            New AxisTracker instance wrapping the scalar and carrying the selected tags.
        """

        default_keys = ["status", "converted", "original_shape", "shape_after", "device", "framework", "trace", "op_params"]
        return self.clone_to(scalar, keys=keys or default_keys, override=override)

    # ====[ Move axis and update tag ]====
    def moveaxis(self, src: Optional[int], dst: Optional[int]) -> "AxisTracker":
        """
        Return a new AxisTracker with one axis moved to a new position.

        This operation is similar to `np.moveaxis`, and updates both the underlying data
        and associated metadata (e.g., layout, axis map) to reflect the new axis ordering.

        Parameters
        ----------
        src : int or None
            Original position of the axis to move. If None, no operation is performed.
        dst : int or None
            New position for the axis. If None, no operation is performed.

        Returns
        -------
        AxisTracker
            New AxisTracker instance with updated axis order and metadata.

        Notes
        -----
        - This operation is non-destructive: it returns a new tracker.
        - If either `src` or `dst` is None, the original tracker is returned unchanged.
        """

        # Skip move if not applicable
        if src is None or dst is None or src == dst:
            return self

        if isinstance(self.image, np.ndarray):
            new_img = np.moveaxis(self.image, src, dst)
        elif isinstance(self.image, torch.Tensor):
            new_img = torch.movedim(self.image, src, dst)
        else:
            raise TypeError("Unsupported image type")

        tracker = self.copy_to(new_img, preserve_uid=True)
        tracker.update_all_axis_tags_after_move(src, dst)
        tracker.update_tag("shape_after", new_img.shape)
        return tracker
    
    # ====[ Move axis and update tag ]====
    def update_all_axis_tags_after_move(self, src: int, dst: int) -> None:
        """
        Update all axis-related metadata to reflect a move of axis from `src` to `dst`.

        This method should be called after performing a moveaxis operation on the data,
        to ensure internal tags such as layout, axis_map, or semantic mappings remain consistent.

        Parameters
        ----------
        src : int
            Original axis index before the move.
        dst : int
            New axis index after the move.

        Returns
        -------
        None
            The internal tags are updated in-place.
        """

        tag = self.get_tag() or {}
        shape = tag.get("shape_after", tag.get("original_shape", None))
        
        if shape is None:
            raise RuntimeError("Cannot update axis tags: shape not found in tag.")
        
        shift = list(range(len(shape)))
        axis_val = shift.pop(src)
        shift.insert(dst, axis_val)
        index_map = {v: i for i, v in enumerate(shift)}
        
        self.tag.setdefault("trace", []).append(f"[moveaxis] src={src}, dst={dst}, index_map={index_map}")
        
        updated_axes: Dict[str, Optional[int]] = {}
        bool_axes: Dict[str, Union[bool, str, None]] = {}
        
        for axis_name in self.axes:
            axis_pos = tag.get(axis_name, None)
            bool_axes_name = axis_name.replace("_axis","")
            bool_axes[bool_axes_name] = False
            if axis_pos is not None:
                new_pos = index_map.get(axis_pos)
                self.update_tag(axis_name, new_pos)
                updated_axes[axis_name] = new_pos
                bool_axes[bool_axes_name] = True
        
        # --- Update axis_map ---
        self._update_tag("axis_map", updated_axes)
        
        # --- Update layout name ---
        if "channel_axis" in updated_axes:  
            if updated_axes["channel_axis"] != len(shape) - 1:
                bool_axes["channel_position"] = "first"    
            else:
                bool_axes["channel_position"]  = "last"
        else:
            bool_axes["channel_position"] = None
        
        layout_str = LayoutResolver.from_flags(**bool_axes)
        
        self.update_tag("layout_name", layout_str)

    # ====[ Stack and propagate tag ]====
    def copy_and_stack(self, others: Sequence["AxisTracker"], axis: int = 0, update_axis_tag: str = "channel_axis") -> "AxisTracker":
        """
        Stack multiple AxisTracker instances along a new or existing axis and propagate metadata.

        This method stacks the current tracker with others in the list, assuming they share
        compatible shapes (except along the stacking axis). Metadata from the source trackers
        is merged, and the specified axis tag is updated accordingly.

        Parameters
        ----------
        others : Sequence[AxisTracker]
            List of other AxisTracker instances to stack with the current one.
        axis : int, optional
            Axis along which to stack the data. Default is 0.
        update_axis_tag : str, optional
            Name of the axis-related tag to update (e.g., "channel_axis", "feature_axis").
            Default is "channel_axis".

        Returns
        -------
        AxisTracker
            A new AxisTracker instance containing the stacked data and updated metadata.

        Notes
        -----
        - All inputs must be AxisTracker instances.
        - Tag merging strategy is inherited from internal logic (e.g., tag copying, UID refresh).
        - Shapes must be compatible for stacking.
        """
        stacked = (
            torch.stack([self.image] + [t.image for t in others], dim=axis)
            if isinstance(self.image, torch.Tensor)
            else np.stack([self.image] + [t.image for t in others], axis=axis)
        )
        tracker = AxisTracker(stacked, self.operator, self.framework)
        tracker._update_tag("axis_map", deepcopy(self.get_tag().get("axis_map", {})) if self.get_tag() else {})
        tracker._update_tag(update_axis_tag, axis)
        tracker._update_tag("shape_after", stacked.shape)
        return tracker

    def stack_from(self, images: Sequence[ArrayLike], axis: int = 0, update_tags: Optional[Mapping[str, Any]] = None) -> "AxisTracker":
        """
        Stack raw images with the current one along a given axis and return a new AxisTracker.

        This method stacks the current image (wrapped in this tracker) with other untracked
        images (arrays), then returns a new AxisTracker around the result. Tags from the
        current tracker are propagated, and optionally updated via `update_tags`.

        Parameters
        ----------
        images : Sequence[ArrayLike]
            A list of raw image arrays to stack with the current one.
        axis : int, optional
            Axis along which to perform the stacking. Default is 0.
        update_tags : dict[str, Any], optional
            Additional or overriding metadata tags to apply on the resulting tracker.

        Returns
        -------
        AxisTracker
            A new tracker wrapping the stacked array and carrying the updated tags.

        Notes
        -----
        - The input images must have shapes compatible with the current image (except along `axis`).
        - This method does not enforce consistency of layout or semantics between arrays.
        - UID is regenerated by default unless explicitly preserved in `update_tags`.
        """
        if images is None:
            raise ValueError("Empty list of images provided.")
        first_type = type(images[0])
        if not all(isinstance(img, first_type) for img in images):
            raise TypeError("All images must be of the same type for stacking.")

        stacked = torch.stack(images, dim=axis) if issubclass(first_type, torch.Tensor) else np.stack(images, axis=axis)
        tracker = self.copy_to(stacked)
        if update_tags:
            for k, v in update_tags.items():
                tracker._update_tag(k, v)
        return tracker

    # ====[ Slice / split tracking ]====
    @staticmethod
    def from_sliced(
        source_img: ArrayLike,
        sliced_img: ArrayLike,
        operator: Optional[Any],
        framework: str,
        remove_axes: Sequence[str] = ("G", "N", "C"),
    ) -> ArrayLike:
        """
        Build a tracked view for a sliced image based on a larger tagged source.

        This method reconstructs an AxisTracker for `sliced_img` by inheriting
        tags from `source_img`, adjusting axis metadata to account for slicing,
        and optionally removing axis tags such as 'G', 'N', 'C'.

        Parameters
        ----------
        source_img : ArrayLike
            The original tagged image from which the slice was extracted.
        sliced_img : ArrayLike
            The resulting sliced view (same backend as source).
        operator : Any, optional
            The operator used to generate the slice (for traceability or logging).
        framework : str
            Backend framework, e.g., 'numpy' or 'torch'.
        remove_axes : Sequence[str], optional
            Axis tags to remove in the resulting tracker. Default is ('G', 'N', 'C').

        Returns
        -------
        ArrayLike
            A new image array wrapped in an AxisTracker, with updated tags
            reflecting the slicing operation.
        """

        if not isinstance(sliced_img, (np.ndarray, torch.Tensor)):
            raise TypeError("sliced_img must be a NumPy array or PyTorch tensor.")

        tagger = AxisTracker(source_img, operator, framework)
        src_tag = tagger.get_tag() or {}
        layout_name = src_tag.get("layout_name", None)
        if layout_name is None:
            raise ValueError("Missing layout_name in source image tag.")

        layout_str = "".join([ax for ax in layout_name if ax not in remove_axes])
        axes = get_layout_axes(framework, layout_str)
        axes.pop("name", None)
        axes.pop("description", None)

        tracker = tagger.copy_to(sliced_img)
        tracker.update_tags(axes)
        tracker.update_tag("shape_after", sliced_img.shape)
        tracker.update_tag("status", "split")
        tracker.update_tag("layout_name", layout_str)

        return tracker.get()

    # ====[ Apply transformation and re-track ]====
    def apply_to_all(self, func: Callable, update_tags: Optional[Mapping[str, Any]] = None, **kwargs) -> "AxisTracker":
        """
        Apply a function to the underlying image and propagate the tags.

        This method applies a user-defined function `func` to the image wrapped by
        the current AxisTracker. The result is then wrapped in a new AxisTracker
        with the same tags (unless overridden via `update_tags`).

        Parameters
        ----------
        func : Callable
            Function to apply to the image. Must return a NumPy array or a Torch tensor.
        update_tags : dict, optional
            Tags to update or add in the resulting tracker.
        **kwargs :
            Additional keyword arguments passed to `func`.

        Returns
        -------
        AxisTracker
            A new tracker wrapping the result of `func(self.image, **kwargs)`, with
            propagated and/or updated tags.

        Raises
        ------
        TypeError
            If the function output is not a NumPy array or Torch tensor.
        """
        output = func(self.image, **kwargs)
        if not isinstance(output, (np.ndarray, torch.Tensor)):
            raise TypeError(f"[apply_to_all] The function must return an ndarray or torch.Tensor, got {type(output)}")

        tracker = self.clone_to(output, override=True, updates=update_tags)
        return tracker

    # ====[ Static helper ]====
    @staticmethod
    def tag_many(
        images: Sequence[Union[ArrayLike, "AxisTracker"]],
        operator: Optional[Any],
        framework: str = "numpy",
        updates: Optional[Mapping[str, Any]] = None,
        return_trackers: bool = False,
    ) -> List[Union["AxisTracker", ArrayLike]]:
        """
        Tag a list of images with a shared operator and framework.

        This utility wraps each image (or existing AxisTracker) in an AxisTracker
        with a shared operator and framework. Additional metadata can be injected
        via `updates`.

        Parameters
        ----------
        images : list of ndarray | Tensor | AxisTracker
            Sequence of raw images or already-tagged AxisTrackers.
        operator : object or None
            Operator used for tagging (can be a callable or config object).
        framework : str, default='numpy'
            Framework used by the operator ('numpy', 'torch', etc.).
        updates : dict, optional
            Extra key-value pairs to inject into the tag for each image.
        return_trackers : bool, default=False
            If True, return the full AxisTracker objects. Otherwise, return only
            the tagged images (raw arrays/tensors).

        Returns
        -------
        list of AxisTracker or list of array-like
            List of tagged images or AxisTrackers, depending on `return_trackers`.
        """

        results: List[Union["AxisTracker", ArrayLike]] = []
        for img in images:
            if isinstance(img, AxisTracker):
                tracker = img
            elif isinstance(img, (np.ndarray, torch.Tensor)):
                tracker = AxisTracker(img, operator, framework)
            else:
                raise TypeError("Each element must be an ndarray, Tensor, or AxisTracker.")

            if updates:
                tracker.update_tags(updates)

            results.append(tracker if return_trackers else tracker.get())

        return results

    # ====[ Getters ]====
    def get(self) -> ArrayLike:
        """
        Return the underlying image or tensor.
        """
        return self.image

    def get_tag(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve tag from operator or directly from array (fallback).
        """
        if self.operator is not None:
            tag = self.operator.get_tag(self.image, self.framework)
            return deepcopy(tag) if tag is not None else None
        elif has_tag(self.image):
            return deepcopy(get_tag(self.image))
        else:
            return None
                
    def reset_tag(self, keys: Optional[Sequence[str]] = None, keep: Optional[Sequence[str]] = None, trace_msg: Optional[str] = None) -> "AxisTracker":
        """
        Reset selected tag entries or full tag while preserving essential info.

        Parameters
        ----------
        keys : list[str] or None
            Specific keys to reset in the tag. If None, all keys are reset.
        keep : list[str] or None
            Keys to preserve even if keys=None (default: ['uid', 'framework', 'device']).
        trace_msg : str or None
            Optional message to append to the trace log.
        """
        old_tag = deepcopy(self.get_tag())
        if old_tag is None:
            return self

        if keep is None:
            keep = ["uid", "framework", "device"]

        preserved = {k: old_tag[k] for k in keep if k in old_tag}
        trace = old_tag.get("trace", [])
        trace.append(f"[reset_tag] {trace_msg}" if trace_msg else f"[reset_tag] reset {'selected keys' if keys else 'all'}")

        if keys is None:
            new_tag = {**preserved, "trace": trace}
        else:
            new_tag = deepcopy(old_tag)
            for k in keys:
                new_tag.pop(k, None)
            new_tag["trace"] = trace

        set_tag(self.image, new_tag)
        self.tag = new_tag  # update cached copy
        return self
  
    def get_axis(self, role: str) -> Optional[int]:
        """
        Return the axis index corresponding to a given semantic role.

        Parameters
        ----------
        role : str
            Semantic axis name to query (e.g., "depth_axis", "height_axis").

        Returns
        -------
        int or None
            The axis index if defined in the tag, otherwise None.
        """
        tag = self.get_tag()
        return tag.get(role) if tag else None

    def get_uid(self) -> Optional[str]:
        """
        Return the unique identifier (UID) from the current tag.

        Returns
        -------
        str or None
            UID string if present in the tag, otherwise None.
        """

        tag = self.get_tag()
        return tag.get("uid") if tag else None

    # ====[ Summary Printer ]====
    def tag_summary(self, keys: Optional[List[str]] = None, include_uid: bool = True) -> None:
        """
        Print a summary of selected metadata keys from the current tag.

        Parameters
        ----------
        keys : list of str, optional
            Keys to display from the tag. If None, a default set of axis-related
            keys will be used.
        include_uid : bool, optional
            If True, include the 'uid' key in the summary.

        Returns
        -------
        None
            The function prints the tag summary to stdout.
        """
        tag = self.get_tag()
        if tag is None:
            print("[AxisTracker] No tag found.")
            return

        keys = keys or [
            "status",
            "framework",
            "channel_axis",
            "batch_axis",
            "direction_axis",
            "depth_axis",
            "height_axis",
            "width_axis",
            "layout_name",
            "shape_before",
            "shape_after",
        ]

        if "feature_axis" in tag:
            keys.append("feature_axis")

        if include_uid:
            keys.append("uid")

        print("====[ AxisTracker Summary ]====")
        for key in keys:
            if key in tag:
                print(f"{key:>15} : {tag[key]}")


    # ====[ Repr ]====
    def __repr__(self) -> str:
        """
        Return a string representation of the tracked image, including type, shape, and UID.

        Returns
        -------
        str
            Formatted string with the image type, shape, and associated UID (if available).
        """
        
        tag = self.get_tag()
        uid = tag.get("uid") if tag else "None"
        shape = getattr(self.image, "shape", "?")
        return f"Tracked[{type(self.image).__name__}, shape={shape}, uid={uid}]"

    # ====[ Small helpers ]====
    @staticmethod
    def _remove_layout_axes(layout: str, remove_list: Sequence[str]) -> str:
        """
        Remove specified axis characters from a layout string.

        Parameters
        ----------
        layout : str
            Original layout string (e.g., "NCHW", "ZYX").
        remove_list : Sequence[str]
            List of axis characters to remove from the layout.

        Returns
        -------
        str
            New layout string with specified axes removed.
        """

        return "".join(ax for ax in layout if ax not in remove_list)

    @staticmethod
    def debug_tag_batch(batch: Sequence[ArrayLike], context: str = "Pre") -> None:
        """
        Print debug information for a batch of tagged images.

        Parameters
        ----------
        batch : Sequence[ArrayLike]
            List or sequence of image-like objects, each expected to carry a tag.
        context : str, optional
            Context label to include in the debug header (e.g., "Pre", "Post").

        Returns
        -------
        None
            The function prints debug information to stdout.
        """

        print(f"\n [DEBUG:{context}] Batch of {len(batch)} images")
        for i, img in enumerate(batch):
            tag = get_tag(img)
            if not tag:
                print(f" >> Img[{i}] : NO TAG")
                continue
            print(f"  >> Img[{i}] uid={tag.get('uid')}, shape={getattr(img, 'shape', '?')}, layout={tag.get('layout_name')}")
            print(f"  >>   trace: {' > '.join(tag.get('trace', []))}")
