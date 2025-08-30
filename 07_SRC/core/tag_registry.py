# ==================================================
# =============  MODULE: tag_registry ==============
# ==================================================
from __future__ import annotations

from typing import Any, Callable, Deque, Optional
import collections
import copy
import uuid
import warnings

import numpy as np
import torch

__all__ = [
    "get_tag",
    "set_tag",
    "has_tag",
    "del_tag",
    "update_tag",
    "copy_tag",
    "get_tag_uid",
    "set_tag_uid",
    "get_tag_summary",
    "get_backend",
    "get_tag_by_id",
    "set_tag_by_id",
    "del_tag_by_id",
    "has_tag_by_id",
    "update_tag_by_id",
    "copy_tag_by_id",
    "deepcopy_tag_by_id",
    "all_tags_by_backend",
    "clear_tags_by_backend",
    "clean_tags_from",
    "clean_invalid_tags",
]

# ====[ Central Tag Backends ]====
# Registry maps object id(int) -> tag(dict)
TAG_BACKENDS: dict[str, dict[str, Any]] = {
    "numpy": {
        "check": lambda x: isinstance(x, np.ndarray),
        "registry": {},  # type: dict[int, dict[str, Any]]
    },
    "torch": {
        "check": lambda x: isinstance(x, torch.Tensor),
        "registry": {},  # type: dict[int, dict[str, Any]]
    },
}

# ====[ Optional Debug Log – Keep Track of Last 100 Tags ]====
TAG_LOG: Deque[tuple[str, int, dict[str, Any]]] = collections.deque(maxlen=100)


# ====[ Internal helpers ]====
def _validate_backend(backend: str) -> None:
    """
    Validate that the provided backend is supported.

    Parameters
    ----------
    backend : str
        Backend name to validate ('numpy' or 'torch').

    Raises
    ------
    ValueError
        If the backend is not in the list of supported backends.
    """
    if backend not in TAG_BACKENDS:
        raise ValueError(f"Unsupported backend '{backend}'. Expected 'numpy' or 'torch'.")


# ====[ Backend Detection ]====
def get_backend(obj: Any) -> Optional[str]:
    """
    Determine the backend ('numpy' or 'torch') of a given array-like object.

    Parameters
    ----------
    obj : Any
        Object to inspect (expected to be a NumPy array or PyTorch tensor).

    Returns
    -------
    str or None
        'numpy' if the object is a NumPy array,
        'torch' if it is a Torch tensor,
        or None if the backend cannot be determined.

    Notes
    -----
    - Detection is performed via isinstance checks.
    """
    for name, backend in TAG_BACKENDS.items():
        if backend["check"](obj):
            return name
    return None

# ====[ Registry by ID ]====
def get_tag_by_id(backend: str, id_: int) -> dict[str, Any]:
    """
    Retrieve the tag dictionary associated with a given object ID and backend.

    Parameters
    ----------
    backend : str
        Backend identifier ('numpy' or 'torch').
    id_ : int
        ID of the object (typically obtained via `id(obj)`).

    Returns
    -------
    dict[str, Any]
        Tag dictionary if found; otherwise an empty dict.
    """
    _validate_backend(backend)
    return TAG_BACKENDS[backend]["registry"].get(id_, {})  # type: ignore[return-value]


def set_tag_by_id(backend: str, id_: int, tag: dict[str, Any]) -> None:
    """
    Assign or overwrite the tag dictionary for a given object ID and backend.

    Parameters
    ----------
    backend : str
        Backend identifier ('numpy' or 'torch').
    id_ : int
        ID of the object (typically obtained via `id(obj)`).
    tag : dict[str, Any]
        Dictionary of metadata to associate with the object.

    Returns
    -------
    None
    """
    _validate_backend(backend)
    TAG_BACKENDS[backend]["registry"][id_] = tag
    # store a shallow copy for debug history
    TAG_LOG.append((backend, id_, copy.copy(tag)))


def del_tag_by_id(backend: str, id_: int) -> None:
    """
    Delete the tag associated with a given object ID and backend.

    Parameters
    ----------
    backend : str
        Backend identifier ('numpy' or 'torch').
    id_ : int
        ID of the object (typically obtained via `id(obj)`).

    Returns
    -------
    None
        Does nothing if the tag does not exist.
    """
    _validate_backend(backend)
    TAG_BACKENDS[backend]["registry"].pop(id_, None)


def has_tag_by_id(backend: str, id_: int) -> bool:
    """
    Check whether a tag exists for the given object ID and backend.

    Parameters
    ----------
    backend : str
        Backend identifier ('numpy' or 'torch').
    id_ : int
        ID of the object (typically obtained via `id(obj)`).

    Returns
    -------
    bool
        True if a tag exists for this ID on the given backend; False otherwise.
    """
    _validate_backend(backend)
    return id_ in TAG_BACKENDS[backend]["registry"]


def update_tag_by_id(backend: str, id_: int, updates: dict[str, Any]) -> None:
    """
    Update specific fields of an existing tag associated with a given object ID and backend.

    Parameters
    ----------
    backend : str
        Backend identifier ('numpy' or 'torch').
    id_ : int
        ID of the object whose tag will be updated.
    updates : dict[str, Any]
        Dictionary of fields to update in the tag.

    Returns
    -------
    None
    """
    _validate_backend(backend)
    if id_ in TAG_BACKENDS[backend]["registry"]:
        TAG_BACKENDS[backend]["registry"][id_].update(updates)


def copy_tag_by_id(backend: str, source_id: int, target_id: int) -> None:
    """
    Copy the tag from one object ID to another within the same backend.

    This creates a shallow copy of the source tag and assigns it to the target.

    Parameters
    ----------
    backend : str
        Backend identifier ('numpy' or 'torch').
    source_id : int
        ID of the object from which to copy the tag.
    target_id : int
        ID of the object to which the tag will be assigned.

    Returns
    -------
    None
    """
    _validate_backend(backend)
    if source_id in TAG_BACKENDS[backend]["registry"]:
        TAG_BACKENDS[backend]["registry"][target_id] = TAG_BACKENDS[backend]["registry"][
            source_id
        ].copy()


def deepcopy_tag_by_id(backend: str, source_id: int, target_id: int) -> None:
    """
    Create a deep copy of the tag from one object ID to another within the same backend.

    This ensures that nested structures in the tag (e.g., dicts, lists) are fully copied.

    Parameters
    ----------
    backend : str
        Backend identifier ('numpy' or 'torch').
    source_id : int
        ID of the object from which to deep-copy the tag.
    target_id : int
        ID of the object to which the tag will be assigned.

    Returns
    -------
    None
    """
    _validate_backend(backend)
    if source_id in TAG_BACKENDS[backend]["registry"]:
        TAG_BACKENDS[backend]["registry"][target_id] = copy.deepcopy(
            TAG_BACKENDS[backend]["registry"][source_id]
        )


def all_tags_by_backend(backend: str) -> dict[int, dict[str, Any]]:
    """
    Return a shallow copy of all tags currently registered for the given backend.

    Parameters
    ----------
    backend : str
        Backend identifier ('numpy' or 'torch').

    Returns
    -------
    dict[int, dict[str, Any]]
        Mapping of object IDs to their associated tag dictionaries.
    """
    _validate_backend(backend)
    return TAG_BACKENDS[backend]["registry"].copy()


def clear_tags_by_backend(backend: str) -> None:
    """
    Remove all tags associated with the specified backend.

    Parameters
    ----------
    backend : str
        Backend identifier ('numpy' or 'torch').

    Returns
    -------
    None
    """
    _validate_backend(backend)
    TAG_BACKENDS[backend]["registry"].clear()


def clean_tags_from(backend: str, ids: list[int]) -> None:
    """
    Delete tags for a list of object IDs within the specified backend.

    Missing IDs are silently ignored.

    Parameters
    ----------
    backend : str
        Backend identifier ('numpy' or 'torch').
    ids : list of int
        List of object IDs whose tags should be removed.

    Returns
    -------
    None
    """
    _validate_backend(backend)
    for i in ids:
        del_tag_by_id(backend, i)


def clean_invalid_tags(backend: str, check_fn: Optional[Callable[[dict[str, Any]], bool]] = None) -> None:
    """
    Remove all tags in the given backend that do not satisfy the validation function.

    Parameters
    ----------
    backend : {'numpy', 'torch'}
        Backend whose tag registry should be cleaned.
    check_fn : callable or None, optional
        Function that takes a tag dictionary and returns True if the tag is valid.
        Tags for which this function returns False will be deleted.
        If None, no tags are removed.

    Returns
    -------
    None
    """
    _validate_backend(backend)
    if check_fn is None:
        return
    to_delete: list[int] = []
    for obj_id, tag in list(TAG_BACKENDS[backend]["registry"].items()):
        if not check_fn(tag):
            to_delete.append(obj_id)
    for i in to_delete:
        del_tag_by_id(backend, i)


# ====[ Unified Interface ]====
def get_tag(obj: Any) -> dict[str, Any]:
    """
    Retrieve the tag dictionary attached to the given object.

    Parameters
    ----------
    obj : Any
        Object to inspect (typically a NumPy array or Torch tensor).

    Returns
    -------
    dict[str, Any]
        Tag dictionary if present; otherwise an empty dict.
    """
    backend = get_backend(obj)
    return get_tag_by_id(backend, id(obj)) if backend else {}  # type: ignore[arg-type]


def set_tag(obj: Any, tag: dict[str, Any]) -> None:
    """
    Attach or overwrite a tag dictionary on the given object.

    Parameters
    ----------
    obj : Any
        Object to tag (must be a supported type, e.g., NumPy array or Torch tensor).
    tag : dict[str, Any]
        Metadata dictionary to associate with the object.

    Returns
    -------
    None
    """
    backend = get_backend(obj)
    if backend:
        set_tag_by_id(backend, id(obj), tag)


def has_tag(obj: Any) -> bool:
    """
    Check whether the given object has a tag registered in its backend.

    Parameters
    ----------
    obj : Any
        Object to check (e.g., NumPy array or Torch tensor).

    Returns
    -------
    bool
        True if a tag exists for the object; False otherwise.
    """
    backend = get_backend(obj)
    return has_tag_by_id(backend, id(obj)) if backend else False  # type: ignore[arg-type]


def del_tag(obj: Any) -> None:
    """
    Delete the tag associated with the given object, if supported by the backend.

    Parameters
    ----------
    obj : Any
        Object whose tag should be removed (e.g., NumPy array or Torch tensor).

    Returns
    -------
    None
    """
    backend = get_backend(obj)
    if backend:
        del_tag_by_id(backend, id(obj))


def update_tag(obj: Any, updates: dict[str, Any]) -> None:
    """
    Update selected fields in the tag associated with the given object.

    Parameters
    ----------
    obj : Any
        Object whose tag should be updated (e.g., NumPy array or Torch tensor).
    updates : dict[str, Any]
        Dictionary of key-value pairs to update in the tag.

    Returns
    -------
    None
        If the object has no tag, the function does nothing.
    """
    backend = get_backend(obj)
    if backend:
        update_tag_by_id(backend, id(obj), updates)


def copy_tag(source_obj: Any, target_obj: Any) -> None:
    """
    Copy the tag from the source object to the target object (shallow copy).

    Parameters
    ----------
    source_obj : Any
        Object with an existing tag (e.g., NumPy array or Torch tensor).
    target_obj : Any
        Object to which the tag will be copied.

    Returns
    -------
    None

    Notes
    -----
    - Both objects must use the same backend ('numpy' or 'torch').
    - If shapes differ (when available), a non-blocking warning is emitted.
    """
    src_backend = get_backend(source_obj)
    tgt_backend = get_backend(target_obj)
    if src_backend == tgt_backend and src_backend:
        if hasattr(source_obj, "shape") and hasattr(target_obj, "shape"):
            if getattr(source_obj, "shape", None) != getattr(target_obj, "shape", None):
                warnings.warn(
                    f"[copy_tag] Shape mismatch: {getattr(source_obj, 'shape', None)} → "
                    f"{getattr(target_obj, 'shape', None)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        copy_tag_by_id(src_backend, id(source_obj), id(target_obj))


# ====[ UID Helpers ]====
def get_tag_uid(obj: Any) -> Optional[str]:
    """
    Retrieve the 'uid' field from the tag associated with the given object.

    Parameters
    ----------
    obj : Any
        Object to inspect (e.g., NumPy array or Torch tensor).

    Returns
    -------
    str or None
        UID string if present in the tag; otherwise None.
    """
    return get_tag(obj).get("uid")


def set_tag_uid(obj: Any) -> str:
    """
    Generate a new UUID4 and assign it to the 'uid' field in the object's tag.

    Parameters
    ----------
    obj : Any
        Object to tag (e.g., NumPy array or Torch tensor).

    Returns
    -------
    str
        The generated UID string (full UUID4).
    """
    uid = str(uuid.uuid4())
    update_tag(obj, {"uid": uid})
    return uid


# ====[ TAG SUMMARY – DEBUG INFO ]====
def get_tag_summary(obj: Any) -> dict[str, Any]:
    """
    Return a compact summary of tag metadata associated with the given object.

    Parameters
    ----------
    obj : Any
        Object to inspect (e.g., NumPy array or Torch tensor).

    Returns
    -------
    dict[str, Any]
        Dictionary containing high-level metadata fields, including:
        - 'status': processing status (e.g., 'raw', 'processed', etc.)
        - 'uid': unique identifier if available
        - 'shape': object shape if defined
        - 'framework': 'numpy' or 'torch'
        - 'axes': dict of axis names and their positions
        - 'trace': list of processing steps recorded in the tag
    """
    tag = get_tag(obj)
    return {
        "status": tag.get("status"),
        "uid": tag.get("uid"),
        "shape": tag.get("shape_after"),
        "framework": tag.get("framework"),
        "axes": {
            "channel": tag.get("channel_axis"),
            "batch": tag.get("batch_axis"),
            "direction": tag.get("direction_axis"),
        },
        "trace": tag.get("trace", []),
    }
