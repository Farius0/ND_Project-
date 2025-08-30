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
    """Ensure backend key exists; raise ValueError otherwise."""
    if backend not in TAG_BACKENDS:
        raise ValueError(f"Unsupported backend '{backend}'. Expected 'numpy' or 'torch'.")


# ====[ Backend Detection ]====
def get_backend(obj: Any) -> Optional[str]:
    """
    Return the backend name ('numpy' or 'torch') for a given object, or None.

    Notes
    -----
    - Detection is based on isinstance checks against NumPy ndarray and Torch Tensor.
    """
    for name, backend in TAG_BACKENDS.items():
        if backend["check"](obj):
            return name
    return None

# ====[ Registry by ID ]====
def get_tag_by_id(backend: str, id_: int) -> dict[str, Any]:
    """Return tag dict for object id on a given backend, or {} if missing."""
    _validate_backend(backend)
    return TAG_BACKENDS[backend]["registry"].get(id_, {})  # type: ignore[return-value]


def set_tag_by_id(backend: str, id_: int, tag: dict[str, Any]) -> None:
    """Set/overwrite tag dict for object id on a given backend."""
    _validate_backend(backend)
    TAG_BACKENDS[backend]["registry"][id_] = tag
    # store a shallow copy for debug history
    TAG_LOG.append((backend, id_, copy.copy(tag)))


def del_tag_by_id(backend: str, id_: int) -> None:
    """Delete tag for object id on a given backend (no-op if missing)."""
    _validate_backend(backend)
    TAG_BACKENDS[backend]["registry"].pop(id_, None)


def has_tag_by_id(backend: str, id_: int) -> bool:
    """Return True if a tag exists for this id on the given backend."""
    _validate_backend(backend)
    return id_ in TAG_BACKENDS[backend]["registry"]


def update_tag_by_id(backend: str, id_: int, updates: dict[str, Any]) -> None:
    """Update in-place selected fields of an existing tag."""
    _validate_backend(backend)
    if id_ in TAG_BACKENDS[backend]["registry"]:
        TAG_BACKENDS[backend]["registry"][id_].update(updates)


def copy_tag_by_id(backend: str, source_id: int, target_id: int) -> None:
    """Shallow-copy tag from source id to target id (same backend)."""
    _validate_backend(backend)
    if source_id in TAG_BACKENDS[backend]["registry"]:
        TAG_BACKENDS[backend]["registry"][target_id] = TAG_BACKENDS[backend]["registry"][
            source_id
        ].copy()


def deepcopy_tag_by_id(backend: str, source_id: int, target_id: int) -> None:
    """Deep-copy tag from source id to target id (same backend)."""
    _validate_backend(backend)
    if source_id in TAG_BACKENDS[backend]["registry"]:
        TAG_BACKENDS[backend]["registry"][target_id] = copy.deepcopy(
            TAG_BACKENDS[backend]["registry"][source_id]
        )


def all_tags_by_backend(backend: str) -> dict[int, dict[str, Any]]:
    """Return a shallow copy of the entire registry for this backend."""
    _validate_backend(backend)
    return TAG_BACKENDS[backend]["registry"].copy()


def clear_tags_by_backend(backend: str) -> None:
    """Clear all tag entries for this backend."""
    _validate_backend(backend)
    TAG_BACKENDS[backend]["registry"].clear()


def clean_tags_from(backend: str, ids: list[int]) -> None:
    """Delete tags for all ids in the provided list (no-op for missing)."""
    _validate_backend(backend)
    for i in ids:
        del_tag_by_id(backend, i)


def clean_invalid_tags(backend: str, check_fn: Optional[Callable[[dict[str, Any]], bool]] = None) -> None:
    """
    Remove tags that do not satisfy `check_fn(tag)`.

    Parameters
    ----------
    backend : {'numpy','torch'}
        Registry to sanitize.
    check_fn : callable or None
        Predicate on tag dict; if returns False, the tag is removed.
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
    """Return the tag dict attached to `obj`, or {} if absent/unsupported."""
    backend = get_backend(obj)
    return get_tag_by_id(backend, id(obj)) if backend else {}  # type: ignore[arg-type]


def set_tag(obj: Any, tag: dict[str, Any]) -> None:
    """Attach/overwrite a tag dict on `obj` if backend is supported."""
    backend = get_backend(obj)
    if backend:
        set_tag_by_id(backend, id(obj), tag)


def has_tag(obj: Any) -> bool:
    """Return True if `obj` has a tag in its backend registry."""
    backend = get_backend(obj)
    return has_tag_by_id(backend, id(obj)) if backend else False  # type: ignore[arg-type]


def del_tag(obj: Any) -> None:
    """Delete tag for `obj` if backend is supported."""
    backend = get_backend(obj)
    if backend:
        del_tag_by_id(backend, id(obj))


def update_tag(obj: Any, updates: dict[str, Any]) -> None:
    """Update selected fields of `obj`'s tag (no-op if missing)."""
    backend = get_backend(obj)
    if backend:
        update_tag_by_id(backend, id(obj), updates)


def copy_tag(source_obj: Any, target_obj: Any) -> None:
    """
    Copy (shallow) the tag from `source_obj` to `target_obj` when backends match.

    Notes
    -----
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
    """Return the `uid` field from `obj`'s tag, if present."""
    return get_tag(obj).get("uid")


def set_tag_uid(obj: Any) -> str:
    """
    Generate a new UUID4 and store it under 'uid' in `obj`'s tag.

    Returns
    -------
    str
        UUID string (full UUID4).
    """
    uid = str(uuid.uuid4())
    update_tag(obj, {"uid": uid})
    return uid


# ====[ TAG SUMMARY – DEBUG INFO ]====
def get_tag_summary(obj: Any) -> dict[str, Any]:
    """
    Return a compact summary of the tag metadata for a given object.

    Returns
    -------
    dict
        Keys: status, uid, shape, framework, axes{...}, trace[list].
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
