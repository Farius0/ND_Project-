# ==================================================
# =============  MODULE: layout_axes  ==============
# ==================================================

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import torch

# from functools import lru_cache

__all__ = [
    "AXES_ORDER",
    "FORMAT_LAYOUTS",
    "get_layout_axes",
    "list_available_layouts",
    "is_valid_layout",
    "infer_layout",
    "get_axis_name_by_index",
    "get_axes_positions",
    "get_layout_from_axes",
    "summarize_layout",
    "validate_layout_dict",
    "describe_all_layouts",
    "LayoutResolver",
    "get_layout_from_tag",
    "resolve_and_clean_layout_tags",
]

# ====[ AXIS ORDER ]====
AXES_ORDER: List[str] = [
    "batch_axis",
    "channel_axis",
    "direction_axis",
    "depth_axis",
    "height_axis",
    "width_axis",
]

# ====[ FORMAT LAYOUTS ]====
FORMAT_LAYOUTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "torch": {
        "NCHW": {
            "name": "NCHW",
            "ndim": 4,
            "direction_axis": None,
            "batch_axis": 0,
            "channel_axis": 1,
            "depth_axis": None,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Batch, Channel, Height, Width (default for CNNs)",
        },
        "NCDHW": {
            "name": "NCDHW",
            "ndim": 5,
            "direction_axis": None,
            "batch_axis": 0,
            "channel_axis": 1,
            "depth_axis": 2,
            "height_axis": 3,
            "width_axis": 4,
            "description": "Batch, Channel, Depth, Height, Width (3D volumes)",
        },
        "NHWC": {
            "name": "NHWC",
            "ndim": 4,
            "direction_axis": None,
            "batch_axis": 0,
            "channel_axis": 3,
            "depth_axis": None,
            "height_axis": 1,
            "width_axis": 2,
            "description": "Batch, Height, Width, Channel (e.g., TensorFlow-style)",
        },
        "CHW": {
            "name": "CHW",
            "ndim": 3,
            "direction_axis": None,
            "batch_axis": None,
            "channel_axis": 0,
            "depth_axis": None,
            "height_axis": 1,
            "width_axis": 2,
            "description": "Channel, Height, Width (no batch)",
        },
        "CDHW": {
            "name": "CDHW",
            "ndim": 4,
            "direction_axis": None,
            "batch_axis": None,
            "channel_axis": 0,
            "depth_axis": 1,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Channel, Depth, Height, Width (no batch)",
        },
        "DHW": {
            "name": "DHW",
            "ndim": 3,
            "direction_axis": None,
            "batch_axis": None,
            "channel_axis": None,
            "depth_axis": 0,
            "height_axis": 1,
            "width_axis": 2,
            "description": "Pure 3D volume (no channel, no batch)",
        },
        "HW": {
            "name": "HW",
            "ndim": 2,
            "direction_axis": None,
            "batch_axis": None,
            "channel_axis": None,
            "depth_axis": None,
            "height_axis": 0,
            "width_axis": 1,
            "description": "Height, Width (common numpy format)",
        },
        "NHW": {
            "name": "NHW",
            "ndim": 3,
            "direction_axis": None,
            "batch_axis": 0,
            "channel_axis": None,
            "depth_axis": None,
            "height_axis": 1,
            "width_axis": 2,
            "description": "Batch, Height, Width (common numpy format)",
        },
        "NDHW": {
            "name": "NDHW",
            "ndim": 4,
            "direction_axis": None,
            "batch_axis": 0,
            "channel_axis": None,
            "depth_axis": 1,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Batch, Depth, Height, Width (3D volumes)",
        },
        "NC": {
            "name": "NC",
            "ndim": 2,
            "direction_axis": None,
            "batch_axis": 0,
            "channel_axis": 1,
            "depth_axis": None,
            "height_axis": None,
            "width_axis": None,
            "description": "Batch, Channel (e.g., vector/matrix data)",
        },
        "C": {
            "name": "C",
            "ndim": 1,
            "direction_axis": None,
            "batch_axis": None,
            "channel_axis": 0,
            "depth_axis": None,
            "height_axis": None,
            "width_axis": None,
            "description": "Channel only (e.g., label or feature vector)",
        },
        "GNCHW": {
            "name": "GNCHW",
            "ndim": 5,
            "direction_axis": 0,
            "batch_axis": 1,
            "channel_axis": 2,
            "depth_axis": None,
            "height_axis": 3,
            "width_axis": 4,
            "description": "Direction, Batch, Channel, Height, Width",
        },
        "GNCDHW": {
            "name": "GNCDHW",
            "ndim": 6,
            "direction_axis": 0,
            "batch_axis": 1,
            "channel_axis": 2,
            "depth_axis": 3,
            "height_axis": 4,
            "width_axis": 5,
            "description": "Direction, Batch, Channel, Depth, Height, Width",
        },
        "GHW": {
            "name": "GHW",
            "ndim": 3,
            "direction_axis": 0,
            "batch_axis": None,
            "channel_axis": None,
            "depth_axis": None,
            "height_axis": 1,
            "width_axis": 2,
            "description": "Direction, Height, Width",
        },
        "GCHW": {
            "name": "GCHW",
            "ndim": 4,
            "direction_axis": 0,
            "batch_axis": None,
            "channel_axis": 1,
            "depth_axis": None,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Direction, Channel, Height, Width",
        },
        "GCDHW": {
            "name": "GCDHW",
            "ndim": 5,
            "direction_axis": 0,
            "batch_axis": None,
            "channel_axis": 1,
            "depth_axis": 2,
            "height_axis": 3,
            "width_axis": 4,
            "description": "Direction, Channel, Depth, Height, Width",
        },
        "GNHWC": {
            "name": "GNHWC",
            "ndim": 5,
            "direction_axis": 0,
            "batch_axis": 1,
            "channel_axis": 4,
            "depth_axis": None,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Direction, Batch, Height, Width, Channel",
        },
        "GNHW": {
            "name": "GNHW",
            "ndim": 4,
            "direction_axis": 0,
            "batch_axis": 1,
            "channel_axis": None,
            "depth_axis": None,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Direction, Batch, Height, Width",
        },
        "GDHW": {
            "name": "GDHW",
            "ndim": 4,
            "direction_axis": 0,
            "batch_axis": None,
            "channel_axis": None,
            "depth_axis": 1,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Direction, Depth, Height, Width",
        },
        "GNDHW": {
            "name": "GNDHW",
            "ndim": 5,
            "direction_axis": 0,
            "batch_axis": 1,
            "channel_axis": None,
            "depth_axis": 2,
            "height_axis": 3,
            "width_axis": 4,
            "description": "Direction, Batch, Depth, Height, Width",
        },
    },
    "numpy": {
        "HW": {
            "name": "HW",
            "ndim": 2,
            "direction_axis": None,
            "batch_axis": None,
            "channel_axis": None,
            "depth_axis": None,
            "height_axis": 0,
            "width_axis": 1,
            "description": "Height, Width (common numpy format)",
        },
        "HWC": {
            "name": "HWC",
            "ndim": 3,
            "direction_axis": None,
            "batch_axis": None,
            "channel_axis": 2,
            "depth_axis": None,
            "height_axis": 0,
            "width_axis": 1,
            "description": "Height, Width, Channel (common numpy format)",
        },
        "CHW": {
            "name": "CHW",
            "ndim": 3,
            "direction_axis": None,
            "batch_axis": None,
            "channel_axis": 0,
            "depth_axis": None,
            "height_axis": 1,
            "width_axis": 2,
            "description": "Channel, Height, Width (no batch)",
        },
        "NHWC": {
            "name": "NHWC",
            "ndim": 4,
            "direction_axis": None,
            "batch_axis": 0,
            "channel_axis": 3,
            "depth_axis": None,
            "height_axis": 1,
            "width_axis": 2,
            "description": "Batch, Height, Width, Channel",
        },
        "NCHW": {
            "name": "NCHW",
            "ndim": 4,
            "direction_axis": None,
            "batch_axis": 0,
            "channel_axis": 1,
            "depth_axis": None,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Batch, Channel, Height, Width",
        },
        "NHW": {
            "name": "NHW",
            "ndim": 3,
            "direction_axis": None,
            "batch_axis": 0,
            "channel_axis": None,
            "depth_axis": None,
            "height_axis": 1,
            "width_axis": 2,
            "description": "Batch, Height, Width",
        },
        "DHW": {
            "name": "DHW",
            "ndim": 3,
            "direction_axis": None,
            "batch_axis": None,
            "channel_axis": None,
            "depth_axis": 0,
            "height_axis": 1,
            "width_axis": 2,
            "description": "Pure 3D volume (no channel, no batch)",
        },
        "HWD": {
            "name": "HWD",
            "ndim": 3,
            "direction_axis": None,
            "batch_axis": None,
            "channel_axis": None,
            "depth_axis": 2,
            "height_axis": 0,
            "width_axis": 1,
            "description": "Height, Width, Depth",
        },
        "DHWC": {
            "name": "DHWC",
            "ndim": 4,
            "direction_axis": None,
            "batch_axis": None,
            "channel_axis": 3,
            "depth_axis": 0,
            "height_axis": 1,
            "width_axis": 2,
            "description": "Depth, Height, Width, Channel",
        },
        "NDHW": {
            "name": "NDHW",
            "ndim": 4,
            "direction_axis": None,
            "batch_axis": 0,
            "channel_axis": None,
            "depth_axis": 1,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Batch, Depth, Height, Width",
        },
        "NDHWC": {
            "name": "NDHWC",
            "ndim": 5,
            "direction_axis": None,
            "batch_axis": 0,
            "channel_axis": 4,
            "depth_axis": 1,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Batch, Depth, Height, Width, Channel",
        },
        "CDHW": {
            "name": "CDHW",
            "ndim": 4,
            "direction_axis": None,
            "batch_axis": None,
            "channel_axis": 0,
            "depth_axis": 1,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Channel, Depth, Height, Width",
        },
        "GHW": {
            "name": "GHW",
            "ndim": 3,
            "direction_axis": 0,
            "batch_axis": None,
            "channel_axis": None,
            "depth_axis": None,
            "height_axis": 1,
            "width_axis": 2,
            "description": "Direction, Height, Width",
        },
        "GHWC": {
            "name": "GHWC",
            "ndim": 4,
            "direction_axis": 0,
            "batch_axis": None,
            "channel_axis": 3,
            "depth_axis": None,
            "height_axis": 1,
            "width_axis": 2,
            "description": "Direction, Height, Width, Channel",
        },
        "GNHWC": {
            "name": "GNHWC",
            "ndim": 5,
            "direction_axis": 0,
            "batch_axis": 1,
            "channel_axis": 4,
            "depth_axis": None,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Direction, Batch, Height, Width, Channel",
        },
        "GNHW": {
            "name": "GNHW",
            "ndim": 4,
            "direction_axis": 0,
            "batch_axis": 1,
            "channel_axis": None,
            "depth_axis": None,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Direction, Batch, Height, Width",
        },
        "GCHW": {
            "name": "GCHW",
            "ndim": 4,
            "direction_axis": 0,
            "batch_axis": None,
            "channel_axis": 1,
            "depth_axis": None,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Direction, Channel, Height, Width",
        },
        "GNCHW": {
            "name": "GNCHW",
            "ndim": 5,
            "direction_axis": 0,
            "batch_axis": 1,
            "channel_axis": 2,
            "depth_axis": None,
            "height_axis": 3,
            "width_axis": 4,
            "description": "Direction, Batch, Channel, Height, Width",
        },
        "GNCDHW": {
            "name": "GNCDHW",
            "ndim": 6,
            "direction_axis": 0,
            "batch_axis": 1,
            "channel_axis": 2,
            "depth_axis": 3,
            "height_axis": 4,
            "width_axis": 5,
            "description": "Direction, Batch, Channel, Depth, Height, Width",
        },
        "GCDHW": {
            "name": "GCDHW",
            "ndim": 5,
            "direction_axis": 0,
            "batch_axis": None,
            "channel_axis": 1,
            "depth_axis": 2,
            "height_axis": 3,
            "width_axis": 4,
            "description": "Direction, Channel, Depth, Height, Width",
        },
        "GDHW": {
            "name": "GDHW",
            "ndim": 4,
            "direction_axis": 0,
            "batch_axis": None,
            "channel_axis": None,
            "depth_axis": 1,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Direction, Depth, Height, Width",
        },
        "GNDHW": {
            "name": "GNDHW",
            "ndim": 5,
            "direction_axis": 0,
            "batch_axis": 1,
            "channel_axis": None,
            "depth_axis": 2,
            "height_axis": 3,
            "width_axis": 4,
            "description": "Direction, Batch, Depth, Height, Width",
        },
    },
    # Kept for completeness; not used for runtime lookups by BaseConverter.
    "auto": {
        "default": {
            "name": "default",
            "ndim": 4,
            "direction_axis": None,
            "batch_axis": 0,
            "channel_axis": 1,
            "depth_axis": None,
            "height_axis": 2,
            "width_axis": 3,
            "description": "Default fallback format",
        }
    },
}


# ==================================================
# ============== FORMAT UTILITIES ==================
# ==================================================

# @lru_cache(maxsize=None)
def get_layout_axes(framework: str, layout_name: str) -> Dict[str, Any]:
    """
    Retrieve the axes layout (batch, channel, direction, ...) for a framework+layout.

    Parameters
    ----------
    framework : {'torch','numpy','auto'}
        Target framework.
    layout_name : str
        Layout mnemonic (e.g., 'NCHW', 'HWC', 'GNCHW').

    Returns
    -------
    dict
        Copy of the layout dict, with at least keys in AXES_ORDER, 'ndim', 'name'.

    Raises
    ------
    ValueError
        If the framework or the layout are unknown.
    """
    fw = framework.lower()
    name = layout_name.upper()

    if fw not in FORMAT_LAYOUTS:
        raise ValueError(
            f"Unknown framework '{framework}'. "
            f"Available: {', '.join(sorted(FORMAT_LAYOUTS.keys()))}"
        )

    layouts = FORMAT_LAYOUTS[fw]
    if name not in layouts:
        available = ", ".join(sorted(layouts.keys()))
        raise ValueError(
            f"Unknown layout '{layout_name}' for framework '{framework}'. "
            f"Available: {available}"
        )

    layout = layouts[name].copy()
    layout.setdefault("name", name)
    return layout


def list_available_layouts(framework: str) -> List[str]:
    """List all layout names available for a given framework."""
    return list(FORMAT_LAYOUTS.get(framework.lower(), {}).keys())


def is_valid_layout(framework: str, layout_name: str) -> bool:
    """Return True if the layout exists for the framework."""
    return layout_name.upper() in FORMAT_LAYOUTS.get(framework.lower(), {})


# Framework-specific fallback when inference fails
_DEFAULT_FALLBACK: Dict[str, str] = {
    "torch": "NCHW",
    "numpy": "HWC",
}


def infer_layout(image: Any, framework: str = "auto") -> str:
    """
    Infer a likely layout string from image shape and framework.

    Notes
    -----
    - Heuristics are intentionally simple and conservative.
    - If the guess isn't valid for the target framework, a framework-specific
      fallback is returned ('NCHW' for torch, 'HWC' for numpy).

    Returns
    -------
    str
        Layout string valid for the (possibly deduced) framework.
    """
    shape = getattr(image, "shape", None)
    if shape is None:
        raise TypeError("Object has no 'shape' attribute for layout inference.")

    # Deduce framework when 'auto'
    if framework == "auto":
        framework = "torch" if torch.is_tensor(image) else "numpy"

    # Guess a standard mnemonic (e.g., 'HWC', 'NCHW', ...)
    layout_str = LayoutResolver.guess_from_shape(tuple(shape), strict=False)

    # Validate against selected framework; otherwise return per-framework fallback
    if is_valid_layout(framework, layout_str):
        return layout_str

    return _DEFAULT_FALLBACK.get(framework, layout_str)


def get_axis_name_by_index(framework: str, layout_name: str, index: int) -> Optional[str]:
    """
    Return the semantic axis name corresponding to a given index in the layout.

    Parameters
    ----------
    framework : str
        Framework to use for layout resolution ('numpy' or 'torch').
    layout_name : str
        Layout string (e.g., 'HWC', 'NCHW', etc.).
    index : int
        Axis index to look up.

    Returns
    -------
    str or None
        Semantic axis name (e.g., 'height_axis', 'channel_axis'), or None if not found.
    """
    layout = get_layout_axes(framework, layout_name)
    for key in AXES_ORDER:
        if layout.get(key) == index:
            return key
    return None


def get_axes_positions(framework: str, layout_name: str) -> Dict[str, int]:
    """
    Return a dictionary mapping semantic axis names to their positions for a given layout.

    Parameters
    ----------
    framework : str
        Framework used to resolve the layout ('numpy' or 'torch').
    layout_name : str
        Layout string to interpret (e.g., 'HWC', 'NCHW', etc.).

    Returns
    -------
    Dict[str, int]
        Dictionary of axis names (e.g., 'height_axis') and their corresponding indices,
        excluding any undefined (None) axes.
    """
    layout = get_layout_axes(framework, layout_name)
    return {k: v for k, v in layout.items() if k in AXES_ORDER and v is not None}


def get_layout_from_axes(
    framework: str, axes: Dict[str, Optional[int]], ndim: Optional[int] = None, strict: bool = True
) -> Optional[str]:
    """
    Return a layout name that matches a given axes dict (and optional ndim).

    Parameters
    ----------
    framework : {'torch','numpy'}
    axes : dict
        Mapping like {'batch_axis': 0, 'channel_axis': 1, ...}.
    ndim : int, optional
        If provided, only layouts with the same ndim are considered.
    strict : bool, default True
        If True, all provided non-None axes must match the candidate layout.

    Returns
    -------
    str | None
        Matching layout name, else None.
    """
    candidates = FORMAT_LAYOUTS.get(framework.lower(), {})
    for name, layout in candidates.items():
        if ndim is not None and layout.get("ndim") != ndim:
            continue
        # Compare only provided keys among AXES_ORDER
        kv = {k: v for k, v in axes.items() if k in AXES_ORDER and v is not None}
        match = all(layout.get(k) == v for k, v in kv.items())
        if match:
            return name if not strict else name
    return None


def summarize_layout(framework: str, layout_name: str) -> None:
    """
    Pretty-print a human-readable summary of axis roles and their positions in the layout.

    Parameters
    ----------
    framework : str
        Framework used for layout interpretation ('numpy' or 'torch').
    layout_name : str
        Layout string to summarize (e.g., 'HWC', 'NCHW').

    Returns
    -------
    None
        The function prints the layout summary to stdout.
    """
    layout = get_layout_axes(framework, layout_name)
    print(f"\n[Layout: '{layout_name}' – Framework: {framework}]")
    for key in sorted(layout.keys()):
        print(f"  {key:<20}: {layout[key]}")
    if "description" in layout:
        print(f"\n  ↪ Description: {layout['description']}")


def validate_layout_dict(layout: Dict[str, Any]) -> None:
    """
    Validate that the layout dictionary is structurally correct and contains valid axis mappings.

    Parameters
    ----------
    layout : Dict[str, Any]
        Dictionary defining axis names and their corresponding indices.

    Returns
    -------
    None
        Raises a ValueError or TypeError if the layout is invalid.
    """
    if "ndim" not in layout:
        raise ValueError("Missing 'ndim' in layout definition.")
    for axis in AXES_ORDER:
        if axis in layout and layout[axis] is not None and not isinstance(layout[axis], int):
            raise TypeError(f"{axis} must be an int or None.")


def describe_all_layouts() -> None:
    """Display all defined layouts for all frameworks."""
    for fw in FORMAT_LAYOUTS:
        for name in FORMAT_LAYOUTS[fw]:
            summarize_layout(fw, name)


# ====[ LAYOUT RESOLVER – AXIS-BASED NAMING SYSTEM ]====

class LayoutResolver:
    """
    Static utilities to generate or interpret axis layouts from flags or dicts.
    """

    @classmethod
    def from_flags(
        cls,
        direction: bool = False,
        batch: bool = False,
        channel: bool = True,
        depth: bool = False,
        height: bool = True,
        width: bool = True,
        channel_position: str = "last",
    ) -> str:
        """
        Build a layout string based on axis presence and preferred channel position.

        Parameters
        ----------
        direction : bool, optional
            Whether to include a direction axis ('D').
        batch : bool, optional
            Whether to include a batch axis ('N').
        channel : bool, optional
            Whether to include a channel axis ('C').
        depth : bool, optional
            Whether to include a depth axis ('Z') for 3D data.
        height : bool, optional
            Whether to include a height axis ('H').
        width : bool, optional
            Whether to include a width axis ('W').
        channel_position : {'first', 'last'}, optional
            Placement of the channel axis: 'first' (e.g., 'CHW') or 'last' (e.g., 'HWC').

        Returns
        -------
        str
            Constructed layout string based on the specified flags.

        Notes
        -----
        - The channel axis ('C') is placed according to `channel_position`.
        - Axes are added in the order: batch, direction, spatial, channel.
        """

        order: List[str] = []

        if direction:
            order.append("G")
        if batch:
            order.append("N")
        if channel and channel_position == "first":
            order.append("C")

        if depth:
            order.append("D")
        if height:
            order.append("H")
        if width:
            order.append("W")

        if channel and channel_position == "last":
            order.append("C")

        return "".join(order)

    @classmethod
    def to_axes_dict(cls, layout_str: str, strict: bool = True) -> Dict[str, int]:
        """
        Convert a layout string (e.g., 'NCHW') into a dictionary mapping axis roles to indices.

        Parameters
        ----------
        layout_str : str
            Layout string where each character represents an axis (e.g., 'N' = batch, 'C' = channel).
        strict : bool, optional
            If True, raises an error on unknown characters or duplicated roles.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping semantic axis names (e.g., 'batch_axis', 'channel_axis') to positions.

        Raises
        ------
        ValueError
            If an unknown character is found or roles are duplicated (when strict=True).
        """

        axis_roles = {
            "G": "direction_axis",
            "N": "batch_axis",
            "C": "channel_axis",
            "D": "depth_axis",
            "H": "height_axis",
            "W": "width_axis",
        }

        layout = layout_str.upper()
        result: Dict[str, int] = {}

        for i, c in enumerate(layout):
            if c not in axis_roles:
                if strict:
                    raise ValueError(f"Invalid axis character '{c}' in layout '{layout_str}'")
                else:
                    continue
            role = axis_roles[c]
            if role in result and strict:
                raise ValueError(f"Duplicate role '{role}' in layout '{layout_str}'")
            result[role] = i

        return result

    @classmethod
    def to_flag_dict(cls, layout_str: str, strict: bool = True) -> Dict[str, bool]:
        """
        Convert a layout string (e.g., 'NCHW') into a dictionary of axis presence flags.

        Parameters
        ----------
        layout_str : str
            Layout string where each character represents a semantic axis (e.g., 'N' for batch).
        strict : bool, optional
            If True, raises an error for unknown or duplicated characters.

        Returns
        -------
        Dict[str, bool]
            Dictionary indicating the presence (True/False) of each axis role:
            {'batch': True, 'channel': True, 'height': True, 'width': True, ...}

        Raises
        ------
        ValueError
            If unknown or duplicated axis characters are encountered (when strict=True).
        """
        role_flags = {
            "G": "direction",
            "N": "batch",
            "C": "channel",
            "D": "depth",
            "H": "height",
            "W": "width",
        }

        flags: Dict[str, bool] = {k: False for k in role_flags.values()}
        layout = layout_str.upper()

        for c in layout:
            if c not in role_flags:
                if strict:
                    raise ValueError(f"Invalid layout character '{c}' in '{layout_str}'")
                continue
            flags[role_flags[c]] = True

        return flags

    @classmethod
    def describe_layout(cls, layout_str: str, as_string: bool = False, strict: bool = True) -> Optional[str]:
        """
        Print or return a human-readable summary of a layout string and its axis roles.

        Parameters
        ----------
        layout_str : str
            Layout string to describe (e.g., 'NCHW').
        as_string : bool, optional
            If True, return the summary as a string instead of printing it.
        strict : bool, optional
            If True, enforce strict validation of the layout string.

        Returns
        -------
        str or None
            Summary string if `as_string=True`; otherwise None (output is printed).
        """
        try:
            axes = cls.to_axes_dict(layout_str, strict=strict)
            flags = cls.to_flag_dict(layout_str, strict=strict)
        except Exception as e:
            msg = f"[Layout: {layout_str}] → Error: {e}"
            if as_string:
                return msg
            print(msg)
            return None

        lines = [f"[Layout: '{layout_str}']"]
        lines += [f"  {k:<20}: {axes[k]}" for k in sorted(axes)]
        lines += ["  ---"]
        lines += [f"  {k:<20}: {v}" for k, v in flags.items()]

        joined = "\n".join(lines)
        if as_string:
            return joined
        print(joined)
        return None

    @staticmethod
    def guess_from_shape(shape: Tuple[int, ...], strict: bool = True) -> str:
        """
        Heuristically guess a layout string (e.g., 'HWC', 'NCHW') from a shape tuple.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the array or tensor to analyze.
        strict : bool, optional
            If True, raises an error when no valid layout can be guessed.

        Returns
        -------
        str
            Guessed layout string based on the number and size of dimensions.

        Notes
        -----
        - Prefers common image conventions (e.g., 1/3/4-channel formats).
        - 4D shapes fall back to 'NCHW' or 'NHWC' when plausible; otherwise 'CDHW'.
        - 5D shapes may return 'GNCHW'/'GNHWC' for grouped inputs, or 'NDHWC'.

        Raises
        ------
        ValueError
            If no reasonable layout can be inferred and strict=True.
        """
        ndim = len(shape)

        if ndim == 2:
            return "HW"

        elif ndim == 3:
            if shape[-1] in (1, 3, 4):  # ...HWC
                return "HWC"
            elif shape[0] in (1, 3, 4):  # CHW...
                return "CHW"
            else:
                return "DHW"  # 3D grayscale volume

        elif ndim == 4:
            if shape[0] < 16 and shape[1] in (1, 3, 4):  # likely batch + channels
                return "NCHW"
            elif shape[-1] in (1, 3, 4):
                return "NHWC"
            else:
                return "CDHW"  # volumetric with channels

        elif ndim == 5:
            if shape[1] in (1, 3, 4):
                return "GNCHW"
            if shape[-1] in (1, 3, 4):
                return "GNHWC"
            return "NDHWC"

        if strict:
            raise ValueError(f"[LayoutResolver] Unable to guess layout from shape {shape}.")
        else:
            return "unknown"


# ====[ Infer Layout Name from Tag Dict ]====
def get_layout_from_tag(tag: Dict[str, Any], framework: str = "numpy") -> Optional[str]:
    """
    Infer the most probable layout name from a tag dictionary.

    Parameters
    ----------
    tag : dict
        Tag with axis definitions (e.g., from conversion).
    framework : {'torch','numpy'}
        Target framework.

    Returns
    -------
    str | None
        Layout name if matched, else None.
    """
    if not isinstance(tag, dict):
        raise TypeError("Tag must be a dictionary.")

    axes = {k: tag.get(k) for k in AXES_ORDER if tag.get(k) is not None}
    ndim = tag.get("ndim", None)
    return get_layout_from_axes(framework, axes, ndim=ndim, strict=False)


def resolve_and_clean_layout_tags(
    tagger: Any,
    framework: str,
    fallback_layout_name: str = "HWC",
    prefix: str = "G",
    remove_prefix: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Extract and clean axis layout tags for ND tagging.

    Parameters
    ----------
    tagger : AxisTracker-like
        Object exposing `get_tag()` which returns a tag dict.
    framework : {'torch','numpy'}
        Target framework.
    fallback_layout_name : str, default 'HWC'
        Used when no layout name exists in the tag.
    prefix : str, default 'G'
        Prefix to ensure the layout includes (e.g., 'G' for gradients).
    remove_prefix : bool, default False
        If True, remove leading `prefix` from the layout name (when present).

    Returns
    -------
    (layout_name, axes) : (str, dict)
        Clean layout name and its axes dict (keys in AXES_ORDER only).
    """
    layout_name = tagger.get_tag().get("layout_name", None)

    # Construct a clean layout name
    if layout_name is None:
        layout_name = prefix + fallback_layout_name if not remove_prefix else fallback_layout_name
    elif remove_prefix and layout_name.startswith(prefix):
        layout_name = layout_name[len(prefix) :]
    elif not remove_prefix and not layout_name.startswith(prefix):
        layout_name = prefix + layout_name

    # Get layout axes from clean name, without touching original
    axes = get_layout_axes(framework, layout_name)
    axes.pop("name", None)
    axes.pop("description", None)

    return layout_name, axes
