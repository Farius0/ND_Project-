# ==================================================
# ===============  MODULE: emojis  =================
# ==================================================
from __future__ import annotations
from typing import Dict, Optional

__all__ = [
    "USE_EMOJIS",
    "enable_emojis",
    "emojis_enabled",
    "emojize",
    "deemojize",
    "emojis_dict",
]

# Global switch
USE_EMOJIS: bool = True

def enable_emojis(flag: bool = True) -> None:
    """
    Enable or disable emoji usage globally across the application.

    This function sets a global flag controlling whether emojis are used
    in logs, progress bars, or UI outputs.

    Parameters
    ----------
    flag : bool, optional
        If True, enable emojis. If False, disable them. Default is True.

    Returns
    -------
    None
        The global flag `USE_EMOJIS` is updated in place.
    """

    global USE_EMOJIS
    USE_EMOJIS = bool(flag)

def emojis_enabled() -> bool:
    """
    Check whether emoji usage is currently enabled.

    Returns
    -------
    bool
        True if emojis are enabled globally, False otherwise.
    """

    return USE_EMOJIS

# Raw emoji table
_EMOJI_TABLE_RAW: Dict[str, str] = {
    "input": "📥",
    "output": "📤",
    "time": "⏱️",
    "success": "✅",
    "error": "❗",
    "warning": "⚠️",
    "info": "ℹ️",
    "debug": "🐞",
    "start": "🚀",
    "stop": "🛑",
    "load": "📦",
    "save": "💾",
    "compute": "🧠",
    "loop": "🔁",
    "check": "🔍",
    "fire": "🔥",
    "memory": "🧮",
    "network": "🌐",
    "disk": "🗃️",
    "folder": "📁",
    "file": "📄",
    "lock": "🔒",
    "unlock": "🔓",
    "camera": "📷",
    "image": "🖼️",
    "graph": "📊",
    "alert": "🚨",
    "cool": "😎",
    "spark": "✨",
}

# Reverse lookup for deemojize
_REVERSE_TABLE: Dict[str, str] = {v: k for k, v in _EMOJI_TABLE_RAW.items()}

def emojize(name: str, default: Optional[str] = None) -> str:
    """
    Return the corresponding emoji if emojis are enabled, otherwise return fallback text.

    Looks up the emoji by name. If not found or if emojis are disabled globally,
    returns the default fallback string or the name itself.

    Parameters
    ----------
    name : str
        Name or key of the emoji to retrieve.
    default : str, optional
        Fallback string to return if emojis are disabled or the key is not found.
        If None, uses `name` as fallback.

    Returns
    -------
    str
        The emoji character or the fallback text.
    """

    if name in _EMOJI_TABLE_RAW:
        return _EMOJI_TABLE_RAW[name] if USE_EMOJIS else name
    return default or name

def deemojize(symbol: str, default: Optional[str] = None) -> str:
    """
    Convert an emoji symbol back to its key name.

    Parameters
    ----------
    symbol : str
        The emoji symbol to convert.
    default : str, optional
        Fallback if the symbol is not found.

    Returns
    -------
    str
        The text key corresponding to the emoji, or the default, or the symbol itself.
    """
    return _REVERSE_TABLE.get(symbol, default or symbol)

def emojis_dict() -> Dict[str, str]:
    """
    Return a dictionary mapping emoji keys to their corresponding values.

    If emojis are enabled, returns key → emoji character.
    If disabled, returns key → key text (fallback as plain text).

    Returns
    -------
    Dict[str, str]
        Dictionary of emoji mappings depending on the global emoji setting.
    """

    if USE_EMOJIS:
        return dict(_EMOJI_TABLE_RAW)
    return {k: k for k in _EMOJI_TABLE_RAW}
