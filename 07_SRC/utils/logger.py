# ==================================================
# ================ Logger Utilities ================
# ==================================================
from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Public API
__all__ = ["DEFAULT_LOG_DIR", "make_file_handler", "get_logger", "get_error_logger", "get_debug_logger"]

# ====[ Global logging configuration ]====
DEFAULT_LOG_DIR: Path = (Path(__file__).parent.parent / "logs").resolve()


def _sync_handler_levels(logger: logging.Logger, level: int) -> None:
    """
    Ensure that all existing handlers attached to a logger use the specified log level.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance whose handlers will be updated.
    level : int
        Logging level to apply to all handlers (e.g., logging.INFO, logging.DEBUG).

    Returns
    -------
    None
        This function modifies handlers in-place and does not return anything.
    """

    for h in logger.handlers:
        try:
            h.setLevel(level)
        except Exception:
            pass


# ====[ Global logging configuration ]====
DEFAULT_LOG_DIR = Path.cwd().parent / "logs"


# ====[ Shared rotating file handler generator ]====
def make_file_handler(
    log_path: Union[str, Path],
    level: int,
    when: str = "midnight",
    backupCount: int = 7,
    encoding: str = "utf-8",
    interval: int = 1,
) -> TimedRotatingFileHandler:
    """
    Create a TimedRotatingFileHandler with a standard formatter.

    Parameters
    ----------
    log_path : str | Path
        Output log file path.
    level : int
        Logging level (e.g., logging.INFO).
    when : str, default 'midnight'
        Rotation interval basis per logging.handlers.TimedRotatingFileHandler.
    backupCount : int, default 7
        Number of backup files to keep.
    encoding : str, default 'utf-8'
        File encoding.
    interval : int, default 1
        Rotation interval multiplier.

    Returns
    -------
    TimedRotatingFileHandler
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = TimedRotatingFileHandler(
        filename=str(path),
        when=when,
        interval=interval,
        backupCount=backupCount,
        encoding=encoding,
        delay=True,  # open file on first emit (safer for tests/CLI)
    )
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    return handler

# ==================================================
# ================ Logger Factory ==================
# ==================================================

# ====[ Main logger: console + file ]====
def get_logger(
    name: str = "global_logger",
    log_dir: Optional[Union[str, Path]] = DEFAULT_LOG_DIR,
    level: int = logging.INFO,
    when: str = "midnight",
    backupCount: int = 7,
) -> logging.Logger:
    """
    Create and configure a logger with both console and file handlers (daily rotation).

    The logger is idempotent: repeated calls with the same `name` do not add duplicate handlers.
    Log propagation is disabled to avoid duplicate messages from the root logger.

    Parameters
    ----------
    name : str, optional
        Name of the logger instance. Default is "global_logger".
    log_dir : str or Path, optional
        Directory where log files will be stored. If None, use DEFAULT_LOG_DIR.
    level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG). Default is logging.INFO.
    when : str, optional
        Time interval for log file rotation (e.g., "midnight", "D", "H"). Default is "midnight".
    backupCount : int, optional
        Number of backup log files to keep. Default is 7.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        base_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        log_path = base_dir / f"{name}_{today}.log"

        file_handler = make_file_handler(log_path, level, when=when, backupCount=backupCount)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    else:
        _sync_handler_levels(logger, level)

    return logger


# ====[ Error logger: file only ]====
def get_error_logger(
    name: str = "error_logger",
    log_dir: Optional[Union[str, Path]] = DEFAULT_LOG_DIR,
    level: int = logging.ERROR,
    backupCount: int = 30,
) -> logging.Logger:
    """
    Create and configure a dedicated error logger with daily rotating file output.

    Unlike the main logger, this logger does not write to the console. It stores
    only error-level (or higher) messages in a rotating log file.

    Parameters
    ----------
    name : str, optional
        Name of the logger instance. Default is "error_logger".
    log_dir : str or Path, optional
        Directory where error logs will be stored. If None, use DEFAULT_LOG_DIR.
    level : int, optional
        Logging level to capture. Default is logging.ERROR.
    backupCount : int, optional
        Number of daily log files to retain. Default is 30.

    Returns
    -------
    logging.Logger
        Configured error logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        base_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        log_path = base_dir / f"errors_{today}.log"

        file_handler = make_file_handler(log_path, level, when="midnight", backupCount=backupCount)
        logger.addHandler(file_handler)
    else:
        _sync_handler_levels(logger, level)

    return logger

# ====[ Debug logger: file only ]====
def get_debug_logger(
    name: str = "debug_logger",
    log_dir: Optional[Union[str, Path]] = DEFAULT_LOG_DIR,
    level: int = logging.DEBUG,
    backupCount: int = 7,
) -> logging.Logger:
    """
    Create and configure a dedicated debug logger with daily rotating file output.

    This logger records debug-level messages and above to a file, but does not
    print to the console. Useful for capturing detailed traces during development.

    Parameters
    ----------
    name : str, optional
        Name of the logger instance. Default is "debug_logger".
    log_dir : str or Path, optional
        Directory where debug logs will be stored. If None, use DEFAULT_LOG_DIR.
    level : int, optional
        Logging level to capture. Default is logging.DEBUG.
    backupCount : int, optional
        Number of daily log files to retain. Default is 7.

    Returns
    -------
    logging.Logger
        Configured debug logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        base_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        log_path = base_dir / f"debug_{today}.log"

        file_handler = make_file_handler(log_path, level, when="midnight", backupCount=backupCount)
        logger.addHandler(file_handler)
    else:
        _sync_handler_levels(logger, level)

    return logger
