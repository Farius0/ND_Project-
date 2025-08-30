# ==================================================
# ========  MODULE: decorators & timing utils  =====
# ==================================================
from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Literal, Optional, TypeVar

from utils.logger import get_logger, get_error_logger, get_debug_logger

# Public API

__all__ = [
    "timer",
    "TimerManager",
    "log_exceptions",
    "timed_wrapper",
    "safe_timer",
    "safe_timer_and_debug",
    "log_warning_if",
    "log_warning_if_decorator",
    "get_all_loggers",
]

F = TypeVar("F", bound=Callable[..., Any])

# ====[ Basic print-based timer ]====
def timer(
    return_result: bool = False,
    return_elapsed: bool = False,
    name: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Simple print-based timing decorator (debug usage).

    Parameters
    ----------
    return_result : bool, default False
        If True, wrapper returns the function's result (always recommended).
    return_elapsed : bool, default False
        If True, wrapper returns the elapsed time (seconds, float) or (result, elapsed) if both True.
    name : str, optional
        Custom label in the printed message (defaults to function name).

    Returns
    -------
    Callable
        A decorator preserving the original signature.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            label = name or func.__name__
            print(f"Execution time for '{label}': {elapsed:.3f} seconds")
            if return_result and return_elapsed:
                return result, elapsed
            if return_result:
                return result
            if return_elapsed:
                return elapsed
        return wrapper
    return decorator

# ====[ Timing manager for cumulative profiling ]====
class TimerManager:
    """
    Lightweight utility for tracking cumulative timing statistics for named tasks.

    This class stores the total elapsed time and the number of occurrences
    for each task label added, which can later be used for profiling or logging.
    """
    def __init__(self):
        self.stats = {}

    def add(self, name: str, elapsed: float) -> None:
        """
        Add an elapsed time entry for a given task name.

        Parameters
        ----------
        name : str
            Identifier for the timed task (e.g., "inference", "preprocessing").
        elapsed : float
            Time duration in seconds to add to the statistics.
        """

        info = self.stats.setdefault(name, {"total": 0.0, "count": 0})
        info["total"] = float(info["total"]) + float(elapsed)
        info["count"] = int(info["count"]) + 1

    def to_dict(self, digits: int = 3) -> Dict[str, Dict[str, float]]:
        """
        Convert the collected timing statistics into a dictionary.

        Each entry contains the total elapsed time, the number of calls,
        and the average time per call, optionally rounded to a given number of digits.

        Parameters
        ----------
        digits : int, optional
            Number of decimal places to round the results. Default is 3.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping task names to timing stats:
            {
                "task_name": {
                    "total": float,
                    "count": int,
                    "avg": float
                },
                ...
            }
        """

        out: Dict[str, Dict[str, float]] = {}
        for name, info in self.stats.items():
            total = float(info["total"])
            count = int(info["count"])
            avg = total / count if count > 0 else 0.0
            out[name] = {"total": round(total, digits), "count": float(count), "avg": round(avg, digits)}
        return out

    def to_list(
        self,
        sort_by: Literal["total", "avg"] = "total",
        descending: bool = True,
        digits: int = 3,
    ) -> list[tuple[str, int, float, float]]:
        """
        Return timing statistics as a sorted list of tuples.

        Each entry contains the task name, number of calls, total time,
        and average time per call.

        Parameters
        ----------
        sort_by : {"total", "avg"}, optional
            Metric to sort the list by: total time or average time. Default is "total".
        descending : bool, optional
            If True, sort from highest to lowest. Default is True.
        digits : int, optional
            Number of decimal places to round the floating-point values. Default is 3.

        Returns
        -------
        list of tuple
            List of (name, count, total, avg) for each task.
        """

        rows: list[tuple[str, int, float, float]] = []
        for name, info in self.stats.items():
            count = int(info["count"])
            total = round(float(info["total"]), digits)
            avg = round((float(info["total"]) / count) if count > 0 else 0.0, digits)
            rows.append((name, count, total, avg))
        key_idx = 2 if sort_by == "total" else 3
        rows.sort(key=lambda x: x[key_idx], reverse=descending)
        return rows

    def to_log(
        self,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO,
        sort_by: Literal["total", "avg"] = "total",
        descending: bool = True,
        digits: int = 3,
    ) -> None:
        """
        Log the current timing statistics using a provided logger.

        The output includes the task name, call count, total time, and average time,
        formatted and sorted according to the provided criteria.

        Parameters
        ----------
        logger : logging.Logger or None, optional
            Logger to use for output. If None, a default logger is used.
        level : int, optional
            Logging level (e.g., logging.INFO, logging.DEBUG). Default is INFO.
        sort_by : {"total", "avg"}, optional
            Metric to sort the output by. Default is "total".
        descending : bool, optional
            If True, sort results in descending order. Default is True.
        digits : int, optional
            Number of decimal places for time values. Default is 3.

        Returns
        -------
        None
            The function logs the stats but does not return anything.
        """

        logger = logger or get_logger()
        logger.log(level, "Execution Time Summary:")
        for name, count, total, avg in self.to_list(sort_by=sort_by, descending=descending, digits=digits):
            logger.log(level, f" | {name:<20} | {count:>3} calls | {total:>8.3f}s total | {avg:>8.3f}s avg")

    def summary(
        self,
        sort_by: Literal["total", "avg"] = "total",
        descending: bool = True,
        digits: int = 3,
    ) -> None:
        """
        Print a human-readable summary of the timing statistics to the console.

        The summary includes the task name, number of calls, total elapsed time,
        and average time per call, formatted and sorted as specified.

        Parameters
        ----------
        sort_by : {"total", "avg"}, optional
            Metric to sort the summary by. Default is "total".
        descending : bool, optional
            If True, sort the summary in descending order. Default is True.
        digits : int, optional
            Number of decimal places for time values. Default is 3.

        Returns
        -------
        None
            This method prints to standard output and returns nothing.
        """

        print("\nExecution Time Summary:")
        for name, count, total, avg in self.to_list(sort_by=sort_by, descending=descending, digits=digits):
            print(f" | {name:<20} | {count:>3} calls | {total:>8.3f}s total | {avg:>8.3f}s avg")

    def decorator(self, name: Optional[str] = None) -> Callable[[F], F]:
        """
        Decorator that accumulates elapsed times under `name` (or function name).
        """
        def wrapper_decorator(func: F) -> F:
            label = name or func.__name__

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                self.add(label, end - start)
                return result

            return wrapper
        return wrapper_decorator

# ====[ Exception logger decorator ]====
def log_exceptions(
    logger_name: str = "error_logger",
    raise_exception: bool = False,
) -> Callable[[F], F]:
    """
    Log exceptions raised by the wrapped function.

    Parameters
    ----------
    logger_name : str, default 'error_logger'
        Name used to get the error logger.
    raise_exception : bool, default False
        If True, re-raise the exception after logging.

    Returns
    -------
    Callable
        A decorator that logs exceptions and optionally re-raises.
    """
    error_logger = get_error_logger(name=logger_name)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_logger.error(f"Exception in '{func.__name__}': {e}", exc_info=True)
                if raise_exception:
                    raise
                return None
        return wrapper
    return decorator

# ====[ Shared timing and error handler core ]====
def timed_wrapper(
    func: F,
    label: str,
    log: bool = True,
    log_errors: bool = True,
    raise_exception: bool = False,
    log_inputs: bool = False,
    log_output: bool = False,
    return_result: bool = False,
    info_logger: Optional[logging.Logger] = None,
    error_logger: Optional[logging.Logger] = None,
    debug_logger: Optional[logging.Logger] = None,
    warning_logger: Optional[logging.Logger] = None,
    warning_message: Optional[str] = None,
) -> F:
    """
    Wrap a function with timing, logging, and optional error handling.

    This utility measures the execution time of the target function and optionally logs:
    - function inputs and outputs,
    - info/debug/error/warning messages,
    - raised exceptions with optional re-raising.

    Parameters
    ----------
    func : Callable
        Function to wrap.
    label : str
        Label used for logging and timing identification.
    log : bool, optional
        If True, log execution time via `info_logger`. Default is True.
    log_errors : bool, optional
        If True, log any exceptions raised via `error_logger`. Default is True.
    raise_exception : bool, optional
        If True, re-raise any caught exceptions. If False, suppress them. Default is False.
    log_inputs : bool, optional
        If True, log the input arguments to the function. Default is False.
    log_output : bool, optional
        If True, log the function's return value. Default is False.
    return_result : bool, optional
        If True, return the result of the function. If False, return nothing. Default is False.
    info_logger : logging.Logger, optional
        Logger used for informational messages (e.g., timing logs).
    error_logger : logging.Logger, optional
        Logger used for error messages when exceptions occur.
    debug_logger : logging.Logger, optional
        Logger used for debugging (e.g., input/output values).
    warning_logger : logging.Logger, optional
        Logger used for warning messages (e.g., fallback behaviors).
    warning_message : str, optional
        Message to send via the warning logger if needed.

    Returns
    -------
    Callable
        The wrapped function with timing, logging, and error handling applied.

    Notes
    -----
    - Elapsed time is computed using a high-resolution timer and is not rounded.
    - Logging messages round time to 3 decimal places for readability.
    """

    info_logger = info_logger or get_logger()
    error_logger = error_logger or get_error_logger()
    debug_logger = debug_logger or get_debug_logger()
    warning_logger = warning_logger or get_logger()

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if log_inputs:
            debug_logger.debug(f"Calling '{label}' with args={args}, kwargs={kwargs}")

        if warning_message:
            warning_logger.warning(f"[{label}] {warning_message}")

        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                error_logger.error(f"Exception in '{label}': {e}", exc_info=True)
            if raise_exception:
                raise
            return (None, None) if return_result else None

        elapsed = time.perf_counter() - start

        if log:
            info_logger.info(f"Execution time for '{label}': {elapsed:.3f} seconds")
        if log_output:
            debug_logger.debug(f"Output of '{label}': {result!r}")

        return (result, elapsed) if return_result else result

    return wrapper

# ====[ Combined safe timer ]====
def safe_timer(
    log: bool = True,
    log_errors: bool = True,
    raise_exception: bool = False,
    name: Optional[str] = None,
    return_result: bool = False,
    info_logger: Optional[logging.Logger] = None,
    error_logger: Optional[logging.Logger] = None,
    warning_logger: Optional[logging.Logger] = None,
    warning_message: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Production-grade decorator to time and monitor function execution with logging.

    Wraps a function with a `timed_wrapper`, logging execution time and optionally
    logging exceptions or results. Designed for robust and silent monitoring
    in production pipelines (no print statements, logger-based only).

    Parameters
    ----------
    log : bool, optional
        If True, log execution time via `info_logger`. Default is True.
    log_errors : bool, optional
        If True, log exceptions via `error_logger`. Default is True.
    raise_exception : bool, optional
        If True, re-raise any exception that occurs. Otherwise, suppress it. Default is False.
    name : str or None, optional
        Name used in logs to identify the function. If None, uses the function's name.
    return_result : bool, optional
        If True, return the result of the wrapped function. Default is False.
    info_logger : logging.Logger, optional
        Logger to use for info-level messages (e.g., timing).
    error_logger : logging.Logger, optional
        Logger to use for errors (e.g., exception traces).
    warning_logger : logging.Logger, optional
        Logger to use for warnings.
    warning_message : str, optional
        Optional message to log via the warning logger.

    Returns
    -------
    Callable[[F], F]
        A decorator that wraps the target function with timing and logging behavior.
    """

    def decorator(func: F) -> F:
        label = name or func.__name__
        return timed_wrapper(
            func,
            label=label,
            log=log,
            log_errors=log_errors,
            raise_exception=raise_exception,
            log_inputs=False,
            log_output=False,
            return_result=return_result,
            info_logger=info_logger,
            error_logger=error_logger,
            debug_logger=None,
            warning_logger=warning_logger,
            warning_message=warning_message,
        )
    return decorator

# ====[ Safe timer with debug logging ]====
def safe_timer_and_debug(
    log: bool = True,
    log_errors: bool = True,
    raise_exception: bool = False,
    log_inputs: bool = True,
    log_output: bool = False,
    name: Optional[str] = None,
    return_result: bool = False,
    info_logger: Optional[logging.Logger] = None,
    error_logger: Optional[logging.Logger] = None,
    debug_logger: Optional[logging.Logger] = None,
    warning_logger: Optional[logging.Logger] = None,
    warning_message: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator for timing function execution with optional input/output debug logging.

    This is an extended version of `safe_timer` that additionally supports logging of
    function arguments and return values for debugging purposes.

    Parameters
    ----------
    log : bool, optional
        If True, log execution time via `info_logger`. Default is True.
    log_errors : bool, optional
        If True, log any exceptions via `error_logger`. Default is True.
    raise_exception : bool, optional
        If True, re-raise exceptions after logging. Otherwise, suppress them. Default is False.
    log_inputs : bool, optional
        If True, log the function arguments via `debug_logger`. Default is True.
    log_output : bool, optional
        If True, log the function return value via `debug_logger`. Default is False.
    name : str or None, optional
        Name to display in logs. If None, the function's own name is used.
    return_result : bool, optional
        If True, return the function result. If False, discard it. Default is False.
    info_logger : logging.Logger, optional
        Logger for informational messages such as timing.
    error_logger : logging.Logger, optional
        Logger for error messages if exceptions occur.
    debug_logger : logging.Logger, optional
        Logger for debug-level messages (inputs, outputs).
    warning_logger : logging.Logger, optional
        Logger for warnings (e.g., fallback notices).
    warning_message : str, optional
        Optional warning message to emit via `warning_logger`.

    Returns
    -------
    Callable[[F], F]
        A decorator that wraps the target function with timing, logging, and debugging.
    """

    def decorator(func: F) -> F:
        label = name or func.__name__
        return timed_wrapper(
            func,
            label=label,
            log=log,
            log_errors=log_errors,
            raise_exception=raise_exception,
            log_inputs=log_inputs,
            log_output=log_output,
            return_result=return_result,
            info_logger=info_logger,
            error_logger=error_logger,
            debug_logger=debug_logger,
            warning_logger=warning_logger,
            warning_message=warning_message,
        )
    return decorator

# ====[ Simple conditional warning logger ]====
def log_warning_if(condition: bool, message: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a warning message if a given condition is True.

    Parameters
    ----------
    condition : bool
        Boolean condition that triggers the warning if True.
    message : str
        Warning message to be logged.
    logger : logging.Logger, optional
        Logger to use. If None, uses the default global warning logger.

    Returns
    -------
    None
        Logs the message if condition is met; otherwise does nothing.
    """

    if condition:
        (logger or get_logger()).warning(message)

# ====[ Decorator version of conditional warning logger ]====
def log_warning_if_decorator(
    condition_fn: Callable[[], bool],
    message: str,
    logger: Optional[logging.Logger] = None,
) -> Callable[[F], F]:
    """
    Decorator that logs a warning before executing the function if a condition is met.

    The condition is evaluated by calling `condition_fn()` just before the decorated
    function runs. If it returns True, the specified warning message is logged.

    Parameters
    ----------
    condition_fn : Callable[[], bool]
        A zero-argument function returning a boolean. If True, the warning is triggered.
    message : str
        Warning message to log when the condition is met.
    logger : logging.Logger, optional
        Logger to use for warning messages. If None, a default logger is used.

    Returns
    -------
    Callable[[F], F]
        A decorator that wraps the target function with conditional warning logic.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if condition_fn():
                (logger or get_logger()).warning(message)
            return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator

# ====[ Utility: get all loggers ]====
def get_all_loggers() -> Dict[str, logging.Logger]:
    """
    Retrieve a dictionary containing all standard configured loggers.

    Includes the default info, error, and debug loggers used across the application.

    Returns
    -------
    Dict[str, logging.Logger]
        Dictionary with logger names as keys (e.g., "info", "error", "debug")
        and corresponding logger instances as values.
    """

    return {
        "info": get_logger(),
        "error": get_error_logger(),
        "debug": get_debug_logger(),
    }
