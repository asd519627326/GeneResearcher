"""Small logging helpers shared by the GeneResearcher runtime."""

from __future__ import annotations

import inspect
import logging
import os
import sys
from functools import wraps
from time import perf_counter
from typing import Any, Callable, TypeVar, cast

_Callable = TypeVar("_Callable", bound=Callable[..., Any])


def progress_enabled(logger: logging.Logger) -> bool:
    """Show progress bars for INFO-level runs in terminals and notebook kernels."""
    if not logger.isEnabledFor(logging.INFO):
        return False
    if sys.stderr.isatty():
        return True
    return (
        "ipykernel" in sys.modules
        or bool(os.environ.get("JPY_PARENT_PID"))
        or bool(os.environ.get("COLAB_RELEASE_TAG"))
    )


def log_operation(operation: str, *, log_failure: bool = False) -> Callable[[_Callable], _Callable]:
    """Log operation start and duration without changing its result or errors."""

    def decorate(function: _Callable) -> _Callable:
        operation_logger = logging.getLogger(function.__module__)

        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                started_at = perf_counter()
                operation_logger.info("%s started.", operation)
                try:
                    result = await function(*args, **kwargs)
                except Exception:
                    if log_failure:
                        operation_logger.exception(
                            "%s failed after %.1f seconds.",
                            operation,
                            perf_counter() - started_at,
                        )
                    raise
                operation_logger.info(
                    "%s completed in %.1f seconds.",
                    operation,
                    perf_counter() - started_at,
                )
                return result

            return cast(_Callable, async_wrapper)

        @wraps(function)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            started_at = perf_counter()
            operation_logger.info("%s started.", operation)
            try:
                result = function(*args, **kwargs)
            except Exception:
                if log_failure:
                    operation_logger.exception(
                        "%s failed after %.1f seconds.",
                        operation,
                        perf_counter() - started_at,
                    )
                raise
            operation_logger.info(
                "%s completed in %.1f seconds.",
                operation,
                perf_counter() - started_at,
            )
            return result

        return cast(_Callable, sync_wrapper)

    return decorate
