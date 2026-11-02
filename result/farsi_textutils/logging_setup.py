"""Logging bootstrap built on :mod:`loguru`.

The reference packages used the standard library ``logging`` module (or bare
``print`` calls) inconsistently.  This module is the single place where the
library talks to loguru:

* loguru's own default ``DEBUG`` sink is removed on import, so importing the
  library never spams ``stderr``;
* :func:`configure_logging` installs the pretty sink (and an optional rotating
  file sink) whenever a caller -- or the CLI -- wants log output;
* setting ``FARSI_TEXTUTILS_LOG_LEVEL`` in the environment configures logging at
  import time, which is handy for ``multiprocessing`` workers.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Literal, TypeVar

from loguru import logger as _logger

__all__ = ["LogLevel", "configure_logging", "get_logger", "logged", "logger"]

LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

DEFAULT_FORMAT: Final[str] = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
    "<level>{level: <8}</level> "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger = _logger
"""The shared loguru logger.  Use ``from farsi_textutils.logging_setup import logger``."""

ENV_LOG_LEVEL: Final[str] = "FARSI_TEXTUTILS_LOG_LEVEL"
"""Environment variable read on import to configure the sink."""

# loguru installs a DEBUG sink on ``stderr`` when it is imported; the library
# removes it so that importing :mod:`farsi_textutils` is silent.
logger.remove()

_F = TypeVar("_F", bound=Callable[..., Any])


def get_logger(name: str | None = None) -> Any:
    """Return a loguru logger bound to *name* (added as contextual ``extra``).

    >>> get_logger("cli").info("hello")  # doctest: +SKIP
    """
    return logger.bind(component=name) if name else logger


def configure_logging(
    level: LogLevel | int = "INFO",
    *,
    logfile: Path | None = None,
    colorize: bool | None = None,
    enqueue: bool = False,
    backtrace: bool = False,
    diagnose: bool = False,
) -> None:
    """Install the library's default sink(s).

    Args:
        level: Minimum severity to emit.
        logfile: Optional path of a rotating file sink.  Parent directories are
            created if needed (``pathlib`` based, no ``os.makedirs``).
        colorize: Force/disable ANSI colours; ``None`` lets loguru decide.
        enqueue: Make the sinks process-safe (useful inside ``multiprocessing``).
        backtrace: Loguru backtrace option, off by default for readable output.
        diagnose: Loguru diagnose option, off by default (never leak values).
    """
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=level,
        format=DEFAULT_FORMAT,
        colorize=colorize if colorize is not None else None,
        enqueue=enqueue,
        backtrace=backtrace,
        diagnose=diagnose,
    )
    if logfile is not None:
        path = Path(logfile).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            path,
            level=level,
            format=DEFAULT_FORMAT,
            rotation="10 MB",
            retention=5,
            compression="gz",
            encoding="utf-8",
            enqueue=enqueue,
            backtrace=backtrace,
            diagnose=diagnose,
        )


def logged(message: str) -> Callable[[_F], _F]:
    """Decorator logging entry and exit of a function.

    >>> @logged("stemming")
    ... def stem(word: str) -> str:  # doctest: +SKIP
    ...     return word
    """

    def decorator(function: _F) -> _F:
        import functools

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.debug("{}: {}() called", message, function.__qualname__)
            try:
                return function(*args, **kwargs)
            except Exception:  # pragma: no cover - logging path
                logger.exception("{}: {}() failed", message, function.__qualname__)
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


_initial_level = os.environ.get(ENV_LOG_LEVEL)
if _initial_level:
    configure_logging(_initial_level)  # type: ignore[arg-type]
