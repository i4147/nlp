"""Parallel helpers: ``multiprocessing.Pool.imap_unordered`` with four workers.

Every long running operation of the library (corpus normalisation, language
model counting, spell-checking a word list, extracting words with branching
entropy) goes through this module so the process count and the chunking policy
are configured in exactly one place.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeVar

from .logging_setup import get_logger

__all__ = [
    "DEFAULT_WORKERS",
    "ParallelConfig",
    "parallel_map",
    "parallel_starmap",
    "worker_init_logging",
]

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_WORKERS: Final[int] = 4
"""Number of worker processes used by the library (Linux: ``fork`` start method)."""

_log = get_logger("parallel")


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    """Tunables for a single parallel run."""

    workers: int = DEFAULT_WORKERS
    chunksize: int = 1
    maxtasksperchild: int | None = None

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.chunksize < 1:
            raise ValueError("chunksize must be >= 1")

    @property
    def effective_workers(self) -> int:
        """Never ask for more processes than the machine can offer."""
        return max(1, min(self.workers, mp.cpu_count() or 1))

    def merged(self, *, workers: int | None = None, chunksize: int | None = None) -> ParallelConfig:
        """Return a copy with the explicitly provided overrides applied."""
        if workers is None and chunksize is None:
            return self
        return ParallelConfig(
            workers=workers if workers is not None else self.workers,
            chunksize=chunksize if chunksize is not None else self.chunksize,
            maxtasksperchild=self.maxtasksperchild,
        )


def _context() -> mp.context.ForkContext:
    """Return the ``fork`` context: the library targets Linux only."""
    return mp.get_context("fork")


def _apply(payload: tuple[Callable[..., R], tuple[Any, ...]]) -> R:
    """Module level trampoline so ``imap_unordered`` can pickle the work item."""
    function, arguments = payload
    return function(*arguments)


def worker_init_logging(level: str = "WARNING") -> None:
    """Initialiser making loguru usable inside pool workers.

    ``fork`` copies the parent's sinks into every child, whose writes can then
    interleave.  Re-configuring with ``enqueue=True`` gives each worker its own
    process-safe sink.
    """
    from .logging_setup import configure_logging

    configure_logging(level, enqueue=True)  # type: ignore[arg-type]


def parallel_map(
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    config: ParallelConfig | None = None,
    workers: int | None = None,
    chunksize: int | None = None,
    initializer: Callable[..., Any] | None = None,
    initargs: Sequence[Any] = (),
) -> Iterator[R]:
    """Yield ``function(item)`` for every item using ``Pool.imap_unordered``.

    Results are produced as soon as *any* worker finishes, which keeps memory
    flat for large corpora and avoids the head-of-line blocking of ``map``.

    Args:
        function: Picklable callable executed in the workers.
        items: Inputs of the callable.
        config: Optional :class:`ParallelConfig`; explicit ``workers`` and
            ``chunksize`` arguments win over it.
        workers: Number of worker processes (defaults to 4).
        chunksize: Number of items handed to a worker at once.
        initializer: Optional pool initialiser (e.g. :func:`worker_init_logging`).
        initargs: Arguments for *initializer*.

    Yields:
        Results in completion order.
    """
    cfg = (config or ParallelConfig()).merged(workers=workers, chunksize=chunksize)
    worker_count = cfg.effective_workers
    _log.debug("parallel_map: {} workers, chunksize {}", worker_count, cfg.chunksize)
    with _context().Pool(
        processes=worker_count,
        initializer=initializer,
        initargs=tuple(initargs),
        maxtasksperchild=cfg.maxtasksperchild,
    ) as pool:
        yield from pool.imap_unordered(function, items, chunksize=cfg.chunksize)


def parallel_starmap(
    function: Callable[..., R],
    argument_tuples: Iterable[tuple[Any, ...]],
    *,
    config: ParallelConfig | None = None,
    workers: int | None = None,
    chunksize: int | None = None,
    initializer: Callable[..., Any] | None = None,
    initargs: Sequence[Any] = (),
) -> Iterator[R]:
    """Like :func:`parallel_map`, but for callables taking several arguments.

    >>> sorted(parallel_starmap(pow, [(2, 3), (3, 2)], workers=2))
    [8, 9]
    """
    cfg = (config or ParallelConfig()).merged(workers=workers, chunksize=chunksize)
    payloads = ((function, arguments) for arguments in argument_tuples)
    return parallel_map(
        _apply,
        payloads,
        config=cfg,
        initializer=initializer,
        initargs=initargs,
    )
