"""Filesystem helpers built exclusively on :mod:`pathlib`.

The reference packages mixed ``os.path`` string concatenation
(``self.dir_path + "resource/..."``), platform specific branches (``C:/Program
Files/...``) and hard-coded Windows separators.  This module provides the small
set of primitives the rest of the library needs, all of them ``pathlib`` based
and Linux-only.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Final, TypeAlias

__all__ = [
    "DATA_DIR",
    "PACKAGE_DIR",
    "PathLike",
    "ensure_directory",
    "ensure_file",
    "iter_files",
    "read_lines",
    "read_text",
    "resolve",
    "write_lines",
    "write_text",
]

PathLike: TypeAlias = str | Path

PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
"""Directory of the installed :mod:`farsi_textutils` package."""

DATA_DIR: Final[Path] = PACKAGE_DIR / "data"
"""Directory holding the static reference tables."""


def resolve(path: PathLike) -> Path:
    """Expand ``~`` and return an absolute :class:`Path`."""
    return Path(path).expanduser().resolve()


def ensure_directory(path: PathLike) -> Path:
    """Create *path* (and parents) if needed and return it as a :class:`Path`."""
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_file(path: PathLike) -> Path:
    """Validate that *path* is an existing regular file and return it."""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")
    return file_path


def iter_files(
    root: PathLike,
    *,
    suffixes: Iterable[str] = (".txt",),
    recursive: bool = True,
) -> Iterator[Path]:
    """Yield regular files below *root* whose suffix is in *suffixes*.

    Args:
        root: Directory to walk.
        suffixes: Accepted suffixes (case-insensitive, including the dot).
        recursive: Walk sub-directories when ``True``.

    Yields:
        Sorted :class:`Path` objects (deterministic order across platforms).
    """
    base = resolve(root)
    if not base.is_dir():
        raise NotADirectoryError(f"not a directory: {base}")
    wanted = {suffix.lower() for suffix in suffixes}
    globber = base.rglob("*") if recursive else base.glob("*")
    for candidate in sorted(globber):
        if candidate.is_file() and (not wanted or candidate.suffix.lower() in wanted):
            yield candidate


def read_text(path: PathLike, *, encoding: str = "utf-8") -> str:
    """Read a text file (``pathlib`` only, no ``open``/``os.path`` juggling)."""
    return ensure_file(path).read_text(encoding=encoding)


def read_lines(path: PathLike, *, encoding: str = "utf-8", skip_blank: bool = True) -> list[str]:
    """Read a text file as a list of stripped lines."""
    lines = [line.strip() for line in read_text(path, encoding=encoding).splitlines()]
    return [line for line in lines if line] if skip_blank else lines


def write_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> Path:
    """Write *text*, creating parent directories, and return the written path."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding=encoding)
    return target


def write_lines(path: PathLike, lines: Iterable[str], *, encoding: str = "utf-8") -> Path:
    r"""Write *lines* terminated by ``\\n`` and return the written path."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding=encoding, newline="\n") as handle:
        for line in lines:
            handle.write(line.rstrip("\n") + "\n")
    return target
