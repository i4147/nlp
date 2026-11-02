"""Batch corpus processing on a directory tree.

The reference packages shipped one-off scripts for this job (``inltk``'s
``download_assets.py``, ``ginza``'s training commands, ``parsivar``'s
``build_stem_dictionary``) that walked directories with ``os.listdir`` and
processed files sequentially.  This module is the supported replacement:

* :meth:`CorpusPipeline.normalise_directory` -- normalise every file of a tree;
* :meth:`CorpusPipeline.tokenise_directory` -- write one tokenised file per input;
* :meth:`CorpusPipeline.word_counts` and :meth:`CorpusPipeline.build_language_model`
  -- corpus statistics for :mod:`farsi_textutils.spell`.

Every bulk step uses ``multiprocessing.Pool.imap_unordered`` with four workers and
``pathlib`` for the filesystem.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .logging_setup import get_logger
from .normalize import Normalizer, NormalizerConfig
from .parallel import DEFAULT_WORKERS, ParallelConfig, parallel_map
from .paths import ensure_directory, iter_files, read_text, write_text
from .spell import NgramModel
from .tokenize import Tokenizer

__all__ = ["CorpusPipeline", "count_words_in_file", "normalise_file", "tokenise_file"]

_log = get_logger("corpus")


@dataclass(frozen=True, slots=True)
class _NormaliseTask:
    source: Path
    target: Path
    config: NormalizerConfig


@dataclass(frozen=True, slots=True)
class _TokeniseTask:
    source: Path
    target: Path
    split_half_space: bool
    sentence_per_line: bool


def normalise_file(task: _NormaliseTask) -> tuple[Path, int]:
    """Normalise one file and write the result next to its target.

    Returns:
        ``(target, number_of_lines)``.
    """
    normalizer = Normalizer(task.config)
    lines = [normalizer.normalize(line) for line in read_text(task.source).splitlines()]
    write_text(task.target, "\n".join(lines))
    return task.target, len(lines)


def tokenise_file(task: _TokeniseTask) -> tuple[Path, int]:
    """Tokenise one file: one token per line, or one sentence per line."""
    tokenizer = Tokenizer()
    output: list[str] = []
    for line in read_text(task.source).splitlines():
        if task.sentence_per_line:
            output.extend(tokenizer.sentence(line))
        else:
            output.extend(tokenizer.word(line))
    write_text(task.target, "\n".join(output))
    return task.target, len(output)


def count_words_in_file(path: Path) -> Counter[str]:
    """Count the words of a single file (worker entry point)."""
    return Counter(Tokenizer().word(read_text(path)))


@dataclass(slots=True)
class CorpusPipeline:
    """Run the library over a directory of text files.

    Args:
        normalizer_config: Options forwarded to :class:`~farsi_textutils.normalize.Normalizer`.
        workers: Worker processes for the bulk steps.

    Example:
        >>> pipeline = CorpusPipeline(workers=2)
        >>> pipeline.workers
        2
    """

    normalizer_config: NormalizerConfig = field(default_factory=NormalizerConfig)
    workers: int = DEFAULT_WORKERS
    suffixes: tuple[str, ...] = (".txt",)

    # --- traversal --------------------------------------------------------
    def files(self, directory: Path | str) -> list[Path]:
        """List the input files of *directory* (sorted, deterministic)."""
        return list(iter_files(directory, suffixes=self.suffixes))

    # --- bulk steps -------------------------------------------------------
    def normalise_directory(
        self,
        source: Path | str,
        target: Path | str,
        *,
        overwrite: bool = True,
    ) -> list[Path]:
        """Normalise every file below *source* into *target*.

        Returns:
            The written files, in completion order.
        """
        destination = ensure_directory(target)
        tasks: list[_NormaliseTask] = []
        for path in self.files(source):
            output = destination / path.name
            if output.exists() and not overwrite:
                continue
            tasks.append(_NormaliseTask(path, output, self.normalizer_config))
        written: list[Path] = []
        results = parallel_map(normalise_file, tasks, config=ParallelConfig(workers=self.workers))
        for written_path, lines in results:
            _log.debug("normalised {} lines -> {}", lines, written_path)
            written.append(written_path)
        _log.info("normalised {} file(s) from {} to {}", len(written), source, destination)
        return written

    def tokenise_directory(
        self,
        source: Path | str,
        target: Path | str,
        *,
        split_half_space: bool = False,
        sentence_per_line: bool = False,
    ) -> list[Path]:
        """Tokenise every file below *source* into *target*."""
        destination = ensure_directory(target)
        tasks = [
            _TokeniseTask(path, destination / path.name, split_half_space, sentence_per_line)
            for path in self.files(source)
        ]
        written: list[Path] = []
        for written_path, tokens in parallel_map(
            tokenise_file, tasks, config=ParallelConfig(workers=self.workers)
        ):
            _log.debug("tokenised {} tokens -> {}", tokens, written_path)
            written.append(written_path)
        return written

    def word_counts(self, source: Path | str) -> Counter[str]:
        """Count every token of every file below *source*."""
        counts: Counter[str] = Counter()
        for partial in parallel_map(
            count_words_in_file, self.files(source), config=ParallelConfig(workers=self.workers)
        ):
            counts.update(partial)
        return counts

    def build_language_model(
        self,
        source: Path | str,
        *,
        output: Path | str | None = None,
        order: int = 2,
    ) -> NgramModel:
        """Train (and optionally persist) an n-gram model from *source*."""
        documents = (read_text(path) for path in self.files(source))
        model = NgramModel.train(documents, order=order, workers=self.workers)
        if output is not None:
            model.save(output)
        return model


def iter_documents(directory: Path | str, *, suffixes: Sequence[str] = (".txt",)) -> Iterator[str]:
    """Yield the text of every matching file below *directory*.

    >>> import pathlib, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = pathlib.Path(tmp) / "a.txt"
    ...     _ = path.write_text("سلام", encoding="utf-8")
    ...     list(iter_documents(tmp))
    ['سلام']
    """
    for path in iter_files(directory, suffixes=tuple(suffixes)):
        yield read_text(path)


def parallel_line_map(
    function: Callable[[str], object],
    directory: Path | str,
    *,
    suffixes: Iterable[str] = (".txt",),
    workers: int = DEFAULT_WORKERS,
) -> Iterator[object]:
    """Apply *function* to every line of every file below *directory*.

    Lines are streamed through ``Pool.imap_unordered`` so the corpus never has to
    fit in memory.

    >>> sorted(parallel_line_map(str.strip, "does-not-exist", workers=1))  # doctest: +SKIP
    []
    """
    lines = (
        line
        for path in iter_files(directory, suffixes=tuple(suffixes))
        for line in read_text(path).splitlines()
    )
    yield from parallel_map(function, lines, config=ParallelConfig(workers=workers))
