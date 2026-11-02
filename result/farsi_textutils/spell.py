"""Spelling correction with a pure-Python n-gram model.

Replaces two heavyweight pieces of the reference stacks:

* ``parsivar.spell_checker.SpellCheck`` -- required two ~100 MB pickles
  (``mybigram_lm.pckl``, ``onegram.pckl``) produced by an external script and
  used ``pickle.load`` on files of unknown provenance;
* the neural / ``sentencepiece`` based language models of the other reference
  projects, which needed torch and ONNX runtimes.

The replacement is :class:`NgramModel`: a dict-backed unigram/bigram counter that
trains directly from a corpus with ``multiprocessing`` workers, serialises to
JSON (never pickle) and scores candidates with interpolated back-off smoothing.

Example:
    >>> model = NgramModel.train(["کتاب من خوب است", "کتاب تو خوب است"])
    >>> checker = SpellChecker(model)
    >>> checker.correct("کتلب")
    'کتاب'
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Final, cast

from . import constants as C
from .data.orthography import ORTHOGRAPHY_MAP
from .exceptions import ModelNotTrainedError
from .logging_setup import get_logger
from .mapping import CharMapper
from .parallel import ParallelConfig, parallel_starmap
from .tokenize import word_tokenize

__all__ = [
    "NgramModel",
    "SpellChecker",
    "SpellSuggestion",
    "edit_candidates",
]

_log = get_logger("spell")

_ORTHOGRAPHY: Final[CharMapper] = CharMapper(ORTHOGRAPHY_MAP, name="spell-orthography")
_ALPHABET: Final[tuple[str, ...]] = (*tuple(C.PERSIAN_LETTERS), C.HALF_SPACE)
_SPLIT_MARKER: Final[str] = "-"
_MERGE_MARKER: Final[str] = "#"


def _token_count_chunk(
    texts: Sequence[str], order: int
) -> tuple[Counter[str], Counter[tuple[str, ...]], Counter[str]]:
    """Count unigrams/bigrams of a chunk of documents (worker entry point).

    Module level so that it can be pickled by :func:`~farsi_textutils.parallel.parallel_map`.
    """
    unigrams: Counter[str] = Counter()
    bigrams: Counter[tuple[str, ...]] = Counter()
    boundaries: Counter[str] = Counter()
    for text in texts:
        tokens = ["<s>", *word_tokenize(text, keep_punctuation=False), "</s>"]
        unigrams.update(tokens[1:-1])
        boundaries.update((tokens[0], tokens[-1]))
        if order >= 2:
            bigrams.update(pairwise(tokens))
    return unigrams, bigrams, boundaries


@dataclass(slots=True)
class NgramModel:
    """A smoothed unigram/bigram language model.

    Args:
        order: ``1`` for a unigram model, ``2`` (default) for bigrams.
        smoothing: Additive smoothing constant used for unseen words.

    The model is trained with :meth:`train`, saved/loaded as JSON with
    :meth:`save`/:meth:`load`.
    """

    order: int = 2
    smoothing: float = 0.5
    unigrams: Counter[str] = field(default_factory=Counter)
    bigrams: Counter[tuple[str, ...]] = field(default_factory=Counter)
    total_tokens: int = 0
    _vocabulary: set[str] = field(default_factory=set, repr=False)

    # --- construction -----------------------------------------------------
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        order: int = 2,
        workers: int = 4,
        chunksize: int = 64,
    ) -> NgramModel:
        """Train a model from *texts* using four worker processes.

        Args:
            texts: Documents (strings).
            order: Model order (1 or 2).
            workers: Worker processes passed to ``Pool.imap_unordered``.
            chunksize: Documents per work item.

        Returns:
            The fitted :class:`NgramModel`.

        >>> model = NgramModel.train(["یک دو سه"], workers=2)
        >>> model.is_known("دو")
        True
        """
        if order < 1:
            raise ValueError("order must be >= 1")
        documents = list(texts)
        if not documents:
            return cls(order=order)
        model = cls(order=order)
        chunks = [documents[index : index + chunksize] for index in range(0, len(documents), chunksize)]
        for unigrams, bigrams, boundaries in parallel_starmap(
            _token_count_chunk,
            [(chunk, order) for chunk in chunks],
            config=ParallelConfig(workers=workers),
        ):
            model.unigrams.update(unigrams)
            model.bigrams.update(bigrams)
            model.total_tokens += sum(unigrams.values()) + 2 * len(boundaries)
        model._vocabulary = set(model.unigrams)
        return model

    # --- probabilities ----------------------------------------------------
    @property
    def vocabulary_size(self) -> int:
        """Number of distinct words of the model."""
        return len(self._vocabulary or self.unigrams)

    def is_known(self, word: str) -> bool:
        """True when *word* was seen during training.

        >>> NgramModel.train(["سلام دنیا"], workers=1).is_known("سلام")
        True
        """
        return word in (self._vocabulary or self.unigrams)

    def word_logprob(self, word: str) -> float:
        """Smoothed log10-probability of a single word.

        Unknown words receive the smoothing floor, which is very negative but
        finite, so candidates stay comparable.
        """
        if not self.total_tokens:
            raise ModelNotTrainedError("the language model has not been trained")
        count = self.unigrams.get(word, 0)
        denominator = self.total_tokens + self.smoothing * self.vocabulary_size
        return math.log10((count + self.smoothing) / denominator)

    def bigram_logprob(self, previous: str, word: str) -> float:
        """Backed-off log10-probability of ``P(word | previous)``."""
        previous_count = self.unigrams.get(previous, 0)
        if previous_count <= 0:
            return self.word_logprob(word)
        pair = self.bigrams.get((previous, word), 0)
        denominator = previous_count + self.smoothing * self.vocabulary_size
        interpolated = 0.8 * (pair + self.smoothing) / denominator
        return math.log10(interpolated + 0.2 * 10 ** self.word_logprob(word))

    def sentence_logprob(self, tokens: Sequence[str]) -> float:
        """Total log10-probability of *tokens*.

        >>> model = NgramModel.train(["کتاب من خوب است"], workers=1)
        >>> round(model.sentence_logprob(["کتاب"]), 2) < 0
        True
        """
        padded = ["<s>", *tokens]
        return sum(self.bigram_logprob(previous, token) for previous, token in pairwise(padded))

    # --- persistence ------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        """Serialisable representation (JSON friendly)."""
        return {
            "order": self.order,
            "smoothing": self.smoothing,
            "total_tokens": self.total_tokens,
            "unigrams": dict(self.unigrams),
            "bigrams": [["|".join(pair), count] for pair, count in self.bigrams.items()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> NgramModel:
        """Rebuild a model from :meth:`to_dict` output."""
        raw_unigrams = cast("dict[str, object]", payload["unigrams"])
        unigrams = Counter({str(word): int(cast(int, count)) for word, count in raw_unigrams.items()})
        bigrams: Counter[tuple[str, ...]] = Counter()
        for pair, count in cast("list[list[object]]", payload["bigrams"]):
            bigrams[tuple(str(pair).split("|", 1))] = int(cast(int, count))
        model = cls(
            order=int(cast(int, payload["order"])),
            smoothing=float(cast(float, payload["smoothing"])),
            unigrams=unigrams,
            bigrams=bigrams,
            total_tokens=int(cast(int, payload["total_tokens"])),
        )
        model._vocabulary = set(unigrams)
        return model

    def save(self, path: Path | str) -> Path:
        """Write the model as UTF-8 JSON (``pathlib`` based, no pickle)."""
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        _log.info("language model saved to {} ({} words)", target, self.vocabulary_size)
        return target

    @classmethod
    def load(cls, path: Path | str) -> NgramModel:
        """Read a model previously written by :meth:`save`."""
        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"language model not found: {source}")
        model = cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
        _log.info("language model loaded from {} ({} words)", source, model.vocabulary_size)
        return model

    # --- corpus helpers ---------------------------------------------------
    @classmethod
    def train_from_directory(
        cls,
        directory: Path | str,
        *,
        pattern: str = "*.txt",
        workers: int = 4,
    ) -> NgramModel:
        """Train from every text file below *directory* (pathlib traversal)."""
        from .paths import iter_files, read_text

        suffix = pattern.replace("*", "")
        texts = (read_text(path) for path in iter_files(directory, suffixes=(suffix,)))
        return cls.train(texts, workers=workers)


@dataclass(frozen=True, slots=True)
class SpellSuggestion:
    """A candidate correction.

    Attributes:
        word: The suggested spelling (may contain ``-`` for a split or the text
            of a merge for a joined word).
        score: Log10-probability under the language model.
        operation: One of ``keep``, ``deletion``, ``insertion``, ``substitution``,
            ``transposition``, ``split`` or ``merge``.
    """

    word: str
    score: float
    operation: str


def edit_candidates(
    word: str,
    *,
    alphabet: Iterable[str] = _ALPHABET,
    include_split: bool = True,
) -> Iterator[tuple[str, str]]:
    """Yield ``(candidate, operation)`` pairs one edit away from *word*.

    The operation names match ``parsivar.spell_checker.SpellCheck``.

    >>> any(candidate == "کتاب" for candidate, _ in edit_candidates("کتب"))
    True
    """
    letters = tuple(alphabet)
    for index in range(len(word)):
        yield word[:index] + word[index + 1 :], "deletion"
    for index in range(len(word) + 1):
        for letter in letters:
            yield word[:index] + letter + word[index:], "insertion"
    for index in range(len(word)):
        for letter in letters:
            if letter != word[index]:
                yield word[:index] + letter + word[index + 1 :], "substitution"
    for index in range(len(word) - 1):
        swapped = word[:index] + word[index + 1] + word[index] + word[index + 2 :]
        yield swapped, "transposition"
    if include_split:
        for index in range(2, len(word) - 1):
            yield f"{word[:index]}{_SPLIT_MARKER}{word[index:]}", "split"


class SpellChecker:
    """Spell corrector driven by an :class:`NgramModel`.

    Args:
        model: Trained language model; :meth:`NgramModel.train` builds one.
        max_suggestions: Number of suggestions returned per token.
        use_context: Rescore candidates with the bigram context of the previous
            word (a cheap stand-in for the reference trigram search).

    Example:
        >>> model = NgramModel.train(["کتاب من خوب است", "کتاب تو خوب است"], workers=1)
        >>> SpellChecker(model).is_correct("کتاب")
        True
        >>> SpellChecker(model).correct("کتلب")
        'کتاب'
    """

    __slots__ = ("max_suggestions", "model", "use_context")

    def __init__(
        self,
        model: NgramModel,
        *,
        max_suggestions: int = 3,
        use_context: bool = True,
    ) -> None:
        self.model = model
        self.max_suggestions = max(1, max_suggestions)
        self.use_context = use_context

    # --- single words -----------------------------------------------------
    def normalise(self, word: str) -> str:
        """Apply orthographic unification before lookup."""
        return _ORTHOGRAPHY.map_string(word)

    def is_correct(self, word: str) -> bool:
        """True when *word* (after normalisation) is in the vocabulary."""
        return self.model.is_known(self.normalise(word))

    def suggestions(
        self,
        word: str,
        *,
        previous: str | None = None,
        limit: int | None = None,
    ) -> list[SpellSuggestion]:
        """Return the most probable corrections of *word*.

        >>> model = NgramModel.train(["کتاب خوب است"], workers=1)
        >>> SpellChecker(model).suggestions("کتب")[0].word
        'کتاب'
        """
        limit = limit or self.max_suggestions
        normalized = self.normalise(word)
        results: dict[str, SpellSuggestion] = {}

        def register(candidate: str, operation: str) -> None:
            if _SPLIT_MARKER in candidate:
                head, _, tail = candidate.partition(_SPLIT_MARKER)
                if not (self.model.is_known(head) and self.model.is_known(tail)):
                    return
                score = (self.model.word_logprob(head) + self.model.word_logprob(tail)) / 2
            else:
                if not self.model.is_known(candidate):
                    return
                score = self.model.word_logprob(candidate)
                if self.use_context and previous:
                    score = (score + self.model.bigram_logprob(previous, candidate)) / 2
            if candidate not in results or results[candidate].score < score:
                results[candidate] = SpellSuggestion(candidate, score, operation)

        if self.model.is_known(normalized):
            context = previous or "<s>"
            results[normalized] = SpellSuggestion(
                normalized, self.model.bigram_logprob(context, normalized), "keep"
            )
        for candidate, operation in edit_candidates(normalized):
            register(candidate, operation)
        ranked = sorted(results.values(), key=lambda suggestion: suggestion.score, reverse=True)
        return ranked[:limit]

    def correct(self, word: str, *, previous: str | None = None) -> str:
        """Return the best spelling of *word* (unchanged when unknown)."""
        suggestions = self.suggestions(word, previous=previous, limit=1)
        return suggestions[0].word if suggestions else word

    # --- whole texts ------------------------------------------------------
    def correct_text(self, text: str) -> str:
        """Correct *text*, preserving the original whitespace and punctuation.

        Merged words (``کتا`` + ``ب``) and split words (``کتا ب``) are repaired
        using the bigram model, mirroring the ``Merg``/``Split`` operations of
        ``parsivar.spell_checker``.

        >>> model = NgramModel.train(["سلام دنیا خوب است"], workers=1)
        >>> SpellChecker(model).correct_text("سلم دنیا")
        'سلام دنیا'
        """
        pieces: list[str] = []
        for chunk in re.split(r"(\s+)", text):
            if not chunk:
                continue
            if chunk.isspace():
                pieces.append(chunk)
                continue
            pieces.append(self._correct_chunk(chunk))
        return "".join(pieces)

    def _correct_chunk(self, chunk: str) -> str:
        """Correct one whitespace-free chunk, leaving punctuation untouched."""
        tokens = word_tokenize(chunk)
        corrected: list[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if not any(char.isalpha() for char in token):
                corrected.append(token)
                index += 1
                continue
            previous = next(
                (candidate for candidate in reversed(corrected) if any(char.isalpha() for char in candidate)),
                None,
            )
            if self.is_correct(token):
                corrected.append(token)
                index += 1
                continue
            merged = self._try_merge(tokens, index, previous)
            if merged is not None:
                corrected.append(merged[0])
                index += merged[1]
                continue
            corrected.append(self.correct(token, previous=previous))
            index += 1
        return "".join(corrected)

    def _try_merge(
        self,
        tokens: Sequence[str],
        index: int,
        previous: str | None,
    ) -> tuple[str, int] | None:
        """Try joining *tokens[index]* with its right neighbour.

        The reference implementation emitted ``word#next`` candidates; here the
        merged form is validated directly.
        """
        if index + 1 >= len(tokens):
            return None
        joined = f"{tokens[index]}{tokens[index + 1]}"
        normalized = self.normalise(joined)
        if not self.model.is_known(normalized):
            return None
        context = previous or "<s>"
        if self.model.bigram_logprob(context, normalized) <= self.model.word_logprob(normalized) - 1.0:
            return None
        return normalized, 2


def train_from_texts(
    texts: Iterable[str],
    *,
    workers: int = 4,
    model_path: Path | str | None = None,
) -> SpellChecker:
    """Train a :class:`SpellChecker` from *texts*, optionally saving the model.

    >>> checker = train_from_texts(["سلام دنیا"], workers=1)
    >>> checker.is_correct("سلام")
    True
    """
    model = NgramModel.train(texts, workers=workers)
    if model_path is not None:
        model.save(model_path)
    return SpellChecker(model)
