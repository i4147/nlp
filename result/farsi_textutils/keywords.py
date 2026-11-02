"""Unsupervised keyword and word discovery.

Two techniques, both ported from the reference projects and both dependency free:

* :class:`WordExtractor` -- an adaptation of ``soynlp.word.WordExtractor``
  (branching entropy + cohesion).  The Korean original works on *character*
  sub-strings because Korean glues particles to stems; Persian glues particles
  with the ZWNJ instead, so this version measures entropy over *token*
  neighbourhoods and additionally treats the ZWNJ separated parts of a token as
  candidates.  It therefore discovers words and particles of a raw corpus
  without any lexicon;
* :func:`extract_keywords` -- classic TF-IDF over documents, the technique
  ``snownlp.summary`` used for key-sentence extraction.

Corpus level counting runs on ``multiprocessing.Pool`` with four workers.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from . import constants as C
from .exceptions import ModelNotTrainedError
from .logging_setup import get_logger
from .parallel import ParallelConfig, parallel_starmap
from .stopwords import content_words
from .tokenize import word_tokenize

__all__ = [
    "KeywordScore",
    "TfidfKeywords",
    "WordExtractor",
    "WordScore",
    "discover_words",
    "extract_keywords",
]

_log = get_logger("keywords")


def _entropy(counter: Counter[str]) -> float:
    """Shannon entropy (nats) of a counter.

    >>> round(_entropy(Counter({"a": 2})), 3)
    0.0
    >>> round(_entropy(Counter({"a": 1, "b": 1})), 3)
    0.693
    """
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy


@dataclass(frozen=True, slots=True)
class WordScore:
    """Statistics of a discovered word.

    Attributes:
        word: The candidate.
        frequency: How often the candidate occurs.
        cohesion_forward: ``max P(next token)`` -- high means the candidate is
            always followed by the same token, i.e. a fragment.
        cohesion_backward: ``max P(previous token)``.
        left_branching_entropy: Diversity of the left contexts.
        right_branching_entropy: Diversity of the right contexts.
        score: Ranking score: ``frequency * mean(entropies)``.
    """

    word: str
    frequency: int
    cohesion_forward: float
    cohesion_backward: float
    left_branching_entropy: float
    right_branching_entropy: float
    score: float


def _count_chunk(
    texts: Sequence[str],
) -> tuple[Counter[str], dict[str, Counter[str]], dict[str, Counter[str]]]:
    """Count tokens and their neighbour distributions for a chunk of documents."""
    frequencies: Counter[str] = Counter()
    left: dict[str, Counter[str]] = {}
    right: dict[str, Counter[str]] = {}
    for text in texts:
        for sentence in text.splitlines() or [text]:
            tokens = [token.strip(C.HALF_SPACE) for token in word_tokenize(sentence, keep_punctuation=False)]
            tokens = [token for token in tokens if token]
            for index, token in enumerate(tokens):
                parts = [token] if C.HALF_SPACE not in token else [token, *token.split(C.HALF_SPACE)]
                for part in parts:
                    if not part:
                        continue
                    frequencies[part] += 1
                    # sentence boundaries are not contexts: they would make every
                    # token look like a fragment with 100% cohesion
                    if index > 0:
                        left.setdefault(part, Counter())[tokens[index - 1]] += 1
                    else:
                        left.setdefault(part, Counter())
                    if index + 1 < len(tokens):
                        right.setdefault(part, Counter())[tokens[index + 1]] += 1
                    else:
                        right.setdefault(part, Counter())
    return frequencies, left, right


@dataclass(slots=True)
class WordExtractor:
    """Discover words of an unsegmented Persian corpus.

    Args:
        min_frequency: Minimum number of occurrences of a candidate.
        min_entropy: Minimum mean branching entropy; low values keep fragments.
        max_cohesion: Reject candidates that are always followed (or preceded)
            by the very same token, i.e. partial words.
        min_length: Minimum candidate length in characters.
        workers: Worker processes used to scan the corpus.

    Example:
        >>> extractor = WordExtractor(min_frequency=2, workers=1)
        >>> extractor.fit([
        ...     "کتابخانه بزرگ است", "کتابخانه کوچک است", "کتابخانه نو است",
        ... ])
        WordExtractor(min_frequency=2, min_entropy=0.0, max_cohesion=0.99, min_length=2, workers=1)
        >>> words = [item.word for item in extractor.extract()]
        >>> "کتابخانه" in words
        True
    """

    min_frequency: int = 5
    min_entropy: float = 0.0
    max_cohesion: float = 0.99
    min_length: int = 2
    workers: int = 4
    frequencies: Counter[str] = field(default_factory=Counter, repr=False)
    _left: dict[str, Counter[str]] = field(default_factory=dict, repr=False)
    _right: dict[str, Counter[str]] = field(default_factory=dict, repr=False)
    _fitted: bool = field(default=False, repr=False)

    def fit(self, documents: Iterable[str]) -> WordExtractor:
        """Scan *documents* with ``Pool.imap_unordered`` and four workers."""
        texts = list(documents)
        chunks = [texts[index : index + 128] for index in range(0, len(texts), 128)] or [[""]]
        frequencies: Counter[str] = Counter()
        left: dict[str, Counter[str]] = {}
        right: dict[str, Counter[str]] = {}
        for chunk_frequencies, chunk_left, chunk_right in parallel_starmap(
            _count_chunk,
            [(chunk,) for chunk in chunks],
            config=ParallelConfig(workers=self.workers),
        ):
            frequencies.update(chunk_frequencies)
            for word, contexts in chunk_left.items():
                left.setdefault(word, Counter()).update(contexts)
            for word, contexts in chunk_right.items():
                right.setdefault(word, Counter()).update(contexts)
        self.frequencies = frequencies
        self._left = left
        self._right = right
        self._fitted = True
        _log.debug("word extractor fitted on {} distinct candidates", len(frequencies))
        return self

    def score(self, word: str) -> WordScore | None:
        """Compute the statistics of a single candidate.

        >>> extractor = WordExtractor(workers=1).fit(["کتاب خوب", "دفتر خوب"])
        >>> round(extractor.score("کتاب").cohesion_forward, 3)
        1.0
        """
        left = self._left.get(word, Counter())
        right = self._right.get(word, Counter())
        frequency = self.frequencies.get(word, 0)
        if not frequency:
            return None
        right_total = sum(right.values())
        left_total = sum(left.values())
        cohesion_forward = max(right.values()) / right_total if right_total else 0.0
        cohesion_backward = max(left.values()) / left_total if left_total else 0.0
        left_entropy = _entropy(left)
        right_entropy = _entropy(right)
        return WordScore(
            word=word,
            frequency=frequency,
            cohesion_forward=cohesion_forward,
            cohesion_backward=cohesion_backward,
            left_branching_entropy=left_entropy,
            right_branching_entropy=right_entropy,
            score=frequency * (left_entropy + right_entropy) / 2,
        )

    def extract(self) -> list[WordScore]:
        """Return every candidate passing the configured thresholds.

        Raises:
            ModelNotTrainedError: When :meth:`fit` has not been called yet.
        """
        if not self._fitted:
            raise ModelNotTrainedError("call WordExtractor.fit() before extract()")
        results: list[WordScore] = []
        for word in self.frequencies:
            if len(word) < self.min_length:
                continue
            stats = self.score(word)
            if stats is None:
                continue
            if stats.frequency < self.min_frequency:
                continue
            if min(stats.left_branching_entropy, stats.right_branching_entropy) < self.min_entropy:
                continue
            if max(stats.cohesion_forward, stats.cohesion_backward) > self.max_cohesion:
                continue
            results.append(stats)
        return sorted(results, key=lambda item: (item.score, item.frequency), reverse=True)

    def extract_words(self, top_k: int | None = None) -> list[str]:
        """Return only the candidate strings."""
        words = [stats.word for stats in self.extract()]
        return words[:top_k] if top_k else words


@dataclass(frozen=True, slots=True)
class KeywordScore:
    """A TF-IDF keyword.

    >>> KeywordScore("کتاب", 2, 3, 1.5).frequency
    2
    """

    word: str
    frequency: int
    document_frequency: int
    score: float


@dataclass(slots=True)
class TfidfKeywords:
    """Rank the keywords of a document set with TF-IDF.

    Example:
        >>> keywords = TfidfKeywords().fit(["کتاب خوب", "کتاب بد", "ماشین"])
        >>> keywords.rank(top_k=1)[0].word
        'کتاب'
    """

    vocabulary: Counter[str] = field(default_factory=Counter, repr=False)
    document_frequency: Counter[str] = field(default_factory=Counter, repr=False)
    documents: int = 0

    def fit(self, documents: Iterable[str]) -> TfidfKeywords:
        """Count terms and document frequencies."""
        self.vocabulary = Counter()
        self.document_frequency = Counter()
        self.documents = 0
        for document in documents:
            words = content_words(document)
            self.vocabulary.update(words)
            self.document_frequency.update(set(words))
            self.documents += 1
        return self

    def rank(self, *, top_k: int = 20, min_frequency: int = 1) -> list[KeywordScore]:
        """Return the *top_k* keywords by TF-IDF score."""
        results: list[KeywordScore] = []
        for word, frequency in self.vocabulary.items():
            if frequency < min_frequency:
                continue
            document_frequency = self.document_frequency[word]
            idf = math.log((1 + self.documents) / (1 + document_frequency)) + 1.0
            results.append(
                KeywordScore(
                    word=word,
                    frequency=frequency,
                    document_frequency=document_frequency,
                    score=frequency * idf,
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


def extract_keywords(
    documents: Iterable[str],
    *,
    top_k: int = 20,
    min_frequency: int = 1,
) -> list[KeywordScore]:
    """Convenience wrapper around :class:`TfidfKeywords`.

    >>> [item.word for item in extract_keywords(["کتاب خوب", "کتاب بد"], top_k=1)]
    ['کتاب']
    """
    return TfidfKeywords().fit(documents).rank(top_k=top_k, min_frequency=min_frequency)


def discover_words(
    documents: Iterable[str],
    *,
    min_frequency: int = 3,
    workers: int = 4,
    top_k: int | None = 50,
) -> list[WordScore]:
    """Discover the words of a raw corpus (see :class:`WordExtractor`).

    >>> discovered = discover_words(["سلام دنیا", "سلام تهران"], min_frequency=1, workers=1, top_k=1)
    >>> [item.word for item in discovered]
    ['سلام']
    """
    extractor = WordExtractor(min_frequency=min_frequency, workers=workers).fit(documents)
    stats = extractor.extract()
    return stats[:top_k] if top_k else stats
