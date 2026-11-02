"""Lexical similarity and retrieval (BM25, TF-IDF, co-occurrence vectors).

Extracted from the reference packages and made Persian aware:

* ``snownlp.sim.BM25`` -- the BM25 ranking function, rewritten as an immutable
  scorer with pre-computed IDF (no mutable global state);
* ``snownlp.summary`` style TF-IDF weighting;
* ``soynlp.vectorizer.WordContextVectorizer`` -- distributional word vectors
  built from co-occurrence counts (numpy-free; returns plain dicts).

These replace embedding models (torch / ONNX / sentencepiece) for the two jobs
that actually matter in a text-utils library: *ranking* documents for a query and
*finding similar words*.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, TypeVar

from .logging_setup import get_logger
from .stopwords import content_words

__all__ = [
    "BM25",
    "DocumentScore",
    "TfidfVectorizer",
    "WordContextVectorizer",
    "cosine_similarity",
    "jaccard_similarity",
]

_log = get_logger("similarity")

T = TypeVar("T")

DEFAULT_K1: Final[float] = 1.5
DEFAULT_B: Final[float] = 0.75


def _tokenize(text: str | Sequence[str]) -> list[str]:
    """Tokenise *text* into content words (or pass a token list through)."""
    return content_words(text) if isinstance(text, str) else list(text)


@dataclass(frozen=True, slots=True)
class DocumentScore:
    """A ranked document.

    >>> DocumentScore(1, "سلام", 2.5).index
    1
    """

    index: int
    text: str
    score: float


@dataclass(slots=True)
class BM25:
    """Okapi BM25 ranker over Persian documents.

    Args:
        k1: Term-frequency saturation parameter.
        b: Length normalisation parameter.

    Example:
        >>> bm25 = BM25().fit(["کتاب خوب است", "ماشین سریع است", "کتاب جدید"])
        >>> bm25.rank("ماشین")[0].index
        1
    """

    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    documents: list[str] = field(default_factory=list)
    _term_frequencies: list[Counter[str]] = field(default_factory=list, repr=False)
    _lengths: list[int] = field(default_factory=list, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _average_length: float = 0.0

    def fit(self, documents: Iterable[str]) -> BM25:
        """Index *documents*; returns ``self`` so calls can be chained."""
        self.documents = list(documents)
        self._term_frequencies = []
        self._lengths = []
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            tokens = _tokenize(document)
            frequencies = Counter(tokens)
            self._term_frequencies.append(frequencies)
            self._lengths.append(len(tokens))
            document_frequency.update(frequencies.keys())
        total = len(self.documents) or 1
        self._average_length = sum(self._lengths) / total if self._lengths else 0.0
        self._idf = {
            word: math.log((total - frequency + 0.5) / (frequency + 0.5) + 1.0)
            for word, frequency in document_frequency.items()
        }
        return self

    @property
    def idf(self) -> Mapping[str, float]:
        """Inverse document frequency of every indexed term."""
        return self._idf

    def score(self, query: str | Sequence[str], index: int) -> float:
        """BM25 score of document *index* for *query*."""
        if not self._term_frequencies:
            raise ValueError("BM25 must be fitted before scoring")
        terms = _tokenize(query)
        frequencies = self._term_frequencies[index]
        length = self._lengths[index] or 1
        score = 0.0
        for term in terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            idf = self._idf.get(term, 0.0)
            denominator = frequency + self.k1 * (1 - self.b + self.b * length / (self._average_length or 1))
            score += idf * (frequency * (self.k1 + 1)) / denominator
        return score

    def rank(self, query: str | Sequence[str], *, top_k: int | None = None) -> list[DocumentScore]:
        """Return documents ordered by decreasing relevance."""
        scored = [
            DocumentScore(index, self.documents[index], self.score(query, index))
            for index in range(len(self.documents))
        ]
        ranked = sorted(scored, key=lambda item: item.score, reverse=True)
        return ranked[:top_k] if top_k else ranked


@dataclass(slots=True)
class TfidfVectorizer:
    """TF-IDF vectors and cosine similarity for Persian documents.

    Example:
        >>> vectorizer = TfidfVectorizer().fit(["کتاب خوب", "کتاب بد"])
        >>> round(vectorizer.similarity("کتاب", "کتاب خوب"), 2) > 0
        True
    """

    vocabulary: dict[str, int] = field(default_factory=dict)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)

    def fit(self, documents: Iterable[str]) -> TfidfVectorizer:
        """Learn the vocabulary and IDF weights."""
        tokenised = [_tokenize(document) for document in documents]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenised:
            document_frequency.update(set(tokens))
        total = len(tokenised) or 1
        vocabulary = sorted(document_frequency)
        self.vocabulary = {word: index for index, word in enumerate(vocabulary)}
        self._idf = {
            word: math.log((1 + total) / (1 + frequency)) + 1.0
            for word, frequency in document_frequency.items()
        }
        return self

    def transform(self, text: str) -> dict[str, float]:
        """Return the sparse TF-IDF vector of *text*."""
        tokens = _tokenize(text)
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = len(tokens)
        return {token: (count / total) * self._idf.get(token, 1.0) for token, count in counts.items()}

    def similarity(self, first: str, second: str) -> float:
        """Cosine similarity of two texts in ``[0, 1]``."""
        return cosine_similarity(self.transform(first), self.transform(second))


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Cosine similarity of two sparse vectors.

    >>> cosine_similarity({"a": 1.0}, {"a": 1.0})
    1.0
    >>> cosine_similarity({"a": 1.0}, {"b": 1.0})
    0.0
    """
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in shared)
    if numerator == 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm)


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    """Jaccard similarity of two token sets.

    >>> round(jaccard_similarity("کتاب خوب", "کتاب بد"), 3)
    0.333
    """
    left_set = set(_tokenize(left) if isinstance(left, str) else list(left))
    right_set = set(_tokenize(right) if isinstance(right, str) else list(right))
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


@dataclass(slots=True)
class WordContextVectorizer:
    """Distributional word vectors from a co-occurrence window.

    Port of ``soynlp.vectorizer.WordContextVectorizer`` (numpy removed): vectors
    are ``dict[str, float]`` normalised to unit length, so similarity is a plain
    dot product.

    Args:
        window: Number of neighbours on each side used as context.
        min_count: Ignore context words seen fewer than this many times.

    Example:
        >>> vectorizer = WordContextVectorizer(window=2, min_count=1)
        >>> vectorizer.fit(["کتابخانه بزرگ است", "دفتر بزرگ است"]) is vectorizer
        True
        >>> sorted(vectorizer.vectors)
        ['بزرگ', 'دفتر', 'کتابخانه']
        >>> vectorizer.similarity("کتابخانه", "دفتر") > 0
        True
    """

    window: int = 3
    min_count: int = 2
    vectors: dict[str, dict[str, float]] = field(default_factory=dict)

    def fit(self, documents: Iterable[str]) -> WordContextVectorizer:
        """Build unit-length context vectors for every word; returns ``self``."""
        co_occurrence: dict[str, Counter[str]] = {}
        for document in documents:
            tokens = _tokenize(document)
            for index, token in enumerate(tokens):
                contexts = co_occurrence.setdefault(token, Counter())
                start = max(0, index - self.window)
                end = min(len(tokens), index + self.window + 1)
                for other_index in range(start, end):
                    if other_index != index:
                        contexts[tokens[other_index]] += 1
        vectors: dict[str, dict[str, float]] = {}
        for token, contexts in co_occurrence.items():
            filtered = {word: float(count) for word, count in contexts.items() if count >= self.min_count}
            norm = math.sqrt(sum(value * value for value in filtered.values()))
            if norm:
                vectors[token] = {word: value / norm for word, value in filtered.items()}
        self.vectors = vectors
        return self

    def vector(self, word: str) -> dict[str, float]:
        """Return the vector of *word* (empty when unknown)."""
        return self.vectors.get(word, {})

    def similarity(self, first: str, second: str) -> float:
        """Similarity of two words."""
        return cosine_similarity(self.vector(first), self.vector(second))

    def most_similar(self, word: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        """Return the *top_k* nearest words by cosine similarity."""
        target = self.vector(word)
        if not target:
            return []
        scored = [
            (candidate, cosine_similarity(target, vector))
            for candidate, vector in self.vectors.items()
            if candidate != word
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
