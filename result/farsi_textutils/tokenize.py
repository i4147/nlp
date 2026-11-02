"""Tokenisation: words, sentences, paragraphs and dictionary-based segmenters.

Sources merged here:

* ``parsivar.tokenizer.Tokenizer`` (the module was empty in the reference
  checkout; only its public API -- ``tokenize_words`` / ``tokenize_sentences`` --
  is preserved);
* ``camel_tools.tokenizers.word.simple_word_tokenize`` -- Unicode aware splitting
  of punctuation and emoji, reimplemented with :mod:`unicodedata` instead of the
  third-party ``regex`` module;
* ``soynlp.tokenizer.LTokenizer`` and ``MaxScoreTokenizer`` -- language agnostic
  dictionary segmenters, pre-seeded with Persian particles and suffixes so they
  can split ``کتاب‌ها`` or ``می‌روم`` without a trained model;
* ``camel_tools``-style sentence splitting with an abbreviation list.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, cast

from . import constants as C
from .exceptions import ConfigurationError
from .logging_setup import get_logger

__all__ = [
    "PERSIAN_PARTICLES",
    "LTokenizer",
    "MaxScoreTokenizer",
    "SentenceSplitter",
    "Tokenizer",
    "TokenizerConfig",
    "paragraph_tokenize",
    "sentence_tokenize",
    "word_tokenize",
]

_log = get_logger("tokenize")

PERSIAN_PARTICLES: Final[tuple[str, ...]] = (
    "ترین",
    "هایی",
    "هایم",
    "هایت",
    "هایش",
    "هایمان",
    "هایتان",
    "هایشان",
    "ها",
    "های",
    "ای",
    "یی",
    "ام",
    "ات",
    "اش",
    "مان",
    "تان",
    "شان",
    "می",
    "نمی",
    "بی",
)
"""Particles used to seed the dictionary based segmenters."""

_RE_PARAGRAPH: Final[re.Pattern[str]] = re.compile(r"\n\s*\n")
_RE_NUMBER: Final[re.Pattern[str]] = re.compile(r"\d+(?:[.\u066b,:/]\d+)*")
_RE_DIGITS: Final[re.Pattern[str]] = re.compile(r"\d+")


def _is_symbol(char: str) -> bool:
    r"""True when *char* is punctuation or a symbol.

    Equivalent to the ``\\p{P}``/``\\p{S}`` classes used by camel-tools; format
    characters such as the ZWNJ (``Cf``) stay inside the word on purpose.
    """
    category = unicodedata.category(char)
    return category[0] in {"P", "S"}


def _is_punctuation_token(token: str) -> bool:
    """True when *token* consists exclusively of punctuation/symbol characters.

    >>> _is_punctuation_token("؟!"), _is_punctuation_token("سلام")
    (True, False)
    """
    return bool(token) and all(_is_symbol(char) for char in token)


def word_tokenize(
    text: str,
    *,
    split_half_space: bool = False,
    split_digits: bool = False,
    keep_punctuation: bool = True,
) -> list[str]:
    """Split *text* into word and punctuation tokens.

    Args:
        text: Input text.
        split_half_space: Also split on the ZWNJ (``کتاب‌ها`` -> ``کتاب``, ``ها``).
        split_digits: Emit digit runs as separate tokens.
        keep_punctuation: Keep punctuation tokens; when ``False`` they are dropped.

    Returns:
        The list of tokens, in order.

    >>> word_tokenize("سلام، دنیا!")
    ['سلام', '،', 'دنیا', '!']
    >>> word_tokenize("قیمت ۱۲۳ تومان است", split_digits=True)
    ['قیمت', '۱۲۳', 'تومان', 'است']
    >>> word_tokenize("کتاب‌ها", split_half_space=True)
    ['کتاب', 'ها']
    """
    tokens: list[str] = []
    for word in text.split():
        if split_half_space and C.HALF_SPACE in word:
            parts: list[str] = []
            for part in word.split(C.HALF_SPACE):
                parts.extend(_split_one(part, split_digits=split_digits))
            tokens.extend(part for part in parts if part)
            continue
        tokens.extend(_split_one(word, split_digits=split_digits))
    if not keep_punctuation:
        tokens = [token for token in tokens if not _is_punctuation_token(token)]
    return [token for token in tokens if token]


def _split_one(word: str, *, split_digits: bool) -> list[str]:
    r"""Split a single whitespace-free chunk into sub-tokens."""
    pieces: list[str] = []
    buffer: list[str] = []
    index = 0
    length = len(word)
    while index < length:
        char = word[index]
        if _is_symbol(char):
            if buffer:
                pieces.append("".join(buffer))
                buffer = []
            pieces.append(char)
            index += 1
            continue
        if char.isdigit():
            if buffer:
                pieces.append("".join(buffer))
                buffer = []
            pattern = _RE_DIGITS if split_digits else _RE_NUMBER
            match = pattern.match(word, index)
            assert match is not None
            pieces.append(match.group())
            index = match.end()
            continue
        buffer.append(char)
        index += 1
    if buffer:
        pieces.append("".join(buffer))
    return pieces


class SentenceSplitter:
    """Split text into sentences, honouring Persian abbreviations and decimals.

    The reference implementation relied on NLTK's Punkt; this splitter is a
    small, dependency free state machine tailored to Persian punctuation.

    >>> splitter = SentenceSplitter()
    >>> splitter.split("دکتر رضایی آمد. او خوشحال بود! واقعا؟")
    ['دکتر رضایی آمد.', 'او خوشحال بود!', 'واقعا؟']
    >>> SentenceSplitter().split("مث. این یک مثال است. تمام")
    ['مث. این یک مثال است.', 'تمام']
    """

    __slots__ = ("_boundary", "_spaced", "abbreviations", "delimiters", "min_length")

    def __init__(
        self,
        *,
        abbreviations: Iterable[str] | None = None,
        delimiters: str = C.SENTENCE_DELIMITERS,
        min_length: int = 1,
    ) -> None:
        self.abbreviations: frozenset[str] = frozenset(abbreviations or C.ABBREVIATIONS)
        self.delimiters = delimiters
        self.min_length = max(1, min_length)
        self._boundary = re.compile(r"[" + re.escape(delimiters) + r"]+(?=\s|$)")
        self._spaced = re.compile(r"[ \t]+")

    def split(self, text: str) -> list[str]:
        """Return the sentences of *text*, in order."""
        if not text or not text.strip():
            return []
        flattened = self._spaced.sub(" ", text.replace("\n", " "))
        sentences: list[str] = []
        start = 0
        for match in self._boundary.finditer(flattened):
            if self._is_abbreviation(flattened[start : match.start()], match.group()):
                continue
            sentences.append(self._finish(flattened[start : match.end()]))
            start = match.end()
        tail = flattened[start:]
        if tail.strip():
            sentences.append(self._finish(tail))
        return [sentence for sentence in sentences if sentence and len(sentence) >= self.min_length]

    def _is_abbreviation(self, prefix: str, delimiter: str) -> bool:
        """True when *delimiter* closes an abbreviation, a number or an initial."""
        if delimiter.strip(".") != "":
            return False  # only the full stop can belong to an abbreviation
        words = prefix.split()
        if not words:
            return False
        last = words[-1].strip("«»\"'()[]")
        if not last:
            return False
        if last in self.abbreviations:
            return True
        if len(last) == 1 and not last.isdigit():
            return True  # initial such as "ع." or a numbered list marker
        return last.replace(".", "").isdigit() and len(last) <= 4

    def _finish(self, sentence: str) -> str:
        return self._spaced.sub(" ", sentence).strip()


def sentence_tokenize(text: str, *, min_length: int = 1) -> list[str]:
    """Split *text* into sentences.

    >>> sentence_tokenize("او آمد. سپس رفت.")
    ['او آمد.', 'سپس رفت.']
    """
    return SentenceSplitter(min_length=min_length).split(text)


def paragraph_tokenize(text: str) -> list[str]:
    r"""Split *text* on blank lines.

    >>> paragraph_tokenize("پاراگراف یک\n\nپاراگراف دو")
    ['پاراگراف یک', 'پاراگراف دو']
    """
    paragraphs = [C.RE_WHITESPACE.sub(" ", block).strip() for block in _RE_PARAGRAPH.split(text)]
    return [paragraph for paragraph in paragraphs if paragraph]


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """Options of :class:`Tokenizer`."""

    split_half_space: bool = False
    split_digits: bool = False
    keep_punctuation: bool = True
    sentence_min_length: int = 1


class Tokenizer:
    """Words / sentences / paragraphs tokenizer.

    The ``parsivar`` public surface (``tokenize_words``, ``tokenize_sentences``)
    is preserved, alongside the more explicit :meth:`word`, :meth:`sentence`
    and :meth:`paragraph` methods.

    >>> Tokenizer().word("می‌روم")
    ['می‌روم']
    >>> Tokenizer(TokenizerConfig(split_half_space=True)).word("کتاب‌ها")
    ['کتاب', 'ها']
    """

    __slots__ = ("_splitter", "config")

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self._splitter = SentenceSplitter(min_length=self.config.sentence_min_length)

    def word(self, text: str) -> list[str]:
        """Tokenise *text* into words and punctuation."""
        return word_tokenize(
            text,
            split_half_space=self.config.split_half_space,
            split_digits=self.config.split_digits,
            keep_punctuation=self.config.keep_punctuation,
        )

    def sentence(self, text: str) -> list[str]:
        """Split *text* into sentences."""
        return self._splitter.split(text)

    def paragraph(self, text: str) -> list[str]:
        """Split *text* into paragraphs."""
        return paragraph_tokenize(text)

    # --- parsivar compatible aliases -------------------------------------
    def tokenize_words(self, text: str) -> list[str]:
        """Alias of :meth:`word` kept for ``parsivar`` compatibility."""
        return self.word(text)

    def tokenize_sentences(self, text: str) -> list[str]:
        """Alias of :meth:`sentence` kept for ``parsivar`` compatibility."""
        return self.sentence(text)

    def __call__(self, text: str) -> list[str]:
        """Tokenise *text* (same as :meth:`word`)."""
        return self.word(text)


class LTokenizer:
    """Left-tokenizer: split each token at its highest scoring left part.

    Port of ``soynlp.tokenizer.LTokenizer`` (numpy-free).  Scores are typically
    ``{particle: -log(frequency)}`` of the part being a *left* part of a word;
    with no scores the tokenizer only applies the default score, i.e. it keeps
    words intact.

    >>> tokenizer = LTokenizer({"کتاب": 2.0, "ها": 1.0}, default_score=0.0)
    >>> tokenizer("کتاب‌ها")
    ['کتاب', 'ها']
    """

    __slots__ = ("_default_score", "_scores")

    def __init__(self, scores: Mapping[str, float] | None = None, default_score: float = 0.0) -> None:
        self._scores: dict[str, float] = dict(scores or {})
        self._default_score = default_score

    def __call__(self, sentence: str, *, flatten: bool = True) -> list[str] | list[list[str]]:
        """Tokenise *sentence* (same as :meth:`tokenize`)."""
        return self.tokenize(sentence, flatten=flatten)

    def tokenize(
        self,
        sentence: str,
        *,
        tolerance: float = 0.0,
        flatten: bool = True,
    ) -> list[str] | list[list[str]]:
        """Split *sentence* into ``(left, right)`` parts per whitespace token.

        Compared to ``soynlp`` only prefixes that actually appear in ``scores``
        can start a split, so an unknown token is never chopped in the middle.
        ``tolerance`` keeps every candidate within that score distance of the
        best one and then prefers the longest remainder.

        With ``flatten=False`` one list of segments is returned per whitespace
        token, mirroring ``soynlp``.
        """
        pairs = [self._split(token, tolerance) for token in sentence.split()]
        if flatten:
            return [stripped for pair in pairs for part in pair if (stripped := part.strip(C.HALF_SPACE))]
        return [[part for part in pair if part] for pair in pairs]

    def _split(self, token: str, tolerance: float) -> tuple[str, str]:
        length = len(token)
        if length <= 2:
            return token, ""
        known = [end for end in range(2, length + 1) if token[:end] in self._scores]
        if not known:
            return token, ""
        scored = [(self._scores[token[:end]], token[:end], token[end:]) for end in known] + [
            (self._default_score, token, "")
        ] * bool(self._default_score)
        if tolerance > 0:
            best_score = max(score for score, _, _ in scored)
            scored = [item for item in scored if best_score - item[0] <= tolerance]
            return max(scored, key=lambda item: len(item[2]))[1:]
        return max(scored, key=lambda item: (item[0], len(item[2])))[1:]


class MaxScoreTokenizer:
    """Segment a token by repeatedly taking the longest highest scoring prefix.

    Port of ``soynlp.tokenizer.MaxScoreTokenizer``; useful with the Persian
    particle dictionary built by :func:`particle_scores`.

    >>> tokenizer = MaxScoreTokenizer({"کتاب": 5.0, "ها": 1.0})
    >>> tokenizer("کتابها")
    ['کتاب', 'ها']
    """

    __slots__ = ("_default_score", "_max_length", "_scores")

    def __init__(
        self,
        scores: Mapping[str, float] | None = None,
        max_length: int = 10,
        default_score: float = 0.0,
    ) -> None:
        if max_length < 1:
            raise ConfigurationError("max_length must be >= 1")
        self._scores: dict[str, float] = dict(scores or {})
        self._max_length = max_length
        self._default_score = default_score

    def __call__(self, sentence: str, *, flatten: bool = True) -> list[str] | list[list[str]]:
        """Tokenise *sentence* (same as :meth:`tokenize`)."""
        return self.tokenize(sentence, flatten=flatten)

    def tokenize(self, sentence: str, *, flatten: bool = True) -> list[str] | list[list[str]]:
        """Tokenise *sentence*, splitting every whitespace token further."""
        segments = [self._recursive(token) for token in sentence.split()]
        if flatten:
            return [segment for token_segments in segments for segment, _ in token_segments]
        return [[segment for segment, _ in token_segments] for token_segments in segments]

    def _recursive(self, token: str, index: int = 0) -> list[tuple[str, int]]:
        if index >= len(token):
            return []
        best_length: int | None = None
        best_score = 0.0
        for length in range(1, min(self._max_length, len(token) - index) + 1):
            candidate = token[index : index + length]
            if candidate not in self._scores:
                continue
            score = self._scores[candidate] * length
            if score > best_score:
                best_length, best_score = length, score
        if best_length is None:
            # Unknown run: keep the characters together until the next known
            # match instead of emitting one token per character.
            stop = len(token)
            for position in range(index + 1, len(token)):
                window = token[position : position + self._max_length]
                if any(window[:length] in self._scores for length in range(1, len(window) + 1)):
                    stop = position
                    break
            head = token[index:stop]
            return [(head, index), *self._recursive(token, stop)]
        head = token[index : index + best_length]
        return [(head, index), *self._recursive(token, index + best_length)]


def particle_scores(
    frequencies: Mapping[str, int],
    *,
    min_count: int = 1,
) -> dict[str, float]:
    """Build ``MaxScoreTokenizer``/``LTokenizer`` scores from corpus counts.

    Words are scored by ``log(count)`` and Persian particles receive a small
    bonus so that ``های`` is preferred over ``ه`` when both occur.

    >>> scores = particle_scores({"کتاب": 10, "ها": 4})
    >>> round(scores["کتاب"], 3), round(scores["ها"], 3)
    (3.303, 2.686)
    """
    import math

    scores: dict[str, float] = {}
    for word, count in frequencies.items():
        if count >= min_count and len(word) > 1:
            scores[word] = math.log(count) + 1.0
    for particle in PERSIAN_PARTICLES:
        if particle in scores:
            scores[particle] += 0.3
    return scores


def split_by_dictionary(
    text: str,
    lexicon: Iterable[str],
    *,
    max_length: int = 10,
) -> list[str]:
    """Segment whitespace-free tokens using a :class:`MaxScoreTokenizer`.

    >>> split_by_dictionary("کتابها", ["کتاب", "ها"])
    ['کتاب', 'ها']
    """
    scores = {word: float(len(word)) for word in lexicon}
    tokens = MaxScoreTokenizer(scores, max_length=max_length)(text)
    return cast("list[str]", tokens)
