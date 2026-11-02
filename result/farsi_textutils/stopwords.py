"""Persian stop words and word-level filtering.

``farsi_tools.stop_words()`` returned a fresh list of >1000 entries on *every*
call.  Two problems with that list: it mixes punctuation, digits, pronouns and
very common adverbs with content words (``بزرگ``, ``خوب``), and rebuilding it
each time is wasteful.  Here:

* :data:`STOP_WORDS` keeps the reference list untouched (provenance), frozen at
  import time in :mod:`farsi_textutils.data.stopwords`;
* :data:`FUNCTION_WORDS` is a curated closed-class list (pronouns, prepositions,
  conjunctions, determiners, copulas);
* :func:`content_words` defaults to the ``function`` preset, which keeps
  adjectives such as ``بزرگ`` -- better for keywords and retrieval -- and the
  ``full`` preset reproduces the old, aggressive behaviour.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Final, Literal

from .data.stopwords import STOP_WORDS
from .logging_setup import get_logger
from .tokenize import word_tokenize

__all__ = [
    "FUNCTION_WORDS",
    "STOP_WORDS",
    "StopPreset",
    "content_words",
    "is_stop_word",
    "normalise_frequencies",
    "remove_stop_words",
    "word_frequencies",
]

_log = get_logger("stopwords")

StopPreset = Literal["function", "full"]

FUNCTION_WORDS: Final[frozenset[str]] = frozenset(
    {
        # pronouns
        "من",
        "تو",
        "او",
        "ما",
        "شما",
        "آنها",
        "ایشان",
        "اون",
        "اونا",
        "وی",
        "خود",
        "خودم",
        "خودت",
        "خودش",
        "خودمان",
        "خودتان",
        "خودشان",
        "اینها",
        # prepositions / postpositions
        "از",
        "به",
        "با",
        "در",
        "بر",
        "برای",
        "تا",
        "جز",
        "مانند",
        "نزد",
        "روی",
        "درباره",
        "بدون",
        "پس",
        "قبل",
        "بعد",
        "میان",
        "طی",
        "جهت",
        "مطابق",
        "علیه",
        "درون",
        "بیرون",
        "زیر",
        "بالای",
        "کنار",
        "پیش",
        "را",
        "رو",
        # conjunctions / subordinators
        "و",
        "یا",
        "اما",
        "ولی",
        "که",
        "اگر",
        "چون",
        "زیرا",
        "بنابراین",
        "هم",
        "نه",
        "بله",
        "آری",
        "بلکه",
        "هرگاه",
        "چنانچه",
        "وقتی",
        "چرا",
        "چگونه",
        # determiners / quantifiers
        "این",
        "آن",
        "هر",
        "همه",
        "هیچ",
        "چند",
        "بعضی",
        "برخی",
        "تمام",
        "دیگر",
        "چنین",
        "چنان",
        "کدام",
        "چه",
        "تنها",
        "هریک",
        "همان",
        "همین",
        # copulas and auxiliaries
        "است",
        "هست",
        "نیست",
        "بود",
        "بودم",
        "بودی",
        "بودیم",
        "بودید",
        "بودند",
        "شد",
        "شده",
        "شدن",
        "کرد",
        "کردم",
        "کردی",
        "کردیم",
        "کردید",
        "کردند",
        "باشد",
        "باشم",
        "باشی",
        "باشیم",
        "باشید",
        "باشند",
        "باید",
        "خواهد",
        "خواهم",
        "خواهی",
        "خواهیم",
        "خواهید",
        "خواهند",
        # high frequency adverbs that carry no topic information
        "نیز",
        "همچنین",
        "فقط",
        "البته",
        "باز",
        "هرگز",
        "همیشه",
        "دوباره",
        "تقریبا",
        "کاملا",
        "حتما",
        "شاید",
        "چقدر",
    }
)
"""Curated closed-class words: safe to remove in every task."""

EXTRA_STOP_WORDS: Final[frozenset[str]] = frozenset(
    {"آنجا", "اینجا", "آنکه", "اینکه", "کسی", "چیزی", "جایی", "طور", "مورد", "قرار"}
)
"""Filler nouns that behave like function words in Persian text."""

_STOP_PRESETS: Final[dict[str, frozenset[str]]] = {
    "function": FUNCTION_WORDS | EXTRA_STOP_WORDS,
    "full": FUNCTION_WORDS | EXTRA_STOP_WORDS | STOP_WORDS,
}

_WORD_TOKEN: Final[re.Pattern[str]] = re.compile(r"^[^\W\d_]+(?:\u200c[^\W\d_]+)*$")
"""Matches alphabetic tokens, allowing the ZWNJ as an internal separator."""


def _is_word_token(token: str) -> bool:
    """True when *token* is made of letters (optionally joined by ZWNJs).

    >>> _is_word_token("کتاب‌های"), _is_word_token("،")
    (True, False)
    """
    return bool(_WORD_TOKEN.match(token))


def is_stop_word(token: str, *, preset: StopPreset = "function") -> bool:
    """True when *token* is punctuation, a digit or a stop word.

    >>> is_stop_word("و"), is_stop_word("کتاب"), is_stop_word("بزرگ", preset="full")
    (True, False, True)
    """
    if not _is_word_token(token):
        return True
    return token in _STOP_PRESETS[preset]


def remove_stop_words(
    tokens: Iterable[str],
    *,
    extra: Iterable[str] = (),
    preset: StopPreset = "function",
) -> list[str]:
    """Drop stop words (and non-word tokens) from *tokens*.

    Args:
        tokens: Input tokens.
        extra: Additional words to drop.
        preset: ``function`` (default, keeps content words) or ``full``.

    >>> remove_stop_words(["این", "کتاب", "بزرگ", "است"])
    ['کتاب', 'بزرگ']
    """
    blocked = _STOP_PRESETS[preset] | frozenset(extra)
    return [token for token in tokens if _is_word_token(token) and token not in blocked]


def content_words(
    text: str,
    *,
    extra: Iterable[str] = (),
    preset: StopPreset = "function",
) -> list[str]:
    """Tokenise *text* and keep only its content words.

    >>> content_words("کتاب‌های خوب در خانه است")
    ['کتاب‌های', 'خوب', 'خانه']
    """
    return remove_stop_words(word_tokenize(text), extra=extra, preset=preset)


def word_frequencies(
    texts: Iterable[str],
    *,
    remove_stops: bool = True,
    extra: Iterable[str] = (),
    preset: StopPreset = "function",
) -> Counter[str]:
    """Count word occurrences across *texts*.

    >>> word_frequencies(["سلام دنیا", "سلام"]).most_common(1)
    [('سلام', 2)]
    """
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = word_tokenize(text)
        counter.update(remove_stop_words(tokens, extra=extra, preset=preset) if remove_stops else tokens)
    return counter


def normalise_frequencies(frequencies: Mapping[str, int]) -> dict[str, float]:
    """Turn raw counts into relative frequencies summing to ``1``.

    >>> normalise_frequencies({"a": 1, "b": 3})
    {'a': 0.25, 'b': 0.75}
    """
    total = sum(frequencies.values())
    if total <= 0:
        return dict.fromkeys(frequencies, 0.0)
    return {word: count / total for word, count in frequencies.items()}
