"""Persian <-> Latin transliteration (فینگلیش / pinglish).

Merges three implementations of the reference packages:

* ``farsitools.PersianTools.convert_farsi_char_to_english`` -- the Persian to
  Latin table used for file names and slugs;
* ``fakefarsi.translator`` -- the table used to build e-mail addresses;
* ``parsivar.normalizer.PinglishNormalizer`` -- Latin ("pinglish") back to
  Persian, generating every plausible spelling of a word.

The transliterator itself follows the design of
``camel_tools.utils.transliterate.Transliterator``: a :class:`~farsi_textutils.mapping.CharMapper`
plus an explicit policy for characters that must be preserved.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from . import constants as C
from .exceptions import ConfigurationError
from .logging_setup import get_logger
from .mapping import CharMapper

__all__ = [
    "MAX_PERSIAN_CANDIDATES",
    "PERSIAN_TO_LATIN",
    "PinglishConverter",
    "slugify",
    "to_latin",
    "to_persian",
]

_log = get_logger("transliterate")

PERSIAN_TO_LATIN: Final[dict[str, str]] = {
    "آ": "a",
    "ا": "a",
    "ب": "b",
    "پ": "p",
    "ت": "t",
    "ث": "s",
    "ج": "j",
    "چ": "ch",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ذ": "z",
    "ر": "r",
    "ز": "z",
    "ژ": "zh",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "z",
    "ط": "t",
    "ظ": "z",
    "ع": "a",
    "غ": "gh",
    "ف": "f",
    "ق": "gh",
    "ک": "k",
    "گ": "g",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "و": "v",
    "ه": "h",
    "ی": "i",
    "ء": "a",
    "ئ": "e",
    "أ": "a",
    "إ": "a",
    "ؤ": "o",
    "ة": "h",
    C.HALF_SPACE: "-",
    " ": "_",
    "‌": "-",
}
"""Persian letter -> Latin digraph table (``farsitools`` + ``fakefarsi``)."""

_LATIN_MAPPER: Final[CharMapper] = CharMapper(PERSIAN_TO_LATIN, name="fa->latin")

LATIN_TO_PERSIAN_ALTERNATIVES: Final[dict[str, tuple[str, ...]]] = {
    "a": ("", "ا"),
    "b": ("ب",),
    "c": ("ث", "ص", "ک"),
    "d": ("د",),
    "e": ("", "ه"),
    "f": ("ف",),
    "g": ("گ",),
    "h": ("ه", "ح"),
    "i": ("ی",),
    "j": ("ج",),
    "k": ("ک",),
    "l": ("ل",),
    "m": ("م",),
    "n": ("ن",),
    "o": ("", "و"),
    "p": ("پ",),
    "q": ("ق", "غ"),
    "r": ("ر",),
    "s": ("س", "ص", "ث"),
    "t": ("ت", "ط"),
    "u": ("و",),
    "v": ("و",),
    "w": ("و",),
    "y": ("ی",),
    "z": ("ز", "ذ", "ض", "ظ"),
    "ch": ("چ",),
    "kh": ("خ",),
    "gh": ("ق", "غ"),
    "sh": ("ش",),
    "zh": ("ژ",),
    "aa": ("ا",),
    "ee": ("ی",),
    "oo": ("و",),
    "ou": ("و",),
    " ": ("",),
}
"""Latin digraph -> candidate Persian spellings (from ``PinglishNormalizer``)."""

_DIGRAPHS: Final[tuple[str, ...]] = ("ch", "kh", "gh", "sh", "zh", "aa", "ee", "oo", "ou")

MAX_PERSIAN_CANDIDATES: Final[int] = 24
"""Upper bound on the spellings produced for a single pinglish word."""


def to_latin(text: str) -> str:
    """Transliterate Persian *text* into Latin letters.

    Characters without a mapping are kept as-is.

    >>> to_latin("سلام")
    'slam'
    >>> to_latin("کتاب‌ها")
    'ktab-ha'
    """
    return _LATIN_MAPPER.map_string(text)


def slugify(text: str, *, separator: str = "-", max_length: int = 80) -> str:
    """Build an ASCII slug from Persian *text*.

    >>> slugify("سلام دنیا")
    'slam-dnia'
    """
    if len(separator) != 1:
        raise ConfigurationError("separator must be a single character")
    latin = to_latin(text).lower()
    decomposed = unicodedata.normalize("NFKD", latin)
    ascii_only = "".join(char for char in decomposed if not unicodedata.combining(char))
    cleaned = [char if char.isalnum() else separator for char in ascii_only]
    slug = "".join(cleaned)
    while separator * 2 in slug:
        slug = slug.replace(separator * 2, separator)
    return slug.strip(separator)[:max_length]


@dataclass(frozen=True, slots=True)
class PinglishConverter:
    """Convert Latin-script Persian ("pinglish") back to Persian spellings.

    The reference ``PinglishNormalizer`` produced the full cartesian product of
    candidate spellings, which explodes for long words.  This implementation
    caps the result (``max_candidates``), normalises the vowel digraphs first and
    keeps the candidates deterministic.

    >>> converter = PinglishConverter()
    >>> "کتاب" in converter.candidates("ketab")
    True
    """

    max_candidates: int = MAX_PERSIAN_CANDIDATES
    alternatives: dict[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ConfigurationError("max_candidates must be >= 1")

    @property
    def table(self) -> dict[str, tuple[str, ...]]:
        """The Latin -> Persian candidate table in use."""
        return dict(self.alternatives or LATIN_TO_PERSIAN_ALTERNATIVES)

    def candidates(self, word: str) -> list[str]:
        """Return every plausible Persian spelling of *word*.

        >>> PinglishConverter(max_candidates=4).candidates("man")[0]
        'من'
        """
        units = self._segment(word.lower())
        results: list[list[str]] = [[]]
        for unit in units:
            options = self.table.get(unit, (unit,))
            expanded: list[list[str]] = []
            for prefix in results:
                for option in options:
                    expanded.append([*prefix, option])
                    if len(expanded) >= self.max_candidates:
                        break
                if len(expanded) >= self.max_candidates:
                    break
            results = expanded
        return ["".join(parts) for parts in results]

    def convert(self, text: str, *, lexicon: Iterable[str] | None = None) -> str:
        """Convert *text*, one spelling per word.

        Pinglish is ambiguous (``a`` can be ``ا`` or nothing at all), so an
        optional *lexicon* is the reliable way to pick the right spelling: the
        first candidate that is a known word wins, otherwise the first candidate
        of the reference table is returned unchanged.

        >>> PinglishConverter().convert("ketab", lexicon={"کتاب"})
        'کتاب'
        >>> PinglishConverter().convert("salam", lexicon={"سلام"})
        'سلام'
        """
        known = frozenset(lexicon) if lexicon is not None else frozenset()
        words: list[str] = []
        for word in text.split():
            candidates = self.candidates(word)
            if not candidates:
                words.append(word)
                continue
            choice = next((candidate for candidate in candidates if candidate in known), candidates[0])
            words.append(choice)
        return " ".join(words)

    # --- internals --------------------------------------------------------
    @staticmethod
    def _segment(word: str) -> list[str]:
        """Split *word* into Latin digraphs/letters."""
        units: list[str] = []
        index = 0
        while index < len(word):
            for digraph in _DIGRAPHS:
                if word.startswith(digraph, index):
                    units.append(digraph)
                    index += len(digraph)
                    break
            else:
                units.append(word[index])
                index += 1
        return units


def to_persian(
    text: str,
    *,
    max_candidates: int = MAX_PERSIAN_CANDIDATES,
    lexicon: Iterable[str] | None = None,
) -> str:
    """Convert pinglish *text* to Persian.

    Only segments that look like Latin words are converted; digits, punctuation
    and already-Persian text pass through untouched.

    >>> to_persian("salam donya", lexicon={"سلام", "دنیا"})
    'سلام دنیا'
    """
    converter = PinglishConverter(max_candidates=max_candidates)
    known = frozenset(lexicon) if lexicon is not None else None
    pieces: list[str] = []
    buffer: list[str] = []
    for char in text:
        if char.isascii() and char.isalpha():
            buffer.append(char)
        else:
            if buffer:
                pieces.append(converter.convert("".join(buffer), lexicon=known))
                buffer = []
            pieces.append(char)
    if buffer:
        pieces.append(converter.convert("".join(buffer), lexicon=known))
    return "".join(pieces)


def alphabets_of(text: str) -> set[str]:
    """Return the scripts present in *text* (``persian``, ``latin``, ``other``).

    >>> sorted(alphabets_of("سلام hello"))
    ['latin', 'persian']
    """
    scripts: set[str] = set()
    for char in text:
        if not char.isalpha():
            continue
        if char in C.PERSIAN_LETTERS or char in C.ARABIC_VARIANTS:
            scripts.add("persian")
        elif char.isascii():
            scripts.add("latin")
        else:
            scripts.add("other")
    return scripts
