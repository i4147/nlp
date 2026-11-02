"""Digit systems, number spelling and ordinal suffixes.

Consolidates ``persian_tools.digits``, ``persian_tools.separator`` and
``persian_tools.ordinal_suffix`` into one module with an explicit ``Language``
enum instead of the ``LANGUAGES`` string constants, validated inputs and
``TypeIs``-free, fully typed return values.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final

from . import constants as C
from .data.numbers import (
    JOINERS,
    MAGNITUDE,
    NUMBER_TEXT,
    PREFIXES,
    SCALE,
    TEN,
    TYPO_LIST,
    UNITS,
)
from .exceptions import ConfigurationError, InvalidValueError
from .logging_setup import get_logger

__all__ = [
    "DIGIT_ALPHABETS",
    "Language",
    "add_ordinal_suffix",
    "add_separator",
    "convert_digits",
    "digits_to_number",
    "number_to_words",
    "remove_ordinal_suffix",
    "remove_separator",
    "to_arabic_digits",
    "to_english_digits",
    "to_persian_digits",
    "words_to_number",
]

_log = get_logger("digits")

MAX_WORD_VALUE: Final[int] = 10**18
"""Largest supported magnitude; anything above returns ``None`` like the source."""


class Language(StrEnum):
    """Digit alphabets supported by the converters."""

    ENGLISH = "en"
    PERSIAN = "fa"
    ARABIC = "ar"


DIGIT_ALPHABETS: Final[dict[Language, str]] = {
    Language.ENGLISH: C.ASCII_DIGITS,
    Language.PERSIAN: C.PERSIAN_DIGITS,
    Language.ARABIC: C.ARABIC_DIGITS,
}

_SEPARATOR_PATTERN: Final[re.Pattern[str]] = re.compile(r"(\d)(?=(\d{3})+(?!\d))")


def convert_digits(number: int | float | str, destination: Language = Language.PERSIAN) -> str:
    """Rewrite every digit of *number* with the *destination* alphabet.

    Args:
        number: Value or string holding ASCII, Persian or Arabic digits.
        destination: Target alphabet.

    Returns:
        The converted string.

    >>> convert_digits(1402, Language.PERSIAN)
    '۱۴۰۲'
    >>> convert_digits("۱۲۳", Language.ENGLISH)
    '123'
    >>> convert_digits(3.5, Language.ARABIC)
    '٣.٥'
    """
    text = str(number)
    for alphabet in DIGIT_ALPHABETS.values():
        for index, char in enumerate(alphabet):
            text = text.replace(char, DIGIT_ALPHABETS[destination][index])
    return text


def to_persian_digits(number: int | float | str) -> str:
    """Convert to Persian digits.

    >>> to_persian_digits("2024")
    '۲۰۲۴'
    """
    return convert_digits(number, Language.PERSIAN)


def to_arabic_digits(number: int | float | str) -> str:
    """Convert to Arabic-Indic digits.

    >>> to_arabic_digits("2024")
    '٢٠٢٤'
    """
    return convert_digits(number, Language.ARABIC)


def to_english_digits(number: int | float | str) -> str:
    """Convert to ASCII digits.

    >>> to_english_digits("۱۲۳")
    '123'
    """
    return convert_digits(number, Language.ENGLISH)


def _words_under_thousand(number: int) -> str:
    """Spell an integer below 1000 (``0`` -> ``''``)."""
    parts: list[str] = []
    remaining = number
    for unit in (100, 10, 1):
        chunk = (remaining // unit) * unit
        if chunk == 0:
            continue
        if remaining in NUMBER_TEXT:
            parts.append(NUMBER_TEXT[remaining])
            break
        parts.append(NUMBER_TEXT[chunk])
        remaining %= unit
    return " و ".join(parts)


def number_to_words(number: int, *, ordinal: bool = False) -> str | None:
    """Spell an integer in Persian.

    Args:
        number: Value to spell (may be negative).
        ordinal: Append the ordinal suffix (``یک`` -> ``یکم``).

    Returns:
        The spelled number, or ``None`` when it exceeds :data:`MAX_WORD_VALUE`.

    >>> number_to_words(2030)
    'دو هزار و سی'
    >>> number_to_words(-5)
    'منفی پنج'
    >>> number_to_words(3, ordinal=True)
    'سوم'

    Raises:
        InvalidValueError: When *number* is not an integer.
    """
    if isinstance(number, bool) or not isinstance(number, int):
        raise InvalidValueError(f"{number!r} is not an integer")
    if number == 0:
        return "صفر" if not ordinal else "صفرم"
    if abs(number) >= MAX_WORD_VALUE:
        return None

    negative = number < 0
    remaining = abs(number)
    groups: list[str] = []
    index = 0
    while remaining > 0:
        chunk_value = remaining % 1000
        if chunk_value:
            chunk = _words_under_thousand(chunk_value)
            scale = SCALE[index] if index < len(SCALE) else ""
            groups.append(f"{chunk} {scale}".strip())
        remaining //= 1000
        index += 1

    words = " و ".join(reversed(groups))
    if negative:
        words = f"منفی {words}"
    return add_ordinal_suffix(words) if ordinal else words


def _tokenise_number_words(text: str) -> list[str]:
    """Split a spelled number into tokens, fixing common typos."""
    for typo, replacement in TYPO_LIST.items():
        text = text.replace(typo, replacement)
    tokens = (token.strip("،,.;؛") for token in text.split(" "))
    return [token for token in tokens if token and token not in JOINERS]


def words_to_number(
    text: str | None,
    *,
    digits: Language = Language.ENGLISH,
    separator: bool = False,
) -> int | str | None:
    """Parse a spelled Persian number.

    Args:
        text: Spelled number; ``None``/empty input returns ``None``.
        digits: Alphabet of the returned string form.
        separator: Group the thousands with a comma.

    Returns:
        The numeric value (as ``str`` when *digits* is not English, ``int``
        otherwise) or ``None`` for empty input.

    >>> words_to_number("دو هزار و سی")
    2030
    >>> words_to_number("شش صد و پنجاه", digits=Language.PERSIAN)
    '۶۵۰'
    """
    if text is None or not text.strip():
        return None

    total = 0
    negative = False
    matched = False
    for token in _tokenise_number_words(remove_ordinal_suffix(text)):
        token = to_english_digits(token)
        if token in PREFIXES:
            negative = token == PREFIXES[0]
        elif token in UNITS:
            total += UNITS[token]
            matched = True
        elif token in TEN:
            total += TEN[token]
            matched = True
        elif token in MAGNITUDE:
            total = max(total, 1) * MAGNITUDE[token]
            matched = True
        elif token.isdigit():
            total += int(token)
            matched = True
    if not matched:
        raise InvalidValueError(f"no number words found in {text!r}")
    value = -total if negative else total
    if separator:
        return add_separator(value)
    if digits is Language.ENGLISH:
        return value
    return convert_digits(value, digits)


def digits_to_number(number: int | float | str) -> int | float:
    """Parse a digit string written in any supported alphabet.

    >>> digits_to_number("۱٫۵")
    1.5
    >>> digits_to_number("١٢٣")
    123
    """
    text = to_english_digits(str(number)).replace("٫", ".").replace(",", "")
    try:
        value = float(text)
    except ValueError as error:
        raise InvalidValueError(f"{number!r} is not a number") from error
    return int(value) if value.is_integer() else value


def add_separator(number: int | float | str, separator: str = ",") -> str:
    """Group the integer part of *number* in threes.

    >>> add_separator(1234567)
    '1,234,567'
    >>> add_separator("1234567.89")
    '1,234,567.89'
    """
    if len(separator) != 1:
        raise ConfigurationError("separator must be a single character")
    text = str(number)
    integer_part, dot, fraction = text.partition(".")
    grouped = _SEPARATOR_PATTERN.sub(rf"\1{separator}", integer_part)
    return grouped + dot + fraction


def remove_separator(number: str, separator: str = ",") -> str:
    """Drop the grouping separator from *number*.

    >>> remove_separator("1,234,567")
    '1234567'
    >>> remove_separator("۱٬۲۳۴")
    '۱۲۳۴'
    """
    separators = {separator, ",", "\u066c", "\u060c", "_", " "}
    return "".join(char for char in number if char not in separators)


def add_ordinal_suffix(word: str) -> str:
    """Turn a spelled number into its ordinal form.

    >>> add_ordinal_suffix("سه")
    'سوم'
    >>> add_ordinal_suffix("ده")
    'دهم'
    >>> add_ordinal_suffix("بیست")
    'بیستم'
    """
    if word.isdigit() or to_english_digits(word).isdigit():
        spelled = number_to_words(int(to_english_digits(word)))
        if spelled is None:  # pragma: no cover - number_to_words raises instead
            raise InvalidValueError(f"cannot spell {word!r}")
        return add_ordinal_suffix(spelled)
    if word.endswith("سه"):
        return f"{word[:-2]}سوم"
    if word.endswith("ی"):
        return f"{word} اُم"
    return f"{word}م"


def remove_ordinal_suffix(word: str) -> str:
    """Strip an ordinal suffix if present.

    >>> remove_ordinal_suffix("سوم")
    'سه'
    >>> remove_ordinal_suffix("بیستم")
    'بیست'
    """
    for marker in ("مین", " اُم", " ام"):
        word = word.replace(marker, "")
    if word.endswith("سوم"):
        return f"{word[:-3]}سه"
    if word.endswith("م") and len(word) > 1:
        return word[:-1]
    return word


def clamp(value: int, minimum: int, maximum: int) -> int:
    """Small helper used by the fake-data generators.

    >>> clamp(12, 0, 10)
    10
    """
    return max(minimum, min(maximum, value))
