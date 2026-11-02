"""Date and number-word normalisation.

Ported from ``parsivar.normalizer.DateNormalizer``: dates written as
``1402/05/12`` or ``12 مرداد 1402`` are rewritten in a canonical ``yYmMdD``
form, and Persian number words (``دو هزار و سی``) are collapsed to digits.
The recursion of the original was replaced by an iterative scanner and all
state lives in module level tables.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from . import constants as C
from .data.numbers import MAGNITUDE, NUMBER_TEXT, SCALE, TEN, UNITS
from .logging_setup import get_logger

__all__ = [
    "NUMBER_WORDS",
    "ORDINAL_WORDS",
    "DateNormalizer",
    "normalize_dates",
    "normalize_numbers",
    "words_to_number",
]

_log = get_logger("dates")

ORDINAL_WORDS: Final[dict[str, int]] = {
    "اول": 1,
    "یکم": 1,
    "دوم": 2,
    "سوم": 3,
    "چهارم": 4,
    "پنجم": 5,
    "ششم": 6,
    "هفتم": 7,
    "هشتم": 8,
    "نهم": 9,
    "دهم": 10,
    "یازدهم": 11,
    "دوازدهم": 12,
    "سیزدهم": 13,
    "چهاردهم": 14,
    "پانزدهم": 15,
    "شانزدهم": 16,
    "هفدهم": 17,
    "هجدهم": 18,
    "نوزدهم": 19,
    "بیستم": 20,
    "سی‌ام": 30,
    "چهلم": 40,
    "پنجاهم": 50,
    "شصتم": 60,
    "هفتادم": 70,
    "هشتادم": 80,
    "نودم": 90,
    "صدم": 100,
    "دویستم": 200,
    "سیصدم": 300,
    "چهارصدم": 400,
    "پانصدم": 500,
    "ششصدم": 600,
    "هفتصدم": 700,
    "هشتصدم": 800,
    "نهصدم": 900,
    "هزارم": 1000,
}

NUMBER_WORDS: Final[dict[str, int]] = {
    **{word: value for value, word in NUMBER_TEXT.items() if word},
    **UNITS,
    **TEN,
    **MAGNITUDE,
    **ORDINAL_WORDS,
    " ": 0,
}
"""Every Persian word that carries a numeric value."""

_JOINER: Final[str] = "و"
_DECIMAL_POINT: Final[str] = "٫"
_RE_DATE_NUMERIC: Final[re.Pattern[str]] = re.compile(r"(?<!\d)(\d{1,4})/(\d{1,2})(?:/(\d{1,2}))?(?!\d)")


def _is_digit(token: str) -> bool:
    """True for an integer/float token written with ASCII or Persian digits."""
    candidate = token.translate(C.DIGIT_TRANSLATION).replace(_DECIMAL_POINT, ".")
    return bool(candidate) and re.fullmatch(r"[-+]?\d*\.?\d+", candidate) is not None


def _to_float(token: str) -> float:
    """Parse a token that may use Persian digits or the Persian decimal mark."""
    return float(token.translate(C.DIGIT_TRANSLATION).replace(_DECIMAL_POINT, "."))


def _is_number(token: str) -> bool:
    """True when *token* is a number word or a numeric literal."""
    return token in NUMBER_WORDS or _is_digit(token)


def words_to_number(tokens: Sequence[str]) -> int | float:
    """Convert a run of Persian number words into its numeric value.

    ``و`` separates additive groups, magnitudes multiply the group before them.

    >>> words_to_number(["دو", "هزار", "و", "سی"])
    2030
    >>> words_to_number(["سه", "و", "نیم"])
    3
    """
    total = 0.0
    group = 1.0
    has_group = False
    for token in tokens:
        if token == _JOINER:
            if has_group:
                total += group
            group = 1.0
            has_group = False
            continue
        if token in MAGNITUDE:
            group *= MAGNITUDE[token]
            has_group = True
        elif _is_digit(token):
            group *= _to_float(token)
            has_group = True
        elif token in NUMBER_WORDS:
            group *= NUMBER_WORDS[token]
            has_group = True
    if has_group:
        total += group
    return int(total) if float(total).is_integer() else total


def _format_number(value: float) -> str:
    """Render a float produced by :func:`words_to_number` without a ``.0`` tail."""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


@dataclass(frozen=True, slots=True)
class DateNormalizer:
    """Canonicalise dates and number words in a token stream.

    >>> DateNormalizer().normalize("تاریخ 12 مرداد 1402 بود")
    'تاریخ y1402m5d12 بود'
    >>> DateNormalizer().normalize("سه هزار و پانصد تومان")
    '3500 تومان'
    """

    convert_numbers: bool = True
    convert_dates: bool = True

    # --- date detection ---------------------------------------------------
    def find_date(self, tokens: Sequence[str]) -> tuple[str, int, int] | None:
        """Locate the first date in *tokens*.

        Returns:
            ``(canonical, start_index, end_index)`` or ``None``.
        """
        for index, token in enumerate(tokens):
            if token == "/":
                found = self._date_from_slashes(tokens, index)
                if found is not None:
                    return found
            if token in C.MONTHS or token in {"سال", "ماه", "روز"}:
                found = self._date_from_month(tokens, index)
                if found is not None:
                    return found
        return None

    def _date_from_slashes(self, tokens: Sequence[str], index: int) -> tuple[str, int, int] | None:
        """Handle ``day / month / year`` and ``year / month / day`` layouts."""
        if index == 0 or index + 1 >= len(tokens):
            return None
        if not (_is_digit(tokens[index - 1]) and _is_digit(tokens[index + 1])):
            return None
        first = int(_to_float(tokens[index - 1]))
        second = int(_to_float(tokens[index + 1]))
        if index + 3 < len(tokens) and tokens[index + 2] == "/" and _is_digit(tokens[index + 3]):
            third = int(_to_float(tokens[index + 3]))
            day, month, year = first, second, third
            return _canonical_date(year, month, day), index - 1, index + 3
        return _canonical_date(0, second, first), index - 1, index + 1

    def _date_from_month(self, tokens: Sequence[str], index: int) -> tuple[str, int, int] | None:
        """Handle ``day <month-name> year`` and ``<month-name> year`` layouts."""
        token = tokens[index]
        month = C.MONTHS.get(token)
        if month is not None and index + 1 < len(tokens) and _is_number(tokens[index + 1]):
            year = int(words_to_number([tokens[index + 1]]))
            if index - 1 >= 0 and _is_number(tokens[index - 1]):
                day = int(words_to_number([tokens[index - 1]]))
                return _canonical_date(year, month, day), index - 1, index + 1
            return _canonical_date(year, month, 0), index, index + 1
        if token == "سال" and index + 1 < len(tokens) and _is_digit(tokens[index + 1]):
            return _canonical_date(int(_to_float(tokens[index + 1])), 0, 0), index + 1, index + 1
        return None

    # --- number detection -------------------------------------------------
    def find_numbers(self, tokens: Sequence[str]) -> list[int]:
        """Return the indices forming the first number expression in *tokens*."""
        start = next((i for i, token in enumerate(tokens) if _is_number(token)), None)
        if start is None:
            return []
        indices = [start]
        cursor = start + 1
        while cursor < len(tokens):
            token = tokens[cursor]
            if token == _JOINER and cursor + 1 < len(tokens) and _is_number(tokens[cursor + 1]):
                indices.extend((cursor, cursor + 1))
                cursor += 2
            elif _is_number(token):
                indices.append(cursor)
                cursor += 1
            else:
                break
        return indices

    # --- public API -------------------------------------------------------
    def normalize_dates(self, tokens: Sequence[str]) -> str:
        """Rewrite every date of *tokens* in ``yYmMdD`` form.

        >>> normalize_dates(["اول", "فروردین", "1402"])
        'y1402m1d1'
        """
        tokens = list(tokens)
        pieces: list[str] = []
        cursor = 0
        while cursor < len(tokens):
            found = self.find_date(tokens[cursor:])
            if found is None:
                pieces.append(" ".join(tokens[cursor:]))
                break
            canonical, start, end = found
            pieces.append(" ".join(tokens[cursor : cursor + start]))
            pieces.append(canonical)
            cursor += end + 1
        return " ".join(" ".join(pieces).split())

    def normalize_numbers(self, tokens: Sequence[str]) -> list[str]:
        """Replace number-word runs by their numeric value.

        >>> normalize_numbers(["دو", "هزار", "و", "سی", "تومان"])
        ['2030', 'تومان']
        """
        tokens = list(tokens)
        for index, token in enumerate(tokens):
            if token.endswith("ین") and token[:-2] in NUMBER_WORDS:
                tokens[index] = token[:-2]
        result: list[str] = []
        cursor = 0
        while cursor < len(tokens):
            indices = self.find_numbers(tokens[cursor:])
            if not indices:
                result.extend(tokens[cursor:])
                break
            first = indices[0]
            result.extend(tokens[cursor : cursor + first])
            run = [tokens[cursor + i] for i in indices]
            result.append(_format_number(words_to_number(run)))
            cursor += indices[-1] + 1
        return result

    def normalize(self, text: str) -> str:
        """Apply number and date normalisation to a raw string."""
        tokens = text.split()
        if self.convert_numbers:
            tokens = self.normalize_numbers(tokens)
        if self.convert_dates:
            return self.normalize_dates(tokens)
        return " ".join(tokens)

    def normalize_text_dates(self, text: str) -> str:
        """Like :meth:`normalize` but leaves bare numbers untouched."""
        return DateNormalizer(convert_numbers=False).normalize(text)


def _canonical_date(year: int, month: int, day: int) -> str:
    """Build the ``yYmMdD`` representation used by ``parsivar``.

    >>> _canonical_date(1402, 5, 12)
    'y1402m5d12'
    """
    return f"y{year}m{month}d{day}"


def normalize_numbers(tokens: Iterable[str]) -> list[str]:
    """Function-level shortcut for :meth:`DateNormalizer.normalize_numbers`."""
    return DateNormalizer().normalize_numbers(list(tokens))


def normalize_dates(tokens: Iterable[str]) -> str:
    """Function-level shortcut for :meth:`DateNormalizer.normalize_dates`."""
    return DateNormalizer().normalize_dates(list(tokens))


def normalize_numeric_dates(text: str) -> str:
    """Rewrite numeric dates such as ``1402/05/12`` in canonical form.

    >>> normalize_numeric_dates("تاریخ 1402/05/12 ثبت شد")
    'تاریخ y1402m5d12 ثبت شد'
    """

    def replace(match: re.Match[str]) -> str:
        year, month, day = match.group(1), match.group(2), match.group(3)
        return _canonical_date(int(year), int(month), int(day or 0))

    normalized = text.translate(C.DIGIT_TRANSLATION)
    return _RE_DATE_NUMERIC.sub(replace, normalized)


_ = SCALE  # re-exported through :mod:`farsi_textutils.data`, kept for symmetry
