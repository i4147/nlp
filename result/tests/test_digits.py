"""Tests for :mod:`farsi_textutils.digits`."""

from __future__ import annotations

import pytest
from farsi_textutils.digits import (
    Language,
    add_ordinal_suffix,
    add_separator,
    convert_digits,
    digits_to_number,
    number_to_words,
    remove_ordinal_suffix,
    remove_separator,
    to_arabic_digits,
    to_english_digits,
    to_persian_digits,
    words_to_number,
)
from farsi_textutils.exceptions import InvalidValueError


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("0123456789", "0123456789"),
        ("۰۱۲۳۴۵۶۷۸۹", "0123456789"),
        ("٠١٢٣٤٥٦٧٨٩", "0123456789"),
        ("سلام ۱۴۰۲", "سلام 1402"),
        (1402, "1402"),
        (12.5, "12.5"),
    ],
)
def test_to_english_digits(source: str | int | float, expected: str) -> None:
    assert to_english_digits(source) == expected


def test_to_persian_and_arabic_digits() -> None:
    assert to_persian_digits("1402") == "۱۴۰۲"
    assert to_arabic_digits("1402") == "١٤٠٢"
    assert convert_digits("1402", Language.PERSIAN) == "۱۴۰۲"
    assert convert_digits("۱۴۰۲", Language.ENGLISH) == "1402"
    assert convert_digits("۱۴۰۲", Language.ARABIC) == "١٤٠٢"
    assert convert_digits(42.5) == "۴۲.۵"


def test_digits_to_number() -> None:
    assert digits_to_number("۱۴۰۲") == 1402
    assert digits_to_number("3.5") == 3.5
    assert digits_to_number(7) == 7


def test_add_separator() -> None:
    assert add_separator(1234567) == "1,234,567"
    assert add_separator("۱۲۳۴۵۶۷") == "۱,۲۳۴,۵۶۷"
    assert add_separator(1234567, separator="٬") == "1٬234٬567"
    assert add_separator(-1234567) == "-1,234,567"
    assert add_separator(1234.5678) == "1,234.5678"


def test_remove_separator() -> None:
    assert remove_separator("1,234,567") == "1234567"
    assert remove_separator("1٬234") == "1234"
    assert remove_separator("۱,۲۳۴") == "۱۲۳۴"


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (0, "صفر"),
        (7, "هفت"),
        (13, "سیزده"),
        (20, "بیست"),
        (99, "نود و نه"),
        (100, "صد"),
        (2030, "دو هزار و سی"),
        (1_000_000, "یک میلیون"),
    ],
)
def test_number_to_words(number: int, expected: str) -> None:
    assert number_to_words(number) == expected


def test_number_to_words_ordinal_and_invalid() -> None:
    assert number_to_words(3, ordinal=True) == "سوم"
    assert number_to_words(1, ordinal=True) == "یکم"
    with pytest.raises(InvalidValueError):
        number_to_words("دو")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ("هفت", 7),
        ("سی و نه", 39),
        ("دو هزار و سی", 2030),
        ("یک میلیون و پانصد", 1_000_500),
    ],
)
def test_words_to_number(words: str, expected: int) -> None:
    assert words_to_number(words) == expected


def test_words_to_number_options() -> None:
    assert words_to_number("۱۲۳") == 123
    assert words_to_number(None) is None
    assert words_to_number("هفت", digits=Language.PERSIAN) == "۷"
    assert words_to_number("دو هزار", separator=True) == "2,000"
    with pytest.raises(InvalidValueError):
        words_to_number("کتاب")


def test_ordinal_suffix() -> None:
    assert add_ordinal_suffix("یک") == "یکم"
    assert add_ordinal_suffix("سه") == "سوم"
    assert add_ordinal_suffix("33") == "سی و سوم"
    assert remove_ordinal_suffix("سوم") == "سه"
    assert remove_ordinal_suffix("یکم") == "یک"
    assert remove_ordinal_suffix("کتاب") == "کتاب"
