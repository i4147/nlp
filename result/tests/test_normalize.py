"""Tests for :mod:`farsi_textutils.normalize`."""

from __future__ import annotations

import pytest
from farsi_textutils.exceptions import ConfigurationError, InvalidValueError
from farsi_textutils.normalize import (
    Normalizer,
    NormalizerConfig,
    SpaceCorrector,
    check_standard_persian_text,
    clean_text,
    collapse_whitespace,
    dediacritize,
    fix_half_spaces,
    normalize,
    remove_extra_spaces,
    standardize,
    tidy_half_spaces,
    unify_letters,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("كتاب", "کتاب"),  # Arabic kaf
        ("ميدان", "میدان"),  # Arabic yeh
        ("ﻧﺎﻡ", "نام"),  # presentation forms (upstream table bug: ﻧ -> ذ)
        ("ﻣﯽﺷﻮﺩ", "میشود"),
        ("ﺍﻣﺮﻭﺯ", "امروز"),
        ("ة", "ه"),
        ("أحمد", "احمد"),
    ],
)
def test_unify_letters(source: str, expected: str) -> None:
    assert unify_letters(source) == expected


def test_unify_letters_keeps_digits() -> None:
    """Digits are converted explicitly, not as a side effect of unification."""
    assert unify_letters("۱۴۰۲") == "۱۴۰۲"


def test_dediacritize() -> None:
    assert dediacritize("كِتَابْ") == "كتاب"
    assert dediacritize("مُدَرِّس") == "مدرس"


def test_whitespace_helpers() -> None:
    assert remove_extra_spaces("  یک   دو \n سه ") == "یک دو سه"
    assert collapse_whitespace("یک\n\n\nدو") == "یک\nدو"
    assert collapse_whitespace("یک \n دو", keep_newlines=False) == "یک دو"


def test_clean_text_spaces_punctuation_and_protects_numbers() -> None:
    assert clean_text("او گفت(سلام)!") == "او گفت (سلام)!"
    assert clean_text("تاریخ 1402/05/12 است") == "تاریخ 1402/05/12 است"
    assert clean_text("قیمت ۳.۵ دلار") == "قیمت ۳.۵ دلار"
    assert clean_text("الف\n\nب", new_line_elimination=True) == "الف ب"


def test_clean_text_drops_non_persian_characters() -> None:
    assert clean_text("سلام 😀 دنیا") == "سلام دنیا"
    assert clean_text("سلام\u200bدنیا") in {"سلامدنیا", "سلام دنیا"}


def test_fix_half_spaces() -> None:
    assert fix_half_spaces("می روم") == "می\u200cروم"
    assert fix_half_spaces("کتاب ها") == "کتاب\u200cها"
    assert fix_half_spaces("نشده است") == "نشده است"
    assert tidy_half_spaces("کتاب \u200c های") == "کتاب\u200cهای"


def test_fix_half_spaces_does_not_glue_copulas() -> None:
    """``بود`` is a verb, not a suffix (regression test)."""
    assert fix_half_spaces("کتاب 1402 بود") == "کتاب 1402 بود"


def test_space_corrector_merges_known_words() -> None:
    corrector = SpaceCorrector({"می\u200cروم": 5, "کتاب\u200cها": 7, "خانه": 3})
    assert corrector.correct("می روم کتاب ها") == "می\u200cروم کتاب\u200cها"
    assert corrector.correct("خانه") == "خانه"


def test_space_corrector_requires_normaliser_option() -> None:
    with pytest.raises(ConfigurationError):
        Normalizer(NormalizerConfig(statistical_space_correction=True))


def test_normalizer_with_space_corrector() -> None:
    corrector = SpaceCorrector({"می\u200cروم": 10, "کتاب\u200cها": 10})
    normalizer = Normalizer(
        NormalizerConfig(statistical_space_correction=True, fix_space=False),
        space_corrector=corrector,
    )
    assert normalizer.normalize("می روم کتاب ها") == "می\u200cروم کتاب\u200cها"


def test_normalize_end_to_end() -> None:
    assert normalize("كتاب‌هاي  من ، خوب است") == "کتاب‌های من، خوب است"
    assert normalize("سلام   دنيا!") == "سلام دنیا!"
    assert normalize("مي روم", fix_space=True) == "می\u200cروم"
    assert (
        normalize(
            "",
        )
        == ""
    )


def test_normalize_options() -> None:
    assert normalize("۱۴۰۲", digits_to_ascii=True) == "1402"
    assert normalize("۱۴۰۲") == "۱۴۰۲"
    with pytest.raises(ConfigurationError):
        normalize("متن", unknown_option=True)  # type: ignore[call-arg]


def test_normalize_dates_option() -> None:
    normalizer = Normalizer(NormalizerConfig(date_normalization=True))
    assert normalizer.normalize("تاریخ 12 مرداد 1402 بود") == "تاریخ y1402m5d12 بود"


def test_check_standard_persian_text() -> None:
    assert check_standard_persian_text("سلام دنیا") is True
    assert check_standard_persian_text("سلام 😀") == "😀"
    assert check_standard_persian_text("hello") == "h"
    assert check_standard_persian_text("hello", includes_latin=True) is True


def test_standardize_pipeline() -> None:
    assert standardize("كتابهاي  من", replace_ascii_digits=False) == "کتابهای من"
    assert standardize("123") == "۱۲۳"
    assert standardize(None) is None
    assert standardize("خط\n\n\nجدید", remove_double_newlines=True) == "خط\nجدید"


def test_standardize_check_persian_raises() -> None:
    with pytest.raises(InvalidValueError):
        standardize("سلام 😀", check_persian=True)
