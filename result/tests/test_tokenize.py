"""Tests for :mod:`farsi_textutils.tokenize`."""

from __future__ import annotations

import pytest
from farsi_textutils.tokenize import (
    LTokenizer,
    MaxScoreTokenizer,
    SentenceSplitter,
    Tokenizer,
    TokenizerConfig,
    paragraph_tokenize,
    sentence_tokenize,
    word_tokenize,
)


def test_word_tokenize_keeps_punctuation_separate() -> None:
    assert word_tokenize("سلام، دنیا!") == ["سلام", "،", "دنیا", "!"]


def test_word_tokenize_keeps_half_space_words() -> None:
    assert word_tokenize("می‌روم به کتاب‌ها") == ["می‌روم", "به", "کتاب‌ها"]


def test_word_tokenize_can_split_half_spaces() -> None:
    tokens = word_tokenize("می‌روم", split_half_space=True)
    assert tokens == ["می", "روم"]


def test_word_tokenize_digits_and_latin() -> None:
    assert word_tokenize("سال 1402 بود") == ["سال", "1402", "بود"]
    assert word_tokenize("Python 3.12") == ["Python", "3.12"]


def test_word_tokenize_empty_and_whitespace() -> None:
    assert word_tokenize("") == []
    assert word_tokenize("   \n\t ") == []


def test_word_tokenize_without_punctuation() -> None:
    assert word_tokenize("سلام، دنیا!", keep_punctuation=False) == ["سلام", "دنیا"]


def test_word_tokenize_quotes_and_brackets() -> None:
    assert word_tokenize("«کتاب» (خوب)") == ["«", "کتاب", "»", "(", "خوب", ")"]


def test_sentence_tokenize() -> None:
    text = "من به خانه رفتم. او آمد؟ بله!"
    assert sentence_tokenize(text) == ["من به خانه رفتم.", "او آمد؟", "بله!"]


def test_sentence_tokenize_persian_question_and_semicolon() -> None:
    assert sentence_tokenize("رفتم؛ خسته بودم. خوبم") == ["رفتم؛", "خسته بودم.", "خوبم"]


def test_sentence_tokenize_min_length() -> None:
    assert sentence_tokenize("بله. خیر. این جمله بلند است.", min_length=5) == [
        "این جمله بلند است.",
    ]


def test_sentence_tokenize_handles_half_space_and_numbers() -> None:
    text = "تاریخ 1402/05/12 بود. عدد 3.5 کوچک است."
    assert sentence_tokenize(text) == ["تاریخ 1402/05/12 بود.", "عدد 3.5 کوچک است."]


def test_sentence_splitter_abbreviations() -> None:
    splitter = SentenceSplitter(abbreviations=["دکتر", "مهندس"])
    assert splitter.split("دکتر احمدی آمد. او خوب است.") == [
        "دکتر احمدی آمد.",
        "او خوب است.",
    ]


def test_sentence_splitter_keeps_multiple_delimiters() -> None:
    splitter = SentenceSplitter()
    assert splitter.split("چرا؟! چون که...") == ["چرا؟!", "چون که..."]


def test_sentence_splitter_empty() -> None:
    assert SentenceSplitter().split("") == []


def test_paragraph_tokenize() -> None:
    text = "پاراگراف یک\n\nپاراگراف دو"
    assert paragraph_tokenize(text) == ["پاراگراف یک", "پاراگراف دو"]
    # a single newline is a soft wrap, only blank lines start a new paragraph
    assert paragraph_tokenize("یک\nسه") == ["یک\nسه"]
    assert paragraph_tokenize("") == []


def test_tokenizer_config() -> None:
    tokenizer = Tokenizer(TokenizerConfig(split_half_space=True, keep_punctuation=False))
    assert tokenizer.word("می‌روم، خانه") == ["می", "روم", "خانه"]
    assert tokenizer("می‌روم") == ["می", "روم"]
    assert tokenizer.sentence("یک. دو.") == ["یک.", "دو."]
    assert tokenizer.paragraph("یک\n\nدو") == ["یک", "دو"]
    assert tokenizer.tokenize_words("یک") == ["یک"]
    assert tokenizer.tokenize_sentences("یک.") == ["یک."]


def test_ltokenizer_splits_by_scores() -> None:
    tokenizer = LTokenizer(scores={"한국": 1.0, "어": 0.5}, default_score=0.1)
    assert tokenizer.tokenize("한국어") == ["한국", "어"]


def test_ltokenizer_persian_particles() -> None:
    tokenizer = LTokenizer(scores={"کتاب": 2.0, "ها": 1.0, "خانه": 2.0})
    assert tokenizer.tokenize("کتابها") == ["کتاب", "ها"]


def test_ltokenizer_tolerance_and_flatten() -> None:
    tokenizer = LTokenizer(scores={"کتاب": 1.0, "کتابه": 2.0})
    assert tokenizer.tokenize("کتابها") == ["کتابه", "ا"]
    assert tokenizer.tokenize("کتابها", tolerance=1.0) == ["کتاب", "ها"]
    assert tokenizer("کتابها کتاب", flatten=False) == [["کتابه", "ا"], ["کتاب"]]
    assert tokenizer("کتابها کتاب") == ["کتابه", "ا", "کتاب"]


def test_ltokenizer_keeps_unknown_tokens_intact() -> None:
    assert LTokenizer(scores={"کتاب": 2.0}).tokenize("دنیا") == ["دنیا"]


def test_max_score_tokenizer_prefers_highest_score() -> None:
    tokenizer = MaxScoreTokenizer(scores={"کتاب": 3.0, "کتابها": 5.0, "ها": 1.0})
    assert tokenizer.tokenize("کتابها") == ["کتابها"]
    assert tokenizer("سلام دنیا", flatten=False) == [["سلام"], ["دنیا"]]


def test_max_score_tokenizer_keeps_unknown_runs_together() -> None:
    tokenizer = MaxScoreTokenizer(scores={"کتاب": 3.0, "ها": 1.0})
    assert tokenizer.tokenize("کتابها دنیا") == ["کتاب", "ها", "دنیا"]
    assert tokenizer("دنیا", flatten=False) == [["دنیا"]]


@pytest.mark.parametrize("text", ["یک دو سه", "یک. دو", "یک\n\nدو"])
def test_tokenizers_are_idempotent_on_whitespace(text: str) -> None:
    tokens = word_tokenize(text)
    assert all(token.strip() for token in tokens)
    assert word_tokenize(" ".join(tokens)) == tokens
