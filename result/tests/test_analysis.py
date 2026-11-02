"""Tests for the statistical modules: similarity, keywords, spelling, stop words."""

from __future__ import annotations

import pytest
from farsi_textutils.exceptions import ModelNotTrainedError
from farsi_textutils.keywords import (
    TfidfKeywords,
    WordExtractor,
    discover_words,
    extract_keywords,
)
from farsi_textutils.similarity import (
    BM25,
    DocumentScore,
    TfidfVectorizer,
    WordContextVectorizer,
    cosine_similarity,
    jaccard_similarity,
)
from farsi_textutils.spell import NgramModel, SpellChecker, edit_candidates
from farsi_textutils.stopwords import (
    STOP_WORDS,
    content_words,
    is_stop_word,
    normalise_frequencies,
    remove_stop_words,
    word_frequencies,
)

DOCUMENTS = ["کتاب خوب است", "کتاب بد است", "ماشین بزرگ است"]


# --------------------------------------------------------------------------- #
# vector space similarity
# --------------------------------------------------------------------------- #


def test_cosine_and_jaccard() -> None:
    assert cosine_similarity({"a": 1.0}, {"a": 1.0}) == pytest.approx(1.0)
    assert cosine_similarity({"a": 1.0}, {"b": 1.0}) == 0.0
    assert cosine_similarity({}, {}) == 0.0
    assert jaccard_similarity({"a", "b"}, {"a"}) == pytest.approx(1 / 2)
    assert jaccard_similarity(set(), set()) == 1.0


def test_tfidf_vectorizer() -> None:
    vectorizer = TfidfVectorizer().fit(DOCUMENTS)
    vector = vectorizer.transform("کتاب خوب")
    assert set(vector) == {"کتاب", "خوب"}
    assert vector["کتاب"] > 0
    assert vectorizer.similarity("کتاب خوب", "کتاب بد") < vectorizer.similarity("کتاب خوب", "کتاب خوب")
    assert vectorizer.transform("") == {}


def test_bm25_ranking() -> None:
    bm25 = BM25().fit(DOCUMENTS)
    ranked = bm25.rank("کتاب")
    assert [item.index for item in ranked[:2]] == [0, 1]
    assert ranked[0].score == pytest.approx(ranked[1].score)
    assert ranked[-1].score == 0.0
    assert isinstance(ranked[0], DocumentScore)
    assert bm25.score("ماشین", 2) > 0
    assert bm25.idf["کتاب"] > 0


def test_word_context_vectorizer() -> None:
    vectorizer = WordContextVectorizer(window=2, min_count=1).fit(DOCUMENTS)
    assert "کتاب" in vectorizer.vectors
    assert vectorizer.vector("ناموجود") == {}
    assert vectorizer.similarity("کتاب", "کتاب") == pytest.approx(1.0)
    assert [word for word, _ in vectorizer.most_similar("کتاب", top_k=2)] == ["خوب", "بد"]
    assert vectorizer.most_similar("ناموجود") == []
    assert vectorizer.fit(DOCUMENTS) is vectorizer


# --------------------------------------------------------------------------- #
# keywords
# --------------------------------------------------------------------------- #


def test_extract_keywords() -> None:
    scored = extract_keywords(DOCUMENTS, top_k=2)
    assert [item.word for item in scored][:1] == ["کتاب"]
    assert scored[0].document_frequency == 2


def test_tfidf_keywords() -> None:
    keywords = TfidfKeywords().fit(DOCUMENTS)
    assert keywords.rank(top_k=1)[0].word == "کتاب"


def test_word_extractor_requires_fit() -> None:
    extractor = WordExtractor()
    with pytest.raises(ModelNotTrainedError):
        extractor.extract()


def test_word_extractor_fit() -> None:
    extractor = WordExtractor(min_frequency=2, workers=1).fit(
        ["کتابخانه بزرگ است", "کتابخانه کوچک است", "کتابخانه نو است"]
    )
    assert "کتابخانه" in [item.word for item in extractor.extract()]
    assert extractor.score("کتابخانه") is not None
    assert extractor.score("ناموجود") is None
    assert "کتابخانه" in extractor.extract_words(top_k=3)


def test_discover_words() -> None:
    found = discover_words(["سلام دنیا", "سلام تهران", "سلام شیراز"], min_frequency=1, workers=1, top_k=1)
    assert [item.word for item in found] == ["سلام"]
    assert all(item.score >= 0 for item in found)


# --------------------------------------------------------------------------- #
# spelling
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def model() -> NgramModel:
    return NgramModel.train(["کتاب خوب است", "کتاب بد است", "او کتاب خواند"], workers=2)


def test_ngram_model_train(model: NgramModel) -> None:
    assert model.is_known("کتاب")
    assert not model.is_known("کتب")
    assert model.word_logprob("کتاب") < 0
    assert model.bigram_logprob("کتاب", "خوب") < 0
    assert model.sentence_logprob(["کتاب", "خوب"]) < 0


def test_ngram_model_persistence(model, tmp_path) -> None:
    path = model.save(tmp_path / "model.json")
    restored = NgramModel.load(path)
    assert restored.is_known("کتاب")
    assert restored.total_tokens == model.total_tokens


def test_ngram_model_empty_raises() -> None:
    with pytest.raises(ModelNotTrainedError):
        NgramModel().word_logprob("کتاب")


def test_ngram_model_validation() -> None:
    with pytest.raises(ValueError):
        NgramModel.train(["متن"], order=0)


def test_spell_checker(model: NgramModel) -> None:
    checker = SpellChecker(model)
    assert checker.is_correct("کتاب")
    suggestions = checker.suggestions("کتب")
    assert suggestions
    assert suggestions[0].word == "کتاب"
    assert checker.correct("کتب") == "کتاب"
    assert checker.correct_text("کتاب کتب است") == "کتاب کتاب است"
    assert checker.normalise("كتاب") == "کتاب"


def test_spell_checker_without_context(model: NgramModel) -> None:
    checker = SpellChecker(model, use_context=False, max_suggestions=1)
    assert len(checker.suggestions("کتب")) <= 1


def test_edit_candidates() -> None:
    candidates = list(edit_candidates("کتاب", include_split=False))
    operations = {operation for _, operation in candidates}
    assert operations == {"deletion", "insertion", "substitution", "transposition"}
    assert ("کتب", "deletion") in candidates
    assert any(operation == "split" for _, operation in edit_candidates("کتاب"))
    assert len(candidates) > 50


# --------------------------------------------------------------------------- #
# stop words
# --------------------------------------------------------------------------- #


def test_stop_word_presets() -> None:
    assert "و" in STOP_WORDS
    assert is_stop_word("و")
    assert not is_stop_word("کتاب")
    assert remove_stop_words(["کتاب", "و", "من", "خوب"]) == ["کتاب", "خوب"]
    assert remove_stop_words(["کتاب"], extra=["کتاب"]) == []


def test_content_words_and_frequencies() -> None:
    assert content_words("کتاب و من") == ["کتاب"]
    frequencies = word_frequencies(["کتاب خوب", "کتاب"], remove_stops=False)
    assert frequencies["کتاب"] == 2
    assert normalise_frequencies({"الف": 2, "ب": 2}) == {"الف": 0.5, "ب": 0.5}
