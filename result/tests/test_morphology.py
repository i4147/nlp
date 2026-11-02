"""Tests for the morphology, tagger and chunker modules."""

from __future__ import annotations

import pytest
from farsi_textutils.chunk import (
    DEFAULT_PERSIAN_GRAMMAR,
    Chunk,
    ChunkParser,
    GrammarRule,
    Leaf,
    RegexpChunker,
    chunk_tokens,
    to_bracket_string,
)
from farsi_textutils.pos import Lexicon, POSTagger, tag_tokens
from farsi_textutils.stem import AffixRules, Stemmer, StemmerConfig, load_lexicon, load_verb_tense_map

# --------------------------------------------------------------------------- #
# stemming
# --------------------------------------------------------------------------- #


def test_stemmer_default_rules() -> None:
    stemmer = Stemmer()
    assert stemmer.stem("می‌روم") == "روم"
    assert stemmer.stem("کتاب‌ها") == "کتاب"
    assert stemmer.stem("خانه‌ام") == "خانه"
    assert stemmer.stem("دنیا") == "دنیا"


def test_stemmer_with_lexicon() -> None:
    stemmer = Stemmer(nouns=frozenset({"کتاب", "خانه"}), verbs=frozenset({"رفت"}))
    assert stemmer.stem("کتاب‌ها") == "کتاب"
    assert stemmer.stem("می‌روم", pos="N") in {"می‌روم", "روم"}


def test_stemmer_verb_analysis() -> None:
    stemmer = Stemmer(verbs=frozenset({"رو"}), verb_map={"رو": "رفت"})
    assert stemmer.stem("می‌رویم") == "رو&رفت"
    assert stemmer.stem_all(["کتاب‌ها", "کتاب"]) == ["کتاب", "کتاب"]


def test_stemmer_does_not_overstrip() -> None:
    stemmer = Stemmer()
    assert stemmer.stem("سلام") == "سلام"
    assert stemmer.stem("سل") == "سل"


def test_stemmer_configuration_flags() -> None:
    config = StemmerConfig(strip_nouns=False, strip_verbs=False)
    stemmer = Stemmer(config=config)
    assert stemmer.stem("کتاب‌ها") == "کتاب‌ها"


def test_affix_rules_are_frozen() -> None:
    rules = AffixRules()
    assert "می" in rules.prefixes
    with pytest.raises(AttributeError):
        rules.prefixes = ()  # type: ignore[misc]


def test_load_lexicon(tmp_path) -> None:
    path = tmp_path / "verbs.txt"
    path.write_text("رفت\nآمد\n\n", encoding="utf-8")
    assert load_lexicon(path) == frozenset({"رفت", "آمد"})
    assert load_verb_tense_map(path) == {}


# --------------------------------------------------------------------------- #
# part of speech tagging
# --------------------------------------------------------------------------- #


def test_tag_tokens_basic() -> None:
    tagged = tag_tokens(["کتاب", "خوب", "است"])
    assert tagged == [("کتاب", "N_SING"), ("خوب", "ADJ_SIM"), ("است", "V_PRS")]


def test_tag_tokens_lexicon_classes() -> None:
    assert tag_tokens(["من"])[0][1] == "PRO"
    assert tag_tokens(["در"])[0][1] == "P"
    assert tag_tokens(["و"])[0][1] == "CONJ"
    assert tag_tokens(["این"])[0][1] == "DET"
    assert tag_tokens(["امروز"])[0][1] == "ADV"
    assert tag_tokens(["را"])[0][1] == "POSTP"
    assert tag_tokens(["1402"])[0][1] == "NUM"
    assert tag_tokens(["Python"])[0][1] == "FW"
    assert tag_tokens(["،"])[0][1] == "PUNC"


def test_tagger_unknown_and_morphology() -> None:
    tagger = POSTagger()
    assert tagger.tag_one("می‌روم") == "V_PRS"
    assert tagger.tag_one("رفتند") == "V_PAST"
    assert tagger.tag_one("کتاب‌ها") == "N_PLUR"
    assert tagger.tag_one("کتاb".replace("b", "ب")) == "N_SING"


def test_tagger_tag_text_and_parse() -> None:
    tagger = POSTagger()
    assert tagger.tag_text("کتاب خوب است")[0] == ("کتاب", "N_SING")
    assert tagger.parse(["کتاب"]) == [("کتاب", "N_SING")]


def test_lexicon_unknown_marking() -> None:
    lexicon = Lexicon(nouns=frozenset({"قلم"}))
    tagger = POSTagger(lexicon, mark_unknown_as_noun=False)
    assert tagger.tag_one("قلم") == "N_SING"
    assert tagger.tag_one("نامعلوم") == "RES"


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #


def test_chunk_tokens_noun_phrase() -> None:
    nodes = chunk_tokens([("کتاب", "N_SING"), ("خوب", "ADJ_SIM")])
    assert len(nodes) == 1
    assert isinstance(nodes[0], Chunk)
    assert nodes[0].label == "NP"
    assert to_bracket_string(nodes) == "[کتاب/N_SING خوب/ADJ_SIM NP]"


def test_chunk_tokens_prepositional_phrase() -> None:
    nodes = chunk_tokens([("به", "P"), ("خانه", "N_SING")])
    assert to_bracket_string(nodes) == "[[به/P PP] [خانه/N_SING NP] NPP]"


def test_chunk_custom_grammar() -> None:
    grammar = [GrammarRule("XN", "<ADJ_SIM><N_SING>")]
    nodes = chunk_tokens([("خوب", "ADJ_SIM"), ("کتاب", "N_SING")], grammar=grammar)
    assert to_bracket_string(nodes) == "[خوب/ADJ_SIM کتاب/N_SING XN]"


def test_chunk_parser_accepts_text() -> None:
    parser = ChunkParser()
    assert "NP" in parser.chunk_text("کتاب خوب")


def test_regexp_chunker_chunks() -> None:
    chunker = RegexpChunker()
    nodes = chunker.parse([("کتاب", "N_SING")])
    assert to_bracket_string(nodes) == "[کتاب/N_SING NP]"
    # the default grammar folds a subject NP into the verb phrase
    tagged = [("کتاب", "N_SING"), ("است", "V_PRS")]
    nodes = chunker.parse(tagged)
    assert [node.label for node in nodes] == ["VP"]
    assert to_bracket_string(nodes) == "[کتاب/N_SING است/V_PRS VP]"


def test_chunk_grammar_validation() -> None:
    with pytest.raises(ValueError):
        GrammarRule("NP", "")
    assert DEFAULT_PERSIAN_GRAMMAR


def test_leaf_rendering() -> None:
    assert to_bracket_string([Leaf("کتاب", "N_SING")]) == "کتاب/N_SING"
