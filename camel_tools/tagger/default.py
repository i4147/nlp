# -*- coding: utf-8 -*-


from collections import OrderedDict

from camel_tools.tagger.common import Tagger
from camel_tools.disambig.common import Disambiguator


def _tag_passthrough(feat, word):
    if len(word.analyses) == 0:
        return word.word

    feat_value = word.analyses[0].analysis.get(feat, None)

    if feat_value is None or feat_value == "NOAN":
        return word.word

    return feat_value


def _tag_none(feat, word):
    if len(word.analyses) == 0:
        return None

    return word.analyses[0].analysis.get(feat, None)


def _tag_lex(feat, word):
    if len(word.analyses) == 0:
        return "{}_0".format(word)

    return word.analyses[0].analysis.get("lex", "{}_0".format(word))


_FEAT_ACTIONS = {
    "diac": _tag_passthrough,
    "bw": _tag_none,
    "lex": _tag_lex,
    "gloss": _tag_none,
    "pos": _tag_none,
    "asp": _tag_none,
    "cas": _tag_none,
    "mod": _tag_none,
    "num": _tag_none,
    "gen": _tag_none,
    "form_num": _tag_none,
    "form_gen": _tag_none,
    "stt": _tag_none,
    "vox": _tag_none,
    "per": _tag_none,
    "enc0": _tag_none,
    "enc1": _tag_none,
    "enc2": _tag_none,
    "prc0": _tag_none,
    "prc1": _tag_none,
    "prc2": _tag_none,
    "prc3": _tag_none,
    "atbtok": _tag_passthrough,
    "atbseg": _tag_passthrough,
    "bwtok": _tag_passthrough,
    "d1tok": _tag_passthrough,
    "d1seg": _tag_passthrough,
    "d2tok": _tag_passthrough,
    "d2seg": _tag_passthrough,
    "d3tok": _tag_passthrough,
    "d3seg": _tag_passthrough,
    "catib6": _tag_none,
    "ud": _tag_none,
    "caphi": _tag_none,
}


class DefaultTaggerError(Exception):
    pass


class InvalidDefaultTaggerDisambiguator(DefaultTaggerError, ValueError):
    def __str__(self):
        return "Invalid disambiguator."


class InvalidDefaultTaggerFeature(DefaultTaggerError, ValueError):
    def __init__(self, feature):
        self._feature = feature

    def __str__(self):
        return "Invalid feature {}".format(repr(self._feature))


class DefaultTagger(Tagger):
    def __init__(self, disambiguator, feature):
        if not isinstance(disambiguator, Disambiguator):
            raise InvalidDefaultTaggerDisambiguator()
        elif not feature in _FEAT_ACTIONS:
            raise InvalidDefaultTaggerFeature(feature)

        self._disambiguator = disambiguator
        self._feature = feature

    def _tag_disambiguated_word(self, word):
        return _FEAT_ACTIONS[self._feature](self._feature, word)

    def tag(self, sentence):

        disambig_words = self._disambiguator.disambiguate(sentence)

        return list(map(self._tag_disambiguated_word, disambig_words))

    @staticmethod
    def feature_list():

        return _FEAT_ACTIONS.keys()
