# -*- coding: utf-8 -*-


from __future__ import absolute_import

from collections import deque

import re

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.generator import Generator
from camel_tools.morphology.errors import ReinflectorError
from camel_tools.morphology.errors import InvalidReinflectorFeature
from camel_tools.morphology.errors import InvalidReinflectorFeatureValue
from camel_tools.utils.dediac import dediac_ar


_CLITIC_FEATS = frozenset(["enc0", "prc0", "prc1", "prc2", "prc3"])
_IGNORED_FEATS = frozenset([
    "diac",
    "lex",
    "bw",
    "gloss",
    "source",
    "stem",
    "stemcat",
    "lmm",
    "dediac",
    "caphi",
    "catib6",
    "ud",
    "d3seg",
    "atbseg",
    "d2seg",
    "d1seg",
    "d1tok",
    "d2tok",
    "atbtok",
    "d3tok",
    "bwtok",
    "root",
    "pattern",
    "freq",
    "pos_logprob",
    "lex_logprob",
    "pos_lex_logprob",
    "stemgloss",
])
_SPECIFIED_FEATS = frozenset(["form_gen", "form_num"])
_CLITIC_IGNORED_FEATS = frozenset(["stt", "cas", "mod"])
_FILTER_FEATS = frozenset(["pos", "lex"])
_ANY_FEATS = frozenset(["per", "gen", "num", "cas", "stt", "vox", "mod", "asp"])

_LEMMA_SPLIT_RE = re.compile("-|_")


class Reinflector(object):
    def __init__(self, db):
        if not isinstance(db, MorphologyDB):
            raise ReinflectorError("DB is not an instance of MorphologyDB")
        if not db.flags.generation:
            raise ReinflectorError("DB does not support reinflection")

        self._db = db

        self._analyzer = Analyzer(db)
        self._generator = Generator(db)

    def reinflect(self, word, feats):

        analyses = self._analyzer.analyze(word)

        if not analyses or len(analyses) == 0:
            return []

        for feat in feats:
            if feat not in self._db.defines:
                raise InvalidReinflectorFeature(feat)
            elif self._db.defines[feat] is not None:
                if feat in _ANY_FEATS and feats[feat] == "ANY":
                    continue
                elif feats[feat] not in self._db.defines[feat]:
                    raise InvalidReinflectorFeatureValue(feat, feats[feat])

        has_clitics = False
        for feat in _CLITIC_FEATS:
            if feat in feats:
                has_clitics = True
                break

        results = deque()

        for analysis in analyses:
            if dediac_ar(analysis["diac"]) != dediac_ar(word):
                continue

            if "pos" in feats and feats["pos"] != analysis["pos"]:
                continue

            lemma = _LEMMA_SPLIT_RE.split(analysis["lex"])[0]

            if "lex" in feats and feats["lex"] != lemma:
                continue

            is_valid = True
            generate_feats = {}

            for feat in analysis.keys():
                if feat in _IGNORED_FEATS:
                    continue
                elif feat in _SPECIFIED_FEATS and feat not in feats:
                    continue
                elif has_clitics and feat in _CLITIC_IGNORED_FEATS:
                    continue
                else:
                    if feat in feats:
                        if feats[feat] == "ANY":
                            continue
                        elif analysis[feat] != "na":
                            generate_feats[feat] = feats[feat]
                        else:
                            is_valid = False
                            break
                    elif analysis[feat] != "na":
                        generate_feats[feat] = analysis[feat]

            if is_valid:
                generated = self._generator.generate(lemma, generate_feats)
                if generated is not None:
                    results.extend(generated)

        results = [dict(y) for y in set(tuple(x.items()) for x in results)]

        return list(results)

    def all_feats(self):

        return self._db.all_feats()

    def tok_feats(self):
        return self._db.tok_feats()
