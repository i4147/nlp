# -*- coding: utf-8 -*-


import json

from cachetools import LFUCache, cached
import editdistance

from camel_tools.utils.dediac import dediac_ar
from camel_tools.disambig.common import Disambiguator, DisambiguatedWord
from camel_tools.disambig.common import ScoredAnalysis
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.utils import strip_lex
from camel_tools.data import CATALOGUE


def _calima_msa_r13_analyzer():
    db = MorphologyDB.builtin_db("calima-msa-r13", "a")
    analyzer = Analyzer(db, "NOAN_PROP")
    return analyzer


def _calima_egy_r13_analyzer():
    db = MorphologyDB.builtin_db("calima-egy-r13", "a")
    analyzer = Analyzer(db, "NOAN_PROP")
    return analyzer


_MLE_ANALYZER_MAP = {"calima-msa-r13": _calima_msa_r13_analyzer, "calima-egy-r13": _calima_egy_r13_analyzer}


def _get_pos_lex_logprob(analysis):
    logprob = analysis.get("pos_lex_logprob", -99.0)
    if logprob is None:
        return -99
    return logprob


_EQ_FEATS = frozenset([
    "asp",
    "cas",
    "enc0",
    "gen",
    "mod",
    "num",
    "per",
    "pos",
    "prc0",
    "prc1",
    "prc2",
    "prc3",
    "form_num",
    "form_gen",
    "enc1",
    "enc2",
    "stt",
    "vox",
])
_DISTANCE_FEATS = frozenset(["diac", "lex", "bw"])


def _score_analysis(analysis, reference):
    score = 0.0

    for feat in _EQ_FEATS:
        if analysis.get(feat, "") == reference.get(feat, ""):
            score += 1

    for feat in _DISTANCE_FEATS:
        feat_r = reference.get(feat, "")
        feat_a = analysis.get(feat, "")
        distance = editdistance.eval(feat_r, feat_a)
        score += max(0.0, (len(feat_r) - distance) / len(feat_r))

    return score


class MLEDisambiguator(Disambiguator):
    def __init__(self, analyzer, mle_path=None, top=1, cache_size=100000):
        if not isinstance(analyzer, Analyzer):
            raise ValueError("Invalid analyzer instance.")
        if not isinstance(top, int):
            raise ValueError("Invalid value for top.")
        if not isinstance(cache_size, int):
            raise ValueError("Invalid value for cache_size.")

        if mle_path is not None:
            with open(mle_path, "r", encoding="utf-8") as mle_fp:
                self._mle = json.load(mle_fp)

                for analysis in self._mle.values():
                    analysis["lex"] = strip_lex(analysis["lex"])
        else:
            self._mle = None

        self._analyzer = analyzer

        if top < 1:
            top = 1
        self._top = top

        if cache_size <= 0:
            cache_size = 0
            self._cache = None
            self._score_fn = self._scored_analyses
        else:
            self._cache = LFUCache(cache_size)
            self._score_fn = self._scored_analyses_cached

    @staticmethod
    def pretrained(model_name=None, analyzer=None, top=1, cache_size=100000):

        if model_name is None:
            model_name = CATALOGUE.components["DisambigMLE"].default

        model_info = CATALOGUE.components["DisambigMLE"].datasets[model_name]
        mle_path = model_info.path / "model.json"

        if analyzer is None:
            analyzer = _MLE_ANALYZER_MAP[model_info.name]()

        return MLEDisambiguator(analyzer, str(mle_path), top, cache_size)

    def _scored_analyses(self, word_dd):
        if self._mle is not None and word_dd in self._mle:
            mle_analysis = self._mle[word_dd]
            analyses = self._analyzer.analyze(word_dd)

            if len(analyses) == 0:
                return []

            scored = [(_score_analysis(a, mle_analysis), a) for a in analyses]
            scored.sort(key=lambda s: (-s[0], len(s[1]["bw"]), s[1]["diac"]))

            max_score = max([s[0] for s in scored])

            if max_score == 0:
                max_score = 1

            scored_analyses = [
                ScoredAnalysis(
                    s / max_score,
                    a,
                    a["diac"],
                    a.get("pos_lex_logprob", -99),
                    a.get("lex_logprob", -99),
                )
                for s, a in scored
            ]

            return scored_analyses[0 : self._top]

        else:
            analyses = self._analyzer.analyze(word_dd)

            if len(analyses) == 0:
                return []

            probabilities = [10 ** _get_pos_lex_logprob(a) for a in analyses]
            max_prob = max(probabilities)

            scored_analyses = [
                ScoredAnalysis(
                    p / max_prob,
                    a,
                    a["diac"],
                    a.get("pos_lex_logprob", -99),
                    a.get("lex_logprob", -99),
                )
                for a, p in zip(analyses, probabilities)
            ]

            scored_analyses.sort()

            return scored_analyses[0 : self._top]

    def _scored_analyses_cached(self, word_dd):
        return self._cache.get(word_dd, self._scored_analyses(word_dd))

    def _disambiguate_word(self, word):
        word_dd = dediac_ar(word)
        scored_analyses = self._score_fn(word_dd)

        return DisambiguatedWord(word, scored_analyses)

    def disambiguate_word(self, sentence, word_ndx):

        return self._disambiguate_word(sentence[word_ndx])

    def disambiguate(self, sentence):

        return [self._disambiguate_word(w) for w in sentence]

    def all_feats(self):

        return self._analyzer.all_feats()

    def tok_feats(self):
        return self._analyzer.tok_feats()
