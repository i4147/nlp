# -*- coding: utf-8 -*-


from __future__ import absolute_import

from collections import deque, namedtuple
import copy
import itertools
import re
from threading import RLock

from cachetools import LFUCache, cached

from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
from camel_tools.utils.charsets import AR_CHARSET, AR_DIAC_CHARSET

from camel_tools.utils.charmap import CharMapper
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.errors import AnalyzerError
from camel_tools.morphology.utils import merge_features
from camel_tools.morphology.utils import simple_ar_to_caphi
from camel_tools.utils.dediac import dediac_ar


_ALL_PUNC = "".join(UNICODE_PUNCT_SYMBOL_CHARSET)

_DIAC_RE = re.compile("[" + re.escape("".join(AR_DIAC_CHARSET)) + "]")
_IS_DIGIT_RE = re.compile("^.*[0-9\u0660-\u0669]+.*$")
_IS_STRICT_DIGIT_RE = re.compile("^[0-9\u0660-\u0669]+$")
_IS_PUNC_RE = re.compile("^[" + re.escape(_ALL_PUNC) + "]+$")
_HAS_PUNC_RE = re.compile("[" + re.escape(_ALL_PUNC) + "]")
_IS_AR_RE = re.compile("^[" + re.escape("".join(AR_CHARSET)) + "]+$")

_NOAN_RE = re.compile("NOAN")

_COPY_FEATS = frozenset(["gloss", "atbtok", "atbseg", "d1tok", "d1seg", "d2tok", "d2seg", "d3tok", "d3seg", "bwtok"])

_UNDEFINED_LEX_FEATS = frozenset(["root", "pattern", "caphi"])

DEFAULT_NORMALIZE_MAP = CharMapper({
    "\u0625": "\u0627",
    "\u0623": "\u0627",
    "\u0622": "\u0627",
    "\u0671": "\u0627",
    "\u0649": "\u064a",
    "\u0629": "\u0647",
    "\u0640": "",
})
""":obj:`~camel_tools.utils.charmap.CharMapper`: The default character map used
for normalization by :obj:`Analyzer`.

Removes the tatweel/kashida character and does the following conversions:

- 'إ' to 'ا'
- 'أ' to 'ا'
- 'آ' to 'ا'
- 'ٱ' to 'ا'
- 'ى' to 'ي'
- 'ة' to 'ه'
"""


_BACKOFF_TYPES = frozenset(["NONE", "NOAN_ALL", "NOAN_PROP", "ADD_ALL", "ADD_PROP"])


class AnalyzedWord(namedtuple("AnalyzedWord", ["word", "analyses"])):
    pass


def _is_digit(word):
    return _IS_DIGIT_RE.match(word) is not None


def _is_strict_digit(word):
    return _IS_STRICT_DIGIT_RE.match(word) is not None


def _is_punc(word):
    return _IS_PUNC_RE.match(word) is not None


def _has_punc(word):
    return _HAS_PUNC_RE.search(word) is not None


def _is_ar(word):
    return _IS_AR_RE.match(word) is not None


def _segments_gen(word, max_prefix=1, max_suffix=1):
    w = len(word)
    for p in range(0, min(max_prefix, w - 1) + 1):
        prefix = word[:p]
        for s in range(max(1, w - p - max_suffix), w - p + 1):
            stem = word[p : p + s]
            suffix = word[p + s :]
            yield (prefix, stem, suffix)


class Analyzer:
    def __init__(self, db, backoff="NONE", norm_map=None, strict_digit=False, cache_size=0):
        if not isinstance(db, MorphologyDB):
            raise AnalyzerError("DB is not an instance of MorphologyDB")
        if not db.flags.analysis:
            raise AnalyzerError("DB does not support analysis")

        self._db = db
        self._backoff = backoff
        self._strict_digit = strict_digit

        if norm_map is None:
            self._norm_map = DEFAULT_NORMALIZE_MAP
        else:
            self._norm_map = norm_map

        if backoff in _BACKOFF_TYPES:
            if backoff == "NONE":
                self._backoff_condition = None
                self._backoff_action = None
            else:
                backoff_toks = backoff.split("_")
                self._backoff_condition = backoff_toks[0]
                self._backoff_action = backoff_toks[1]
        else:
            raise AnalyzerError("Invalid backoff mode {}".format(repr(backoff)))

        if isinstance(cache_size, int):
            if cache_size > 0:
                cache = LFUCache(cache_size)
                self.analyze = cached(cache, lock=RLock())(self.analyze)

        else:
            raise AnalyzerError("Invalid cache size {}".format(repr(cache_size)))

    def _normalize(self, word):
        if self._norm_map is None:
            return word
        return self._norm_map.map_string(word)

    def _combined_analyses(self, word_dediac, prefix_analyses, stem_analyses, suffix_analyses):
        combined = deque()

        for p in itertools.product(prefix_analyses, stem_analyses):
            prefix_cat = p[0][0]
            prefix_feats = p[0][1]
            stem_cat = p[1][0]
            stem_feats = p[1][1]

            if stem_cat in self._db.prefix_stem_compat[prefix_cat]:
                for suffix_cat, suffix_feats in suffix_analyses:
                    if (
                        (stem_cat not in self._db.stem_suffix_compat)
                        or (prefix_cat not in self._db.prefix_suffix_compat)
                        or (suffix_cat not in self._db.stem_suffix_compat[stem_cat])
                        or (suffix_cat not in self._db.prefix_suffix_compat[prefix_cat])
                    ):
                        continue

                    merged = merge_features(self._db, prefix_feats, stem_feats, suffix_feats)
                    merged["stem"] = stem_feats["diac"]
                    merged["stemcat"] = stem_cat

                    merged_dediac = dediac_ar(merged["diac"])
                    if word_dediac.replace("\u0640", "") != merged_dediac:
                        merged["source"] = "spvar"

                    combined.append(merged)

        return combined

    def _combined_backoff_analyses(self, stem, word_dediac, prefix_analyses, stem_analyses, suffix_analyses):
        combined = deque()

        for p in itertools.product(prefix_analyses, stem_analyses):
            prefix_cat = p[0][0]
            prefix_feats = p[0][1]
            stem_cat = p[1][0]
            stem_feats = copy.copy(p[1][1])

            if stem_cat in self._db.prefix_stem_compat[prefix_cat]:
                for suffix_cat, suffix_feats in suffix_analyses:
                    if (suffix_cat not in self._db.stem_suffix_compat[stem_cat]) or (
                        prefix_cat not in self._db.prefix_suffix_compat
                        or suffix_cat not in self._db.prefix_suffix_compat[prefix_cat]
                    ):
                        continue

                    if self._backoff_action == "PROP" and "NOUN_PROP" not in stem_feats["bw"]:
                        continue

                    stem_feats["bw"] = _NOAN_RE.sub(stem, stem_feats["bw"])
                    stem_feats["diac"] = _NOAN_RE.sub(stem, stem_feats["diac"])
                    stem_feats["lex"] = _NOAN_RE.sub(stem, stem_feats["lex"])
                    stem_feats["caphi"] = simple_ar_to_caphi(stem)

                    merged = merge_features(self._db, prefix_feats, stem_feats, suffix_feats)

                    merged["stem"] = stem_feats["diac"]
                    merged["stemcat"] = stem_cat
                    merged["source"] = "backoff"
                    merged["pattern"] = "backoff"
                    merged["gloss"] = stem_feats["gloss"]

                    combined.append(merged)

        return combined

    def analyze(self, word):

        word = word.strip()

        if word == "":
            return []

        analyses = deque()
        word_dediac = dediac_ar(word)
        word_normal = self._normalize(word_dediac)

        if (self._strict_digit and _is_strict_digit(word)) or (not self._strict_digit and _is_digit(word)):
            result = copy.copy(self._db.defaults["digit"])
            result["diac"] = word
            result["stem"] = word
            result["stemgloss"] = word
            result["stemcat"] = None
            result["lex"] = word
            result["bw"] = word + "/NOUN_NUM"
            result["source"] = "digit"

            for feat in _COPY_FEATS:
                if feat in self._db.defines:
                    result[feat] = word

            for feat in _UNDEFINED_LEX_FEATS:
                if feat in self._db.defines:
                    result[feat] = "DIGIT"

            if "catib6" in self._db.defines:
                result["catib6"] = "NOM"
            if "ud" in self._db.defines:
                result["ud"] = "NUM"

            result["pos_logprob"] = -99.0
            result["lex_logprob"] = -99.0
            result["pos_lex_logprob"] = -99.0

            if "form_gen" in self._db.defines and result["gen"] == "-":
                result["gen"] = result["form_gen"]

            if "form_num" in self._db.defines and result["num"] == "-":
                result["num"] = result["form_num"]

            return [result]

        elif _is_punc(word):
            result = copy.copy(self._db.defaults["punc"])
            result["diac"] = word
            result["stem"] = word
            result["stemgloss"] = word
            result["stemcat"] = None
            result["lex"] = word
            result["bw"] = word + "/PUNC"
            result["source"] = "punc"

            for feat in _COPY_FEATS:
                if feat in self._db.defines:
                    result[feat] = word

            for feat in _UNDEFINED_LEX_FEATS:
                if feat in self._db.defines:
                    result[feat] = "PUNC"

            if "catib6" in self._db.defines:
                result["catib6"] = "PNX"
            if "ud" in self._db.defines:
                result["ud"] = "PUNCT"

            result["pos_logprob"] = -99.0
            result["lex_logprob"] = -99.0
            result["pos_lex_logprob"] = -99.0

            if "form_gen" in self._db.defines and result["gen"] == "-":
                result["gen"] = result["form_gen"]

            if "form_num" in self._db.defines and result["num"] == "-":
                result["num"] = result["form_num"]

            return [result]

        elif _has_punc(word):
            pass

        elif not _is_ar(word):
            result = copy.copy(self._db.defaults["latin"])
            result["pos"] = "foreign"
            result["diac"] = word
            result["stem"] = word
            result["stemgloss"] = word
            result["stemcat"] = None
            result["lex"] = word
            result["bw"] = word + "/FOREIGN"
            result["source"] = "foreign"

            for feat in _COPY_FEATS:
                if feat in self._db.defines:
                    result[feat] = word

            for feat in _UNDEFINED_LEX_FEATS:
                if feat in self._db.defines:
                    result[feat] = "FOREIGN"

            if "catib6" in self._db.defines:
                result["catib6"] = "FOREIGN"

            if "ud" in self._db.defines:
                result["ud"] = "X"

            result["pos_logprob"] = -99.0
            result["lex_logprob"] = -99.0
            result["pos_lex_logprob"] = -99.0

            if "form_gen" in self._db.defines and result["gen"] == "-":
                result["gen"] = result["form_gen"]

            if "form_num" in self._db.defines and result["num"] == "-":
                result["num"] = result["form_num"]

            return [result]

        else:
            segments_gen = _segments_gen(word_normal, self._db.max_prefix_size, self._db.max_suffix_size)

            for segmentation in segments_gen:
                prefix = segmentation[0]
                stem = segmentation[1]
                suffix = segmentation[2]

                prefix_analyses = self._db.prefix_hash.get(prefix, None)
                suffix_analyses = self._db.suffix_hash.get(suffix, None)

                if prefix_analyses is None or suffix_analyses is None:
                    continue

                stem_analyses = self._db.stem_hash.get(stem, None)

                if stem_analyses is not None:
                    combined = self._combined_analyses(word_dediac, prefix_analyses, stem_analyses, suffix_analyses)
                    analyses.extend(combined)

        if (self._backoff_condition == "NOAN" and len(analyses) == 0) or (self._backoff_condition == "ADD"):
            segments_gen = _segments_gen(word_normal, self._db.max_prefix_size, self._db.max_suffix_size)

            backoff_cats = self._db.stem_backoffs[self._backoff_action]
            stem_analyses = [(cat, analysis) for cat, analysis in self._db.stem_hash["NOAN"] if cat in backoff_cats]

            for segmentation in segments_gen:
                prefix = segmentation[0]
                stem = segmentation[1]
                suffix = segmentation[2]

                prefix_analyses = self._db.prefix_hash.get(prefix, None)
                suffix_analyses = self._db.suffix_hash.get(suffix, None)

                if prefix_analyses is None or suffix_analyses is None:
                    continue

                combined = self._combined_backoff_analyses(
                    stem, word_dediac, prefix_analyses, stem_analyses, suffix_analyses
                )
                analyses.extend(combined)

        result = list(analyses)

        return result

    def analyze_words(self, words):

        return list(map(lambda w: AnalyzedWord(w, self.analyze(w)), words))

    def all_feats(self):

        return self._db.all_feats()

    def tok_feats(self):
        return self._db.tok_feats()
