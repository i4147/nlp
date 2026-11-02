# -*- coding: utf-8 -*-


import re
from collections import deque
from camel_tools.utils.dediac import dediac_ar


_REMOVE_PLUSES = re.compile(r"(_\+|\+_)+")


def _default_dediac(tok):
    return dediac_ar(tok)


def _bwtok_dediac(tok):
    return _REMOVE_PLUSES.sub(r"\g<1>", dediac_ar(tok).strip("+_"))


_DIAC_TYPE = {
    "atbtok": _default_dediac,
    "atbseg": _default_dediac,
    "bwtok": _bwtok_dediac,
    "d1tok": _default_dediac,
    "d1seg": _default_dediac,
    "d2tok": _default_dediac,
    "d2seg": _default_dediac,
    "d3tok": _default_dediac,
    "d3seg": _default_dediac,
}


class MorphologicalTokenizer(object):
    def __init__(self, disambiguator, scheme, split=False, diac=False):
        self._disambiguator = disambiguator
        self._scheme = scheme
        self._split = split
        self._diacf = lambda w: w if diac else _DIAC_TYPE[self._scheme](w)

    def tokenize(self, words):

        disambig_words = self._disambiguator.disambiguate(words)
        result = deque()

        for disambig_word in disambig_words:
            scored_analyses = disambig_word.analyses
            if len(scored_analyses) > 0:
                analysis = scored_analyses[0].analysis
                tok = analysis.get(self._scheme, None)

                if tok is None or tok == "NOAN":
                    tok = disambig_word.word
                    result.append(self._diacf(tok))
                elif self._split:
                    tok = self._diacf(tok)
                    result.extend(tok.split("_"))
                else:
                    result.append(self._diacf(tok))

            else:
                result.append(disambig_word.word)

        return list(result)
