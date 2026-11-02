# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys

try:
    from MeCab import Tagger
except ImportError:
    pass

from konlpy import utils
from konlpy.tag._common import validate_phrase_inputs


__all__ = ["Mecab"]


attrs = ["tags", "semantic", "has_jongsung", "read", "type", "first_pos", "last_pos", "original", "indexed"]


def parse(result, allattrs=False, join=False):
    def split(elem, join=False):
        if not elem:
            return ("", "SY")
        splited = elem.split("\t", 1)
        if len(splited) != 2:
            return ("", "SY")
        s, t = splited
        if join:
            return s + "/" + t.split(",", 1)[0]
        else:
            return (s, t.split(",", 1)[0])

    return [split(elem, join=join) for elem in result.splitlines()[:-1]]


class Mecab:
    def __init__(self, dicpath="/usr/local/lib/mecab/dic/mecab-ko-dic"):
        self.dicpath = dicpath
        try:
            self.tagger = Tagger("-d %s" % dicpath)
            self.tagset = utils.read_json("%s/data/tagset/mecab.json" % utils.installpath)
        except RuntimeError:
            raise Exception(
                'The MeCab dictionary does not exist at "%s". Is the dictionary correctly installed?\nYou can also try entering the dictionary path when initializing the Mecab class: "Mecab(\'/some/dic/path\')"'
                % dicpath
            )
        except NameError:
            raise Exception("Install MeCab in order to use it: http://konlpy.org/en/latest/install/")

    def __setstate__(self, state):

        self.__init__(dicpath=state["dicpath"])

    def __getstate__(self):

        return {"dicpath": self.dicpath}

    def pos(self, phrase, flatten=True, join=False):
        validate_phrase_inputs(phrase)

        if sys.version_info[0] < 3:
            phrase = phrase.encode("utf-8")
            if flatten:
                result = self.tagger.parse(phrase).decode("utf-8")
                return parse(result, join=join)
            else:
                return [parse(self.tagger.parse(eojeol).decode("utf-8"), join=join) for eojeol in phrase.split()]
        else:
            if flatten:
                result = self.tagger.parse(phrase)
                return parse(result, join=join)
            else:
                return [parse(self.tagger.parse(eojeol), join=join) for eojeol in phrase.split()]

    def morphs(self, phrase):

        return [s for s, t in self.pos(phrase)]

    def nouns(self, phrase):

        tagged = self.pos(phrase)
        return [s for s, t in tagged if t.startswith("N")]
