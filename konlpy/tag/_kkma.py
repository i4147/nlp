# -*- coding: utf-8 -*-
from __future__ import absolute_import

import re
import jpype

from konlpy import jvm, utils
from konlpy.tag._common import validate_phrase_inputs


__all__ = ["Kkma"]


class Kkma:
    def __init__(self, jvmpath=None, max_heap_size=1024):
        if not jpype.isJVMStarted():
            jvm.init_jvm(jvmpath, max_heap_size)

        kkmaJavaPackage = jpype.JPackage("kr.lucypark.kkma")
        KkmaInterfaceJavaClass = kkmaJavaPackage.KkmaInterface
        self.jki = KkmaInterfaceJavaClass()
        self.tagset = utils.read_json("%s/data/tagset/kkma.json" % utils.installpath)

    def nouns(self, phrase):

        nouns = self.jki.extractNoun(phrase)
        if not nouns:
            return []
        return [nouns.get(i).getString() for i in range(nouns.size())]

    def pos(self, phrase, flatten=True, join=False):
        validate_phrase_inputs(phrase)

        sentences = self.jki.morphAnalyzer(phrase)
        morphemes = []
        if not sentences:
            return morphemes

        for i in range(sentences.size()):
            sentence = sentences.get(i)
            for j in range(sentence.size()):
                eojeol = sentence.get(j)
                if flatten:
                    for k in range(eojeol.size()):
                        morpheme = eojeol.get(k)
                        if join:
                            morphemes.append(morpheme.getString() + "/" + morpheme.getTag())
                        else:
                            morphemes.append((morpheme.getString(), morpheme.getTag()))
                else:
                    if join:
                        morphemes.append([
                            eojeol.get(k).getString() + "/" + eojeol.get(k).getTag() for k in range(eojeol.size())
                        ])
                    else:
                        morphemes.append([
                            (eojeol.get(k).getString(), eojeol.get(k).getTag()) for k in range(eojeol.size())
                        ])

        return morphemes

    def morphs(self, phrase):

        return [s for s, t in self.pos(phrase)]

    def sentences(self, phrase):

        sentences = self.jki.morphAnalyzer(phrase)
        if not sentences:
            return []
        return [sentences.get(i).getSentence() for i in range(sentences.size())]
