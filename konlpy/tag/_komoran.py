# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import os

import jpype

from konlpy import jvm, utils
from konlpy.tag._common import validate_phrase_inputs


__all__ = ["Komoran"]


class Komoran:
    def __init__(self, jvmpath=None, userdic=None, modelpath=None, max_heap_size=1024):
        if not jpype.isJVMStarted():
            jvm.init_jvm(jvmpath, max_heap_size)

        if modelpath:
            self.modelpath = modelpath
        else:
            self.modelpath = os.path.join(utils.installpath, "java", "data", "models")
        self.tagset = utils.read_json("%s/data/tagset/komoran.json" % utils.installpath)

        komoranJavaPackage = jpype.JPackage("kr.co.shineware.nlp.komoran.core")

        try:
            self.jki = komoranJavaPackage.Komoran(self.modelpath)
        except TypeError:
            raise IOError("Cannot access komoran-dic. Please leave an issue at https://github.com/konlpy/konlpy/issues")

        if userdic:
            self.jki.setUserDic(userdic)

    def pos(self, phrase, flatten=True, join=False):
        validate_phrase_inputs(phrase)

        sentences = phrase.split("\n")
        morphemes = []
        if not sentences:
            return morphemes

        for sentence in sentences:
            if not sentence:
                continue
            result = self.jki.analyze(sentence).getTokenList()
            result = [(token.getMorph(), token.getPos()) for token in result]

            if join:
                result = ["{}/{}".format(morph, pos) for morph, pos in result]

            morphemes.append(result)

        if flatten:
            return sum(morphemes, [])
        return morphemes

    def nouns(self, phrase):

        tagged = self.pos(phrase)
        return [s for s, t in tagged if t.startswith("NN")]

    def morphs(self, phrase):

        return [s for s, t in self.pos(phrase)]
