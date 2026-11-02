# -*- coding: utf-8 -*-
from __future__ import absolute_import

import jpype

from konlpy import jvm, utils
from konlpy.tag._common import validate_phrase_inputs


def Twitter(jvmpath=None):

    from warnings import warn

    warn('"Twitter" has changed to "Okt" since KoNLPy v0.4.5.')
    return Okt(jvmpath)


class Okt:
    def __init__(self, jvmpath=None, max_heap_size=1024):
        if not jpype.isJVMStarted():
            jvm.init_jvm(jvmpath, max_heap_size)

        oktJavaPackage = jpype.JPackage("kr.lucypark.okt")
        OktInterfaceJavaClass = oktJavaPackage.OktInterface
        self.jki = OktInterfaceJavaClass()
        self.tagset = utils.read_json("%s/data/tagset/twitter.json" % utils.installpath)

    def pos(self, phrase, norm=False, stem=False, join=False):
        validate_phrase_inputs(phrase)

        tokens = self.jki.tokenize(phrase, jpype.java.lang.Boolean(norm), jpype.java.lang.Boolean(stem)).toArray()
        if join:
            return [t for t in tokens]
        else:
            return [tuple(t.rsplit("/", 1)) for t in tokens]

    def nouns(self, phrase):

        tagged = self.pos(phrase)
        return [s for s, t in tagged if t == "Noun"]

    def morphs(self, phrase, norm=False, stem=False):

        return [s for s, t in self.pos(phrase, norm=norm, stem=stem)]

    def phrases(self, phrase):

        return [p for p in self.jki.phrases(phrase).toArray()]

    def normalize(self, phrase):
        text = self.jki.normalize(phrase)
        return text
