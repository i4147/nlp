# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

import re

import jpype

from konlpy import jvm, utils
from konlpy.tag._common import validate_phrase_inputs


__all__ = ["Hannanum"]


tag_re = "(.+?\\/\\w+)\\+?"


def parse(result, flatten=False, join=False):
    def parse_opt(opt, join=False):
        if join:
            return [u for u in re.findall(tag_re, opt.strip())]
        else:
            return [tuple(u.rsplit("/", 1)) for u in re.findall(tag_re, opt.strip())]

    if not result:
        return []

    elems = result.strip().splitlines()
    index = [i for i, e in enumerate(elems) if not e]
    parts = utils.partition(elems, index)

    if flatten:
        return sum([parse_opt(opt, join=join) for part in parts for opt in list(filter(None, part))[1:]], [])
    else:
        return [[parse_opt(opt, join=join) for opt in list(filter(None, part))[1:]] for part in parts]


class Hannanum:
    def __init__(self, jvmpath=None, max_heap_size=1024):
        if not jpype.isJVMStarted():
            jvm.init_jvm(jvmpath, max_heap_size)

        jhannanumJavaPackage = jpype.JPackage("kr.lucypark.jhannanum.comm")
        HannanumInterfaceJavaClass = jhannanumJavaPackage.HannanumInterface
        self.jhi = HannanumInterfaceJavaClass()
        self.tagset = utils.read_json("%s/data/tagset/hannanum.json" % utils.installpath)

    def analyze(self, phrase):

        result = self.jhi.morphAnalyzer(phrase)
        return parse(result)

    def pos(self, phrase, ntags=9, flatten=True, join=False):
        validate_phrase_inputs(phrase)

        if ntags == 9:
            result = self.jhi.simplePos09(phrase)
        elif ntags == 22:
            result = self.jhi.simplePos22(phrase)
        else:
            raise Exception("ntags in [9, 22]")
        return parse(result, flatten=flatten, join=join)

    def nouns(self, phrase):

        tagged = self.pos(phrase)
        return [s for s, t in tagged if t.startswith("N")]

    def morphs(self, phrase):

        return [s for s, t in self.pos(phrase)]
