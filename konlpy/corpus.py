# -*- coding: utf-8 -*-
from __future__ import absolute_import
from konlpy import utils

import os


class CorpusLoader:
    def __init__(self, name=None):
        if not name:
            raise Exception("You need to input the name of the corpus")
        else:
            self.name = name

    def abspath(self, filename=None):
        basedir = "%s/data/corpus/%s" % (utils.installpath, self.name)
        if filename:
            return "%s/%s" % (basedir, filename)
        else:
            return "%s/" % basedir

    def fileids(self):
        return os.listdir(self.abspath())

    def open(self, filename):
        return utils.load_txt(self.abspath(filename))


kolaw = CorpusLoader("kolaw")
kobill = CorpusLoader("kobill")
