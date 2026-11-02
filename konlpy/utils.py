# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

import io
import json
import os
import re
import pprint as pp
import sys
from threading import Thread


installpath = os.path.dirname(os.path.realpath(__file__))
replace_set = [
    ("·", "/"),
    ("․", "/"),
    ("ㆍ", "/"),
    ("･", "/"),
    ("～", "~"),
    ("❑", "-"),
    ("‘", "'"),
    ("’", "'"),
    ("“", '"'),
    ("”", '"'),
    ("「", "<"),
    ("」", ">"),
]


if sys.version_info[0] < 3:

    class UnicodePrinter(pp.PrettyPrinter):
        def format(self, object, context, maxlevels, level):
            if isinstance(object, unicode):
                encoding = sys.stdout.encoding or "utf-8"
                return (object.encode(encoding), True, False)
            return pp.PrettyPrinter.format(self, object, context, maxlevels, level)


def concordance(phrase, text, show=False):

    terms = text.split()
    indexes = [i for i, term in enumerate(terms) if phrase in term]
    if show:
        for i in indexes:
            print("%d\t%s" % (i, " ".join(terms[max(0, i - 3) : i + 3])))
    return indexes


if sys.version_info[0] < 3:
    from konlpy import csvutils

    def csvread(f, encoding="utf-8"):
        reader = csvutils.UnicodeReader(f)
        return [row for row in reader]

    def csvwrite(data, f):
        return csvutils.UnicodeWriter(f).writerows(data)


def partition(list_, indices):
    return [list_[i:j] for i, j in zip([0] + indices, indices + [None])]


if sys.version_info[0] < 3:

    def pprint(obj, **kwargs):

        if "stream" in kwargs.keys():
            return UnicodePrinter(stream=kwargs["stream"]).pprint(obj)
        return UnicodePrinter().pprint(obj)

else:
    pprint = pp.pprint


def select(phrase):
    for a, b in replace_set:
        phrase = phrase.replace(a, b)
    return phrase


def char2hex(c):
    return hex(ord(c))


if sys.version_info[0] >= 3:
    unichr = chr


def hex2char(h):
    return unichr(int(h, 16))


def load_txt(filename, encoding="utf-8"):
    return io.open(filename, "r", encoding=encoding)


def read_txt(filename, encoding="utf-8"):
    with io.open(filename, "r", encoding=encoding) as f:
        return f.read()


def read_json(filename, encoding="utf-8"):
    with io.open(filename, "r", encoding=encoding) as f:
        return json.load(f)


def delete_links(string):

    return re.sub(r"http\S+", "", string)


def delete_mentions(string):

    return re.sub(r"@\S+", "", string)


class PropagatingThread(Thread):
    def run(self):
        self.exception = None
        try:
            if hasattr(self, "_Thread__target"):
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exception = e

    def join(self):
        super(PropagatingThread, self).join()
        if self.exception:
            raise self.exception
        return self.ret
