# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import sys


if sys.version_info[0] >= 3:
    basestring = str


def validate_phrase_inputs(phrase):
    msg = "phrase input should be string, not %s" % type(phrase)
    assert isinstance(phrase, basestring), msg
