# -*- coding: utf-8 -*-


from __future__ import absolute_import

import six


def isunicode(obj):

    return isinstance(obj, six.text_type)


def force_unicode(s, encoding="utf-8"):

    if s is None or isinstance(s, six.text_type):
        return s

    return s.decode(encoding)


def force_encoding(s, encoding="utf-8"):

    if s is None or isinstance(s, six.binary_type):
        return s

    return s.encode(encoding)
