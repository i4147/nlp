# -*- coding: utf-8 -*-


import collections


__all__ = [
    "DIDPred",
    "DialectIdError",
    "UntrainedModelError",
    "InvalidDataSetError",
    "PretrainedModelError",
]


class DIDPred(collections.namedtuple("DIDPred", ["top", "scores"])):
    pass


class DialectIdError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)


class UntrainedModelError(DialectIdError):
    def __init__(self, msg):
        DialectIdError.__init__(self, msg)


class InvalidDataSetError(DialectIdError, ValueError):
    def __init__(self, dataset):
        msg = 'Invalid data set name {}. Valid names are "DEV" and "TEST"'.format(repr(dataset))
        DialectIdError.__init__(self, msg)


class PretrainedModelError(DialectIdError):
    def __init__(self, msg):
        DialectIdError.__init__(self, msg)
