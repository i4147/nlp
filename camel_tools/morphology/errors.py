# -*- coding: utf-8 -*-


class MorphologyError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)


class DatabaseError(MorphologyError):
    def __init__(self, msg):
        MorphologyError.__init__(self, msg)


class InvalidBuiltinDatabaseName(DatabaseError):
    def __init__(self, dbname):
        DatabaseError.__init__(self, "Invalid builtin database with name {}".format(repr(dbname)))


class InvalidDatabaseFlagError(DatabaseError):
    def __init__(self, flag):
        DatabaseError.__init__(self, "Invalid flag value {}".format(repr(flag)))


class DatabaseParseError(DatabaseError):
    def __init__(self, msg):
        DatabaseError.__init__(self, "Error parsing database ({})".format(msg))


class AnalyzerError(MorphologyError):
    def __init__(self, msg):
        MorphologyError.__init__(self, msg)


class GeneratorError(MorphologyError):
    def __init__(self, msg):
        MorphologyError.__init__(self, msg)


class InvalidGeneratorFeature(GeneratorError):
    def __init__(self, feat):
        GeneratorError.__init__(self, "Invalid feature {}".format(repr(feat)))


class InvalidGeneratorFeatureValue(GeneratorError):
    def __init__(self, feat, val):
        GeneratorError.__init__(self, "Invalid value {} for feature {}".format(repr(val), repr(feat)))


class ReinflectorError(MorphologyError):
    def __init__(self, msg):
        MorphologyError.__init__(self, msg)


class InvalidReinflectorFeature(ReinflectorError):
    def __init__(self, feat):
        GeneratorError.__init__(self, "Invalid feature {}".format(repr(feat)))


class InvalidReinflectorFeatureValue(ReinflectorError):
    def __init__(self, feat, val):
        GeneratorError.__init__(self, "Invalid value {} for feature {}".format(repr(val), repr(feat)))
