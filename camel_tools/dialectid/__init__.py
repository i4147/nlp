# -*- coding: utf-8 -*-


from camel_tools.dialectid.common import DIDPred, DialectIdError
from camel_tools.dialectid.common import UntrainedModelError
from camel_tools.dialectid.common import InvalidDataSetError
from camel_tools.dialectid.common import PretrainedModelError
from camel_tools.dialectid.model6 import DIDModel6
from camel_tools.dialectid.model26 import DIDModel26

DialectIdentifier = DIDModel26

__all__ = [
    "DIDPred",
    "DialectIdError",
    "UntrainedModelError",
    "InvalidDataSetError",
    "PretrainedModelError",
    "DialectIdentifier",
    "DIDModel26",
    "DIDModel6",
]
