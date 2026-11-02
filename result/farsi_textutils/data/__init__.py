"""Static reference data ported from the source packages.

Every table here is plain, immutable Python data so that importing
:mod:`farsi_textutils.data` never pulls third-party dependencies.
"""

from __future__ import annotations

from .banks import BANK_RECORD_KEYS, BANKS
from .names import FAMILY_NAMES, FEMALE_NAMES, MALE_NAMES
from .numbers import (
    JOINERS,
    MAGNITUDE,
    NUMBER_TEXT,
    PREFIXES,
    SCALE,
    TEN,
    TYPO_LIST,
    UNITS,
)
from .operators import MOBILE_PREFIXES, OPERATOR_NAMES
from .orthography import ORTHOGRAPHY_MAP
from .places import PLACES
from .plate import CATEGORIES, PROVINCES, TYPE_CAR, TYPE_MOTORCYCLE
from .stopwords import STOP_WORDS

__all__ = [
    "BANKS",
    "BANK_RECORD_KEYS",
    "CATEGORIES",
    "FAMILY_NAMES",
    "FEMALE_NAMES",
    "JOINERS",
    "MAGNITUDE",
    "MALE_NAMES",
    "MOBILE_PREFIXES",
    "NUMBER_TEXT",
    "OPERATOR_NAMES",
    "ORTHOGRAPHY_MAP",
    "PLACES",
    "PREFIXES",
    "PROVINCES",
    "SCALE",
    "STOP_WORDS",
    "TEN",
    "TYPE_CAR",
    "TYPE_MOTORCYCLE",
    "TYPO_LIST",
    "UNITS",
]
