"""Iranian national identification number (کد ملی).

Checksum, random generation, place-of-registration lookup and extraction from
free text.  Ported from ``persian_tools.national_id``.
"""

from __future__ import annotations

import random
import re
from typing import Final, TypedDict

from ..data.places import PLACES
from ..digits import to_english_digits
from ..exceptions import InvalidNationalId
from ..logging_setup import get_logger

__all__ = [
    "NATIONAL_ID_LENGTH",
    "NationalIdPlace",
    "extract",
    "find_place",
    "generate",
    "normalize",
    "validate",
]

_log = get_logger("validators.national_id")

NATIONAL_ID_LENGTH: Final[int] = 10

_PLACE_INDEX: Final[dict[str, dict[str, str]]] = {}
for _place in PLACES:
    for _code in str(_place["code"]).split("-"):
        _PLACE_INDEX.setdefault(_code, dict(_place))


class NationalIdPlace(TypedDict):
    """Registration place of a national ID."""

    code: list[str]
    city: str
    province: str


def normalize(national_id: str) -> str:
    """Convert digits to ASCII, drop separators and left-pad to ten digits.

    >>> normalize("۰۴۹۹۳۷۱۲۳۴")
    '0499371234'
    """
    cleaned = re.sub(r"[\s\-_.]", "", to_english_digits(str(national_id)))
    return cleaned.zfill(NATIONAL_ID_LENGTH)


def validate(national_id: str) -> bool:
    """Validate the national ID checksum.

    >>> validate("0499370899")
    True
    >>> validate("0499370890")
    False
    """
    code = normalize(national_id)
    if not code.isdigit() or len(code) != NATIONAL_ID_LENGTH:
        return False
    if int(code[3:-1]) == 0:
        return False
    checksum = sum(int(code[index]) * (10 - index) for index in range(9))
    remainder = checksum % 11
    expected = remainder if remainder < 2 else 11 - remainder
    return int(code[-1]) == expected


def generate(rng: random.Random | None = None) -> str:
    """Generate a random valid national ID.

    >>> validate(generate())
    True
    """
    generator = rng or random
    body = f"{generator.randint(10**6, 10**9 - 1):09d}"
    checksum = sum(int(body[index]) * (10 - index) for index in range(9))
    remainder = checksum % 11
    return body + str(remainder if remainder < 2 else 11 - remainder)


def find_place(national_id: str) -> NationalIdPlace | None:
    """Return the city/province that issued the ID.

    Raises:
        InvalidNationalId: When the number itself is invalid.

    >>> place = find_place("0499370899")
    >>> place["province"] if place else None
    'تهران'
    """
    if not validate(national_id):
        raise InvalidNationalId(national_id)
    code = normalize(national_id)[:3]
    place = _PLACE_INDEX.get(code)
    if place is None:
        return None
    return {
        "code": str(place["code"]).split("-"),
        "city": str(place["city"]),
        "province": str(place["province"]),
    }


def extract(text: str) -> list[str]:
    """Find every national ID looking number inside *text*.

    >>> extract("کد ملی من 0499370899 است")
    ['0499370899']
    """
    pattern = re.compile(r"(?<!\d)([۰-۹٠-٩\d]{10})(?!\d)")
    return [normalize(match) for match in pattern.findall(to_english_digits(text))]
