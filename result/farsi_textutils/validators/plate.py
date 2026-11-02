"""Iranian licence plates (پلاک خودرو و موتورسیکلت).

Ported from ``persian_tools.plate``: parsing of the two plate layouts, province
and category lookup, validity checks and rendering of the official template.
"""

from __future__ import annotations

import re
from typing import Final, TypedDict

from ..data.plate import CATEGORIES, PROVINCES, TYPE_CAR, TYPE_MOTORCYCLE
from ..digits import to_english_digits
from ..exceptions import InvalidPlate
from ..logging_setup import get_logger

__all__ = [
    "CAR_DIGITS",
    "MOTORCYCLE_DIGITS",
    "PlateInfo",
    "get_info",
    "is_valid",
    "normalize",
    "render",
]

_log = get_logger("validators.plate")

CAR_DIGITS: Final[int] = 7
MOTORCYCLE_DIGITS: Final[int] = 8
_NON_DIGIT: Final[re.Pattern[str]] = re.compile(r"\D")


class PlateInfo(TypedDict):
    """Decoded licence plate."""

    type: str
    template: str
    province: str | None
    category: str | None


def normalize(plate: str) -> dict[str, str | None]:
    """Split a plate into its letter part and its digits.

    >>> normalize("۱۲ ب ۳۴۵ - ۶۷")
    {'char': 'ب', 'numbers': '1234567'}
    """
    text = to_english_digits(str(plate))
    letters = "".join(
        char for char in text if not char.isdigit() and not char.isspace() and char not in "-_.|"
    )
    return {"char": letters or None, "numbers": _NON_DIGIT.sub("", text)}


def _car_info(char: str | None, numbers: str) -> PlateInfo:
    """Decode a car plate (``12ب345-67`` layout)."""
    province_code = int(numbers[5:7])
    template = numbers[0:2] + str(char) + numbers[2:5] + "ایران" + str(province_code)
    return {
        "type": TYPE_CAR,
        "template": template,
        "province": PROVINCES.get(TYPE_CAR, {}).get(province_code),
        "category": CATEGORIES.get(str(char)) if char else None,
    }


def _motorcycle_info(numbers: str) -> PlateInfo:
    """Decode a motorcycle plate (``123-45678`` layout)."""
    province_code = int(numbers[0:3])
    return {
        "type": TYPE_MOTORCYCLE,
        "template": f"{province_code}-{numbers[3:]}",
        "province": PROVINCES.get(TYPE_MOTORCYCLE, {}).get(province_code),
        "category": None,
    }


def get_info(plate: str) -> PlateInfo:
    """Decode a plate, dispatching on the number of digits.

    Raises:
        InvalidPlate: When the plate has neither 7 nor 8 digits.

    >>> get_info("12ب345-67")["province"]
    'اصفهان'
    """
    parsed = normalize(plate)
    numbers = str(parsed["numbers"])
    if len(numbers) == CAR_DIGITS:
        return _car_info(parsed["char"], numbers)  # type: ignore[arg-type]
    if len(numbers) == MOTORCYCLE_DIGITS:
        return _motorcycle_info(numbers)
    raise InvalidPlate(plate, reason=f"expected 7 or 8 digits, got {len(numbers)}")


def is_valid(plate: str) -> bool:
    """Check that a plate exists (known province and category).

    >>> is_valid("12ب345-67"), is_valid("12x345-67")
    (True, False)
    """
    parsed = normalize(plate)
    numbers = str(parsed["numbers"])
    if not numbers or set(numbers[:-1]) == {"0"}:
        return False
    if len(numbers) not in (CAR_DIGITS, MOTORCYCLE_DIGITS):
        return False
    try:
        info = get_info(plate)
    except InvalidPlate:
        return False
    if info["type"] == TYPE_CAR and info["category"] is None:
        return False
    return info["province"] is not None


def render(plate: str) -> str:
    """Render the official plate template.

    >>> render("12ب345-67")
    '12ب345ایران67'
    """
    return get_info(plate)["template"]
