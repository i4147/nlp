"""Iranian legal-entity identification number (شناسه ملی حقوقی).

Ported from ``persian_tools.legal_id``: an eleven digit number whose last digit
is a weighted checksum with a rotating coefficient table.
"""

from __future__ import annotations

import random
import re
from typing import Final

from ..digits import to_english_digits
from ..logging_setup import get_logger

__all__ = ["COEFFICIENTS", "LEGAL_ID_LENGTH", "extract", "generate", "normalize", "validate"]

_log = get_logger("validators.legal_id")

LEGAL_ID_LENGTH: Final[int] = 11
COEFFICIENTS: Final[tuple[int, ...]] = (29, 27, 23, 19, 17)


def normalize(legal_id: str) -> str:
    """Strip separators, convert digits and left-pad to eleven digits.

    >>> normalize("۱۰۳۲۰۲۴۵۷۸۹")
    '10320245789'
    """
    cleaned = re.sub(r"[\s\-_.]", "", to_english_digits(str(legal_id)))
    return cleaned.zfill(LEGAL_ID_LENGTH)


def _checksum(body: str) -> int:
    """Compute the check digit of a ten digit body."""
    constant = int(body[-1]) + 2
    total = sum((int(digit) + constant) * COEFFICIENTS[index % 5] for index, digit in enumerate(body))
    remainder = total % 11
    return 0 if remainder == 10 else remainder


def validate(legal_id: str) -> bool:
    """Validate a legal-entity identification number.

    >>> first = generate()
    >>> validate(first)
    True
    >>> validate(first[:-1] + str((int(first[-1]) + 1) % 10))
    False
    """
    code = normalize(legal_id)
    if not code.isdigit() or len(code) != LEGAL_ID_LENGTH:
        return False
    return _checksum(code[:10]) == int(code[-1])


def generate(rng: random.Random | None = None) -> str:
    """Generate a random valid legal ID.

    >>> len(generate())
    11
    """
    generator = rng or random
    body = f"{generator.randint(10**8, 10**10 - 1):010d}"
    return body + str(_checksum(body))


def extract(text: str) -> list[str]:
    """Find every legal ID looking number inside *text*.

    >>> extract("شناسه 10320245789 است")
    ['10320245789']
    """
    pattern = re.compile(r"(?<!\d)([۰-۹٠-٩\d]{11})(?!\d)")
    return [normalize(match) for match in pattern.findall(to_english_digits(text))]
