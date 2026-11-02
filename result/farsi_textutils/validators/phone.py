"""Iranian mobile phone numbers.

Ported from ``persian_tools.phone_number``.  The operator lookup walks the
prefix table of :mod:`farsi_textutils.data.operators` from the shortest to the
longest prefix, exactly like the reference implementation.
"""

from __future__ import annotations

import random
import re
from typing import Final, NotRequired, TypedDict

from ..data.operators import MOBILE_PREFIXES
from ..digits import to_english_digits
from ..exceptions import InvalidPhoneNumber, InvalidValueError
from ..logging_setup import get_logger

__all__ = [
    "VALID_TOKENS",
    "MobileExtraction",
    "OperatorInfo",
    "extract",
    "generate",
    "normalize",
    "operator_data",
    "validate",
]

_log = get_logger("validators.phone")

VALID_TOKENS: Final[tuple[str, ...]] = ("+98", "98", "0098", "0")
_PREFIX_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?:\+98|98|0098|0)")
_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"[\s\-().\u2010-\u2015]")


class OperatorInfo(TypedDict):
    """Metadata of a mobile numbering prefix."""

    province: list[str]
    base: str
    type: list[str]
    operator: str


class MobileExtraction(TypedDict):
    """A mobile number found in a text."""

    index: int
    base: str
    pure: str
    is_valid: bool
    operator_data: NotRequired[OperatorInfo]


def _suffix(phone_number: str) -> str:
    """Strip the country/trunk prefix and return the significant digits."""
    cleaned = _SEPARATORS.sub("", to_english_digits(phone_number))
    return _PREFIX_PATTERN.sub("", cleaned)


def _lookup(suffix: str) -> OperatorInfo | None:
    """Find the operator whose prefix matches *suffix*, shortest first."""
    for length in range(3, 6):
        candidate = suffix[:length]
        if candidate in MOBILE_PREFIXES:
            return dict(MOBILE_PREFIXES[candidate])  # type: ignore[return-value]
    return None


def validate(phone_number: str) -> bool:
    """True when *phone_number* belongs to the Iranian mobile numbering plan.

    >>> validate("09123456789"), validate("00123456789")
    (True, False)
    """
    suffix = _suffix(phone_number)
    return len(suffix) == 10 and bool(suffix.isdigit()) and _lookup(suffix) is not None


def normalize(phone_number: str, token: str = "0") -> str:
    """Rewrite *phone_number* with the requested prefix token.

    Args:
        phone_number: Input number in any accepted format.
        token: One of :data:`VALID_TOKENS` (``0``, ``+98``, ``98``, ``0098``).

    Raises:
        InvalidValueError: When *token* is not supported.
        InvalidPhoneNumber: When the number itself is invalid.

    >>> normalize("+989123456789")
    '09123456789'
    >>> normalize("09123456789", token="+98")
    '+989123456789'
    """
    if token not in VALID_TOKENS:
        raise InvalidValueError(f"unsupported token {token!r}; expected one of {VALID_TOKENS}")
    if not validate(phone_number):
        raise InvalidPhoneNumber(phone_number)
    return f"{token}{_suffix(phone_number)}"


def operator_data(phone_number: str) -> OperatorInfo:
    """Return operator, province and SIM type of *phone_number*.

    Raises:
        InvalidPhoneNumber: When the number is not a valid mobile number.

    >>> operator_data("09123456789")["operator"]
    'همراه اول'
    """
    suffix = _suffix(phone_number)
    info = _lookup(suffix)
    if info is None:
        raise InvalidPhoneNumber(phone_number)
    return info


def extract(text: str, *, filter_invalid: bool = True) -> list[MobileExtraction]:
    """Find mobile numbers inside *text*.

    >>> [item["pure"] for item in extract("با 09123456789 تماس بگیرید")]
    ['09123456789']
    """
    pattern = re.compile(r"(?:\+98|0098|98|0)?9\d{9}")
    results: list[MobileExtraction] = []
    for index, match in enumerate(pattern.finditer(to_english_digits(text)), start=1):
        raw = match.group()
        pure = normalize(raw)
        item: MobileExtraction = {
            "index": index,
            "base": raw,
            "pure": pure,
            "is_valid": validate(pure),
        }
        if item["is_valid"]:
            item["operator_data"] = operator_data(pure)
        results.append(item)
    return [item for item in results if item["is_valid"]] if filter_invalid else results


def generate(rng: random.Random | None = None, *, prefix: str = "912") -> str:
    """Generate a random (syntactically valid) mobile number.

    >>> validate(generate())
    True
    """
    generator = rng or random
    return f"0{prefix}{generator.randint(0, 9_999_999):07d}"
