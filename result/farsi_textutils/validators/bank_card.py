"""Iranian bank card numbers (16 digits, Luhn checksum).

Ported from ``persian_tools.bank.card_number``.  The bank lookup uses the
``card_prefix`` table of :mod:`farsi_textutils.data.banks`.
"""

from __future__ import annotations

import re
from typing import Final, NotRequired, TypedDict, cast

from ..data.banks import BANKS
from ..digits import to_english_digits
from ..exceptions import InvalidCardNumber
from ..logging_setup import get_logger

__all__ = [
    "CARD_LENGTH",
    "BankData",
    "CardExtraction",
    "bank_data",
    "extract",
    "generate_luhn",
    "normalize",
    "validate",
]

_log = get_logger("validators.bank_card")

CARD_LENGTH: Final[int] = 16
_SEPARATORS: Final[str] = "-_.* "


class BankData(TypedDict, total=False):
    """Metadata of the bank that issued a card or SHEBA number."""

    nickname: str
    name: str
    persian_name: str
    card_prefix: list[str]
    sheba_code: list[str]
    account_number: str
    formatted_account_number: str


class CardExtraction(TypedDict):
    """One card number found in a text."""

    index: int
    base: str
    pure: str
    is_valid: bool
    bank_data: NotRequired[BankData | None]


_CARD_PREFIX_INDEX: Final[dict[str, dict[str, object]]] = {
    prefix: dict(bank) for bank in BANKS for prefix in cast("list[str]", bank.get("card_prefix", []))
}


def normalize(card_number: str) -> str:
    """Convert digits and remove the usual separators.

    >>> normalize("۶۰۳۷-۹۹۷۵-۱۲۳۴-۵۶۷۸")
    '6037997512345678'
    """
    text = to_english_digits(str(card_number))
    for separator in _SEPARATORS:
        text = text.replace(separator, "")
    return text


def validate(card_number: str) -> bool:
    """Validate a card number with the Luhn algorithm.

    >>> validate("6037997512345678")
    False
    >>> validate("6037997512345670")
    True
    """
    code = normalize(card_number)
    if not code.isdigit() or len(code) != CARD_LENGTH:
        return False
    if int(code[1:10]) == 0 or int(code[10:]) == 0:
        return False
    total = 0
    for index, char in enumerate(code):
        value = int(char) * (2 if index % 2 == 0 else 1)
        total += value - 9 if value > 9 else value
    return total % 10 == 0


def bank_data(card_number: str) -> BankData | None:
    """Return the metadata of the issuing bank.

    Raises:
        InvalidCardNumber: When the card number itself is invalid.

    >>> bank_data("6037997512345670")["persian_name"]
    'بانک ملی ایران'
    """
    if not validate(card_number):
        raise InvalidCardNumber(card_number)
    code = normalize(card_number)
    bank = _CARD_PREFIX_INDEX.get(code[:6])
    if bank is None:
        return None
    return cast(BankData, {key: value for key, value in bank.items() if key != "account_number_calculator"})


def extract(
    text: str,
    *,
    check_validation: bool = True,
    detect_bank_name: bool = False,
    filter_invalid: bool = True,
) -> list[CardExtraction]:
    """Find every card number inside *text*.

    Args:
        text: Free text (Persian or ASCII digits).
        check_validation: Include the ``is_valid`` flag.
        detect_bank_name: Resolve the bank of valid cards.
        filter_invalid: Drop invalid candidates from the result.

    >>> [item["pure"] for item in extract("کارت 6037-9975-1234-5670")]
    ['6037997512345670']
    """
    pattern = re.compile(r"[۰-۹٠-٩\d][۰-۹٠-٩\d\-_.* ]{14,22}[۰-۹٠-٩\d]")
    results: list[CardExtraction] = []
    for index, match in enumerate(pattern.finditer(to_english_digits(text)), start=1):
        pure = normalize(match.group())
        if len(pure) != CARD_LENGTH:
            continue
        item: CardExtraction = {
            "index": index,
            "base": match.group(),
            "pure": pure,
            "is_valid": validate(pure) if check_validation else True,
        }
        if detect_bank_name and item["is_valid"]:
            item["bank_data"] = bank_data(pure)
        results.append(item)
    if filter_invalid and check_validation:
        results = [item for item in results if item["is_valid"]]
    return results


def generate_luhn(prefix: str = "603799", rng_body: str | None = None) -> str:
    """Complete a 15 digit body with its Luhn check digit.

    >>> validate(generate_luhn("603799", "751234567"))
    True
    """
    body = (prefix + (rng_body or "751234567"))[:15].ljust(15, "0")
    for candidate in range(10):
        if validate(body + str(candidate)):
            return body + str(candidate)
    raise InvalidCardNumber(body)  # pragma: no cover - unreachable
