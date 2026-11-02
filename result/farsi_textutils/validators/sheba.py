"""Iranian SHEBA numbers (IBAN, ``IR`` + 24 digits).

Ported from ``persian_tools.bank.sheba``.  The three account-number calculators
of the reference implementation (Parsian, Pasargad and Shahr banks) are plain
functions here instead of lambdas stored inside the data table, so the metadata
stays picklable data.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Final, cast

from ..data.banks import BANKS
from ..digits import to_english_digits
from ..exceptions import InvalidShebaNumber
from ..logging_setup import get_logger
from .bank_card import BankData

__all__ = [
    "ACCOUNT_NUMBER_CALCULATORS",
    "SHEBA_LENGTH",
    "bank_code_from_card",
    "bank_data",
    "generate",
    "normalize",
    "validate",
]

_log = get_logger("validators.sheba")

SHEBA_LENGTH: Final[int] = 26
_COUNTRY_CODE: Final[str] = "IR"


def parsian_account_number(sheba: str) -> dict[str, str]:
    """Parsian (``IRxx 0540...``) account numbers.

    >>> parsian_account_number("IR060540000000000000000001")["normal"][:4]
    '0000'
    """
    tail = sheba[14:]
    formatted = f"0{tail[:2]}-0{tail[2:9]}-{tail[9:]}"
    return {"normal": tail, "formatted": formatted}


def pasargad_account_number(sheba: str) -> dict[str, str]:
    """Pasargad account numbers (leading zeros are not significant)."""
    tail = sheba[7:].lstrip("0")
    tail = tail[: len(tail) - 2]
    return {"normal": tail, "formatted": "-".join([tail[:3], tail[3:6], tail[6:14], tail[14:]])}


def shahr_account_number(sheba: str) -> dict[str, str]:
    """Shahr bank account numbers."""
    tail = sheba[7:].lstrip("0")
    return {"normal": tail, "formatted": tail}


ACCOUNT_NUMBER_CALCULATORS: Final[dict[str, Callable[[str], dict[str, str]]]] = {
    "parsian_account_number_calculator": parsian_account_number,
    "pasargad_account_number_calculator": pasargad_account_number,
    "shahr_account_number_calculator": shahr_account_number,
}
"""Maps the calculator names stored in the bank table to their implementation."""


def normalize(sheba: str) -> str:
    """Uppercase the country code and remove spaces/dashes.

    >>> normalize("ir06 0540 0000 0000 0000 0000 01")
    'IR060540000000000000000001'
    """
    text = to_english_digits(str(sheba)).upper()
    text = re.sub(r"[\s\-.]", "", text)
    if text.startswith(_COUNTRY_CODE):
        return text
    if text.isdigit() and len(text) == 24:
        return _COUNTRY_CODE + text
    return text


def _mod97(number: str) -> int:
    """ISO 7064 MOD-97-10 remainder computed in chunks (never big ints)."""
    remainder = 0
    for char in number:
        remainder = (remainder * 10 + int(char)) % 97
    return remainder


def validate(sheba: str) -> bool:
    """Validate a SHEBA number.

    >>> validate("IR060540000000000000000001")
    False
    >>> validate("IR" + "1" * 24)
    False
    """
    code = normalize(sheba)
    if len(code) != SHEBA_LENGTH or not code[2:].isdigit():
        return False
    rearranged = code[4:] + f"{ord(code[0]) - 55}{ord(code[1]) - 55}" + code[2:4]
    return _mod97(rearranged) == 1


def bank_data(sheba: str) -> BankData | None:
    """Return the metadata (and account number) of the bank holding *sheba*.

    Raises:
        InvalidShebaNumber: When the SHEBA itself is invalid.
    """
    if not validate(sheba):
        raise InvalidShebaNumber(sheba)
    code = normalize(sheba)
    sheba_code = code[4:7]
    for bank in BANKS:
        if sheba_code in cast("list[str]", bank.get("sheba_code", [])):
            data: BankData = cast(
                BankData,
                {key: value for key, value in bank.items() if key != "account_number_calculator"},
            )
            calculator_name = bank.get("account_number_calculator")
            if isinstance(calculator_name, str):
                calculator = ACCOUNT_NUMBER_CALCULATORS.get(calculator_name)
                if calculator is not None:
                    account = calculator(code)
                    data["account_number"] = account["normal"]
                    data["formatted_account_number"] = account["formatted"]
            return data
    return None


def bank_code_from_card(card_number: str) -> str:
    """Return the three digit SHEBA bank code of the bank owning a card.

    >>> bank_code_from_card("6037997512345670")
    '017'
    """
    from .bank_card import bank_data as card_bank_data
    from .bank_card import normalize as normalize_card

    data = card_bank_data(normalize_card(card_number))
    if not data or not data.get("sheba_code"):
        raise InvalidShebaNumber(card_number)
    return str(data["sheba_code"][0])  # type: ignore[index]


def generate(body: str = "0540000000000000000000") -> str:
    """Complete a 24 digit body with its two check digits.

    >>> validate(generate())
    True
    """
    digits = re.sub(r"\D", "", body).ljust(22, "0")[:22]
    for check in range(100):
        candidate = f"{_COUNTRY_CODE}{check:02d}{digits}"
        if validate(candidate):
            return candidate
    raise InvalidShebaNumber(body)  # pragma: no cover - unreachable
