"""Utility bill identifiers (قبض آب، برق، گاز، تلفن ...).

Ported from ``persian_tools.bill``: a bill is identified by a 13 digit
``bill_id`` plus a ``payment_id``; together they form the 26 digit barcode used
at payment terminals.  Both parts carry their own checksum.
"""

from __future__ import annotations

import re
from typing import Final, TypedDict

from ..digits import to_english_digits
from ..exceptions import MissingArgumentError
from ..logging_setup import get_logger

__all__ = [
    "BILL_TYPES",
    "CURRENCY_RIAL",
    "CURRENCY_TOMAN",
    "BillDetail",
    "checksum",
    "extract_barcodes",
    "get_detail",
    "validate",
]

_log = get_logger("validators.bill")

BILL_TYPES: Final[dict[int, str]] = {
    1: "آب",
    2: "برق",
    3: "گاز",
    4: "تلفن ثابت",
    5: "تلفن همراه",
    6: "عوارض شهرداری",
    8: "سازمان مالیات",
    9: "جرایم راهنمایی و رانندگی",
}
"""Bill type digits (the tenth digit of the bill identifier)."""

CURRENCY_TOMAN: Final[int] = 100
CURRENCY_RIAL: Final[int] = 1000


class BillDetail(TypedDict):
    """Everything that can be derived from a bill identifier."""

    amount: int | None
    type: str | None
    barcode: str | None
    bill_id: int | None
    payment_id: int | None
    is_valid: bool
    is_valid_bill_id: bool
    is_valid_payment_id: bool


def checksum(number: str) -> int:
    """Weighted mod-11 checksum used by bill identifiers.

    Digits are weighted from the right with the repeating sequence
    ``2, 3, 4, 5, 6, 7, 2, 3, ...``.

    >>> checksum("1234567")
    4
    """
    total = 0
    weight = 2
    for char in reversed(number):
        if weight == 8:
            weight = 2
        total += int(char) * weight
        weight += 1
    remainder = total % 11
    return 0 if remainder < 2 else 11 - remainder


def _valid_payment_id(bill_id: str, payment_id: str) -> bool:
    """Validate the two check digits embedded in a payment identifier."""
    if len(payment_id) < 6:
        return False
    control_one, control_two = payment_id[-2], payment_id[-1]
    body = payment_id[:-2]
    if checksum(body) != int(control_one):
        return False
    return checksum(bill_id + body + control_one) == int(control_two)


def _valid_bill_id(bill_id: str) -> bool:
    """Validate the check digit and the type digit of a bill identifier."""
    if len(bill_id) < 6:
        return False
    return checksum(bill_id[:-1]) == int(bill_id[-1]) and BILL_TYPES.get(int(bill_id[-2:-1])) is not None


def get_detail(
    *,
    bill_id: int | str | None = None,
    payment_id: int | str | None = None,
    barcode: str | None = None,
    currency: int = CURRENCY_TOMAN,
) -> BillDetail:
    """Resolve the details of a utility bill.

    Args:
        bill_id: 13 digit bill identifier.
        payment_id: Payment identifier (at least 6 digits).
        barcode: 26 digit barcode; ``bill_id``/``payment_id`` are derived from it
            when the two explicit arguments are absent.
        currency: Multiplier used to convert the amount (``100`` for Toman,
            ``1000`` for Rial).

    Returns:
        A :class:`BillDetail` with the decoded type, amount and validity flags.

    Raises:
        MissingArgumentError: Neither a barcode nor a bill/payment pair given.

    >>> get_detail(bill_id="1234567890123", payment_id="123456789012")["is_valid"]
    False
    """
    if bill_id is not None and payment_id is not None:
        bill_text = to_english_digits(str(bill_id)).zfill(13)
        payment_text = to_english_digits(str(payment_id))
        derived_barcode = f"{bill_text}000{payment_text}"
    elif barcode is not None:
        raw = to_english_digits(barcode)
        bill_text, payment_text, derived_barcode = raw[:13], raw[16:26], raw
    else:
        raise MissingArgumentError("provide either bill_id + payment_id or a barcode")

    valid_bill = _valid_bill_id(bill_text)
    valid_payment = _valid_payment_id(bill_text, payment_text)
    amount = int(payment_text[:-5]) * currency if valid_payment and len(payment_text) > 5 else None
    bill_type = BILL_TYPES.get(int(bill_text[-2:-1])) if bill_text.isdigit() else None

    return {
        "amount": amount,
        "type": bill_type,
        "barcode": derived_barcode,
        "bill_id": int(bill_text) if bill_text.isdigit() else None,
        "payment_id": int(payment_text) if payment_text.isdigit() else None,
        "is_valid": valid_bill and valid_payment,
        "is_valid_bill_id": valid_bill,
        "is_valid_payment_id": valid_payment,
    }


def validate(
    *,
    bill_id: int | str | None = None,
    payment_id: int | str | None = None,
    barcode: str | None = None,
) -> bool:
    """Shorthand for ``get_detail(...)["is_valid"]``.

    >>> validate(bill_id="1234567890123", payment_id="123456789012")
    False
    """
    return get_detail(bill_id=bill_id, payment_id=payment_id, barcode=barcode)["is_valid"]


def extract_barcodes(text: str) -> list[str]:
    """Find 26 digit bill barcodes inside *text*.

    >>> extract_barcodes("بارکد 12345678901234567890123456")
    ['12345678901234567890123456']
    """
    return re.findall(r"(?<!\d)([۰-۹٠-٩\d]{26})(?!\d)", to_english_digits(text))
