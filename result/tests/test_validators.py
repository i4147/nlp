"""Tests for :mod:`farsi_textutils.validators`."""

from __future__ import annotations

import random

import pytest
from farsi_textutils.exceptions import (
    InvalidCardNumber,
    InvalidNationalId,
    InvalidShebaNumber,
    MissingArgumentError,
)
from farsi_textutils.validators import bank_card, bill, legal_id, national_id, phone, plate, sheba

# --------------------------------------------------------------------------- #
# national id
# --------------------------------------------------------------------------- #


def test_national_id_validate() -> None:
    assert national_id.validate("0499370899")
    assert national_id.validate("۰۴۹۹۳۷۰۸۹۹")
    assert not national_id.validate("0499370898")
    assert not national_id.validate("0000000000")
    assert not national_id.validate("123")


def test_national_id_normalize_and_extract() -> None:
    assert national_id.normalize("۰۴۹-۹۳۷۰۸۹۹") == "0499370899"
    assert national_id.extract("کد ملی من 0499370899 است") == ["0499370899"]


def test_national_id_place() -> None:
    place = national_id.find_place("0499370899")
    assert place is not None
    assert place["province"]
    with pytest.raises(InvalidNationalId):
        national_id.find_place("0499370898")


def test_national_id_generate_round_trip() -> None:
    rng = random.Random(42)
    for _ in range(20):
        value = national_id.generate(rng)
        assert len(value) == national_id.NATIONAL_ID_LENGTH
        assert national_id.validate(value)


# --------------------------------------------------------------------------- #
# legal id
# --------------------------------------------------------------------------- #


def test_legal_id() -> None:
    assert legal_id.validate("29100760136")
    assert not legal_id.validate("29100760135")
    assert legal_id.validate("۱۰۳۲۰۲۴۶۲۴۱")
    assert legal_id.normalize("۲۹-۱۰۰۷۶۰۱۳۶") == "29100760136"
    assert legal_id.extract("شناسه حقوقی 29100760136") == ["29100760136"]
    for _ in range(20):
        assert legal_id.validate(legal_id.generate(random.Random(7)))


def test_legal_id_rejects_wrong_length() -> None:
    assert not legal_id.validate("123")


# --------------------------------------------------------------------------- #
# bank card
# --------------------------------------------------------------------------- #


def test_bank_card_validate() -> None:
    assert bank_card.validate("6037997512345670")
    assert not bank_card.validate("6037997512345671")
    assert bank_card.validate("۶۰۳۷-۹۹۷۵-۱۲۳۴-۵۶۷۰")
    assert bank_card.normalize("۶۰۳۷ ۹۹۷۵ ۱۲۳۴ ۵۶۷۰") == "6037997512345670"


def test_bank_card_bank_data() -> None:
    data = bank_card.bank_data("6037997512345670")
    assert data is not None
    assert data["persian_name"] == "بانک ملی ایران"
    with pytest.raises(InvalidCardNumber):
        bank_card.bank_data("6037997512345671")


def test_bank_card_generate() -> None:
    for _ in range(10):
        card = bank_card.generate_luhn()
        assert len(card) == bank_card.CARD_LENGTH
        assert bank_card.validate(card)


def test_bank_card_extract() -> None:
    text = "کارت من 6037-9975-1234-5670 و 6037-9975-1234-5671 است"
    found = bank_card.extract(text)
    assert [item["pure"] for item in found] == ["6037997512345670"]
    everything = bank_card.extract(text, filter_invalid=False)
    assert len(everything) == 2


# --------------------------------------------------------------------------- #
# sheba
# --------------------------------------------------------------------------- #


def test_sheba_validate_and_generate() -> None:
    generated = sheba.generate()
    assert sheba.validate(generated)
    assert len(sheba.normalize(generated)) == sheba.SHEBA_LENGTH
    assert not sheba.validate("IR000000000000000000000000")
    assert sheba.normalize("ir060540102180020080870007") == "IR060540102180020080870007"


def test_sheba_bank_data() -> None:
    generated = sheba.generate()
    data = sheba.bank_data(generated)
    assert data is not None
    assert data["persian_name"]
    assert "account_number" in data
    with pytest.raises(InvalidShebaNumber):
        sheba.bank_data("IR000000000000000000000000")


def test_bank_code_from_card() -> None:
    assert sheba.bank_code_from_card("6037997512345670") == "017"


# --------------------------------------------------------------------------- #
# bill
# --------------------------------------------------------------------------- #


BILL_ID = "1234567890125"
PAYMENT_ID = "1000000085"


def test_bill_validate_pair() -> None:
    assert bill.validate(bill_id=BILL_ID, payment_id=PAYMENT_ID)
    assert not bill.validate(bill_id="1234567890124", payment_id=PAYMENT_ID)
    assert not bill.validate(bill_id=BILL_ID, payment_id="1000000086")


def test_bill_barcode() -> None:
    detail = bill.get_detail(bill_id=BILL_ID, payment_id=PAYMENT_ID)
    assert detail["type"] == "برق"
    assert detail["amount"] == 1_000_000
    assert len(detail["barcode"]) == 26
    assert bill.validate(barcode=detail["barcode"])
    assert bill.get_detail(barcode=detail["barcode"])["payment_id"] == int(PAYMENT_ID)


def test_bill_checksum_and_extract() -> None:
    assert bill.checksum("1234567890125") > 0
    assert bill.extract_barcodes("بارکد 12345678901250001000000085 تمام") == ["12345678901250001000000085"]
    assert bill.extract_barcodes("12345") == []


def test_bill_missing_arguments() -> None:
    with pytest.raises(MissingArgumentError):
        bill.validate(bill_id=BILL_ID)


# --------------------------------------------------------------------------- #
# phone
# --------------------------------------------------------------------------- #


def test_phone_validate() -> None:
    assert phone.validate("09123456789")
    assert phone.validate("۰۹۱۲۳۴۵۶۷۸۹")
    assert phone.validate("+989123456789")
    assert not phone.validate("0212345678")


def test_phone_normalize_and_operator() -> None:
    assert phone.normalize("+98 912 345 6789") == "09123456789"
    info = phone.operator_data("09123456789")
    assert info["operator"] == "همراه اول"
    assert phone.generate(prefix="935")
    assert phone.validate(phone.generate(prefix="935"))


def test_phone_extract() -> None:
    text = "شماره‌ها: 09123456789 و 09351234567"
    found = phone.extract(text)
    assert [item["pure"] for item in found] == ["09123456789", "09351234567"]


# --------------------------------------------------------------------------- #
# plate
# --------------------------------------------------------------------------- #


def test_plate_car() -> None:
    assert plate.is_valid("12ب345-67")
    assert plate.is_valid("۱۲ب۳۴۵۶۷")
    assert not plate.is_valid("12x345-67")
    assert not plate.is_valid("12ب345-6")
    info = plate.get_info("12ب345-67")
    assert info["province"]
    assert info["category"] == "شخصی"
    assert plate.render("12ب345-67") == "12ب345ایران67"


def test_plate_motorcycle() -> None:
    assert plate.is_valid("123-45678")
    info = plate.get_info("123-45678")
    assert info["type"] == plate.TYPE_MOTORCYCLE
    assert info["category"] is None


def test_plate_normalize() -> None:
    parts = plate.normalize("۱۲ب۳۴۵-۶۷")
    assert parts["char"] == "ب"
    assert parts["numbers"] == "1234567"


# --------------------------------------------------------------------------- #
# exceptions
# --------------------------------------------------------------------------- #


def test_exception_hierarchy() -> None:
    for error in (InvalidCardNumber, InvalidNationalId, InvalidShebaNumber):
        with pytest.raises(error):
            raise error("x")
