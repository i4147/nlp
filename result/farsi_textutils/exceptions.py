"""Exception hierarchy used across :mod:`farsi_textutils`.

The reference packages raised bare ``Exception`` objects or returned ``None`` to
signal failure.  Everything raised here derives from :class:`FarsiTextUtilsError`
so callers can catch the whole library at once, while still being able to react
to a specific failure mode.
"""

from __future__ import annotations


class FarsiTextUtilsError(Exception):
    """Base class for every error raised by :mod:`farsi_textutils`."""


class ConfigurationError(FarsiTextUtilsError):
    """Raised when an option or a dependency of a component is invalid."""


class InvalidValueError(FarsiTextUtilsError, ValueError):
    """Raised when a string cannot be interpreted (identifier, plate, ...)."""


class InvalidNationalId(InvalidValueError):
    """A national identification number (کد ملی) failed validation."""

    def __init__(self, national_id: str) -> None:
        self.national_id = national_id
        super().__init__(f"{national_id!r} is not a valid Iranian national ID")


class InvalidLegalId(InvalidValueError):
    """A legal-entity identification number (شناسه حقوقی) failed validation."""

    def __init__(self, legal_id: str) -> None:
        self.legal_id = legal_id
        super().__init__(f"{legal_id!r} is not a valid Iranian legal ID")


class InvalidCardNumber(InvalidValueError):
    """A 16 digit bank card number failed the Luhn check."""

    def __init__(self, card_number: str) -> None:
        self.card_number = card_number
        super().__init__(f"{card_number!r} is not a valid Iranian bank card number")


class InvalidShebaNumber(InvalidValueError):
    """An IR-SHEBA (IBAN) number failed the ISO 7064 MOD-97 check."""

    def __init__(self, sheba: str) -> None:
        self.sheba = sheba
        super().__init__(f"{sheba!r} is not a valid Iranian SHEBA (IBAN) number")


class InvalidPhoneNumber(InvalidValueError):
    """A mobile number does not belong to the Iranian numbering plan."""

    def __init__(self, phone_number: str) -> None:
        self.phone_number = phone_number
        super().__init__(f"{phone_number!r} is not a valid Iranian mobile number")


class InvalidPlate(InvalidValueError):
    """A licence plate could not be parsed or does not exist."""

    def __init__(self, plate: str, reason: str = "unknown plate layout") -> None:
        self.plate = plate
        self.reason = reason
        super().__init__(f"{plate!r} is not a valid Iranian plate: {reason}")


class InvalidBillId(InvalidValueError):
    """A bill (قبض) identifier or payment identifier failed its checksum."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class MissingArgumentError(FarsiTextUtilsError):
    """Not enough arguments were supplied to compute a result."""


class ModelNotTrainedError(FarsiTextUtilsError):
    """A statistical component was used before being fitted."""


__all__ = [
    "ConfigurationError",
    "FarsiTextUtilsError",
    "InvalidBillId",
    "InvalidCardNumber",
    "InvalidLegalId",
    "InvalidNationalId",
    "InvalidPhoneNumber",
    "InvalidPlate",
    "InvalidShebaNumber",
    "InvalidValueError",
    "MissingArgumentError",
    "ModelNotTrainedError",
]
