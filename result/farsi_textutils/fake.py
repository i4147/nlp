"""Fake Persian data generation.

Replaces ``fakefarsi`` (plus ``PyFTBot``, which was a Telegram helper around it)
with a single seedable generator.  Names come from
:mod:`farsi_textutils.data.names`; every identifier is generated *valid*, i.e.
passing the checksum of the matching validator.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Final, TypedDict

from .data.names import FAMILY_NAMES, FEMALE_NAMES, MALE_NAMES
from .data.plate import PROVINCES, TYPE_CAR
from .exceptions import ConfigurationError
from .logging_setup import get_logger
from .transliterate import slugify
from .validators import bank_card, national_id, phone, sheba

__all__ = [
    "FakeProfile",
    "PersianFaker",
    "fake_email",
    "fake_full_name",
    "fake_national_id",
    "fake_phone",
    "fake_profile",
    "fake_sentence",
]

_log = get_logger("fake")

_PROVINCES: Final[tuple[str, ...]] = tuple(sorted(set(PROVINCES.get(TYPE_CAR, {}).values())))
_WORDS: Final[tuple[str, ...]] = (
    "کتاب",
    "خانه",
    "مدرسه",
    "دانشگاه",
    "دانشجو",
    "معلم",
    "کودک",
    "شهر",
    "ایران",
    "تهران",
    "ماشین",
    "مادر",
    "پدر",
    "دوست",
    "آب",
    "نان",
    "درخت",
    "گل",
    "آسمان",
    "دریا",
    "کوه",
    "راه",
    "کار",
    "زندگی",
)
_VERBS: Final[tuple[str, ...]] = ("است", "بود", "شد", "می‌رود", "می‌آید", "می‌بیند", "دارد")
_PUNCTUATION: Final[tuple[str, ...]] = ("", "", ".", "!", "؟")


class FakeProfile(TypedDict):
    """A coherent fake person."""

    first_name: str
    last_name: str
    full_name: str
    age: int
    city: str
    phone: str
    email: str
    national_id: str


@dataclass(slots=True)
class PersianFaker:
    """Seedable generator of fake Persian data.

    Args:
        seed: Seed for the internal :class:`random.Random`; identical seeds give
            identical data (UTF-8 safe, unlike ``random.seed(str)`` misuse).

    Example:
        >>> faker = PersianFaker(seed=42)
        >>> faker.full_name() == PersianFaker(seed=42).full_name()
        True
        >>> phone.validate(faker.phone_number())
        True
    """

    seed: int | None = None
    _random: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.seed is not None and not isinstance(self.seed, int):
            raise ConfigurationError("seed must be an int or None")
        self._random = random.Random(self.seed)

    # --- people -----------------------------------------------------------
    def first_name(self, *, gender: str | None = None) -> str:
        """Return a Persian given name.

        Args:
            gender: ``"male"``, ``"female"`` or ``None`` for both.
        """
        if gender == "male":
            return self._random.choice(MALE_NAMES)
        if gender == "female":
            return self._random.choice(FEMALE_NAMES)
        if gender is not None:
            raise ConfigurationError("gender must be 'male', 'female' or None")
        return self._random.choice(MALE_NAMES + FEMALE_NAMES)

    def last_name(self) -> str:
        """Return a Persian family name."""
        return self._random.choice(FAMILY_NAMES)

    def full_name(self, *, gender: str | None = None) -> str:
        """Return a full name."""
        return f"{self.first_name(gender=gender)} {self.last_name()}"

    def age(self, *, minimum: int = 10, maximum: int = 99) -> int:
        """Return an age within ``[minimum, maximum]``."""
        if minimum > maximum:
            raise ConfigurationError("minimum must be <= maximum")
        return self._random.randint(minimum, maximum)

    def city(self) -> str:
        """Return the name of an Iranian province/city from the plate table."""
        return self._random.choice(_PROVINCES)

    # --- identifiers ------------------------------------------------------
    def phone_number(self, *, prefix: str = "912") -> str:
        """Return a valid mobile number."""
        return phone.generate(self._random, prefix=prefix)

    def national_id(self) -> str:
        """Return a valid national ID."""
        return national_id.generate(self._random)

    def legal_id(self) -> str:
        """Return a valid legal-entity ID."""
        from .validators import legal_id as legal

        return legal.generate(self._random)

    def card_number(self) -> str:
        """Return a valid bank card number of a real bank prefix."""
        prefix = self._random.choice(("603799", "610433", "627412", "627353", "589463"))
        body = f"{self._random.randint(0, 10**8 - 1):08d}" + f"{self._random.randint(1, 99999):05d}"
        return bank_card.generate_luhn(prefix, body)

    def sheba(self) -> str:
        """Return a valid SHEBA (IBAN) number."""
        return sheba.generate(f"{self._random.randint(0, 10**22 - 1):022d}")

    def email(self, *, domain: str = "example.com") -> str:
        """Return an e-mail address derived from a random name."""
        local = slugify(f"{self.first_name()} {self.last_name()}", separator=".")
        return f"{local or 'user'}@{domain}"

    # --- text -------------------------------------------------------------
    def sentence(self, *, words: int = 6) -> str:
        """Return a fake sentence of *words* nouns plus a closing verb."""
        body = [self._random.choice(_WORDS) for _ in range(max(1, words))]
        return f"{' '.join(body)} {self._random.choice(_VERBS)}{self._random.choice(_PUNCTUATION)}"

    def paragraph(self, *, sentences: int = 3) -> str:
        """Return a fake paragraph of *sentences* sentences."""
        return " ".join(self.sentence() for _ in range(max(1, sentences)))

    def profile(self) -> FakeProfile:
        """Return a coherent fake person record.

        >>> sorted(PersianFaker(seed=1).profile()) == [
        ...     "age", "city", "email", "first_name", "full_name",
        ...     "last_name", "national_id", "phone",
        ... ]
        True
        """
        first = self.first_name()
        last = self.last_name()
        return {
            "first_name": first,
            "last_name": last,
            "full_name": f"{first} {last}",
            "age": self.age(),
            "city": self.city(),
            "phone": self.phone_number(),
            "email": self.email(),
            "national_id": self.national_id(),
        }


_DEFAULT_FAKER: Final[PersianFaker] = PersianFaker()


def fake_full_name(*, seed: int | None = None) -> str:
    """Return a random full name.

    >>> bool(fake_full_name(seed=7))
    True
    """
    return PersianFaker(seed).full_name()


def fake_phone(*, seed: int | None = None) -> str:
    """Return a random valid mobile number."""
    return PersianFaker(seed).phone_number()


def fake_national_id(*, seed: int | None = None) -> str:
    """Return a random valid national ID."""
    return PersianFaker(seed).national_id()


def fake_email(*, seed: int | None = None) -> str:
    """Return a random e-mail address."""
    return PersianFaker(seed).email()


def fake_sentence(*, seed: int | None = None) -> str:
    """Return a random Persian sentence."""
    return PersianFaker(seed).sentence()


def fake_profile(*, seed: int | None = None) -> FakeProfile:
    """Return a random fake person record."""
    return PersianFaker(seed).profile()
