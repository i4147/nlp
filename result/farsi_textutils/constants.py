"""Character inventories and regular expressions for Persian text.

All tables are immutable (``Final``/``frozenset``/``tuple``) so importing this
module is cheap and safe from several processes.
"""

from __future__ import annotations

import re
from typing import Final

# --- separators ----------------------------------------------------------

HALF_SPACE: Final[str] = "\u200c"
"""Zero-width non-joiner (نیم‌فاصله), the Persian word-internal separator."""

ZERO_WIDTH_CHARS: Final[frozenset[str]] = frozenset(
    {
        "\u200b",  # zero width space
        "\u200c",  # zero width non-joiner (kept only when asked for)
        "\u200d",  # zero width joiner
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
        "\ufeff",  # byte order mark
    }
)

# --- digits --------------------------------------------------------------

PERSIAN_DIGITS: Final[str] = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS: Final[str] = "٠١٢٣٤٥٦٧٨٩"
ASCII_DIGITS: Final[str] = "0123456789"

DIGIT_MAP: Final[dict[str, str]] = {
    **dict(zip(PERSIAN_DIGITS, ASCII_DIGITS, strict=True)),
    **dict(zip(ARABIC_DIGITS, ASCII_DIGITS, strict=True)),
}

DIGIT_TRANSLATION: Final[dict[int, int]] = {ord(source): ord(target) for source, target in DIGIT_MAP.items()}
"""``str.translate`` table mapping every Persian/Arabic digit to ASCII."""

ASCII_TO_PERSIAN_DIGITS: Final[dict[int, int]] = str.maketrans(ASCII_DIGITS, PERSIAN_DIGITS)
ASCII_TO_ARABIC_DIGITS: Final[dict[int, int]] = str.maketrans(ASCII_DIGITS, ARABIC_DIGITS)
PERSIAN_TO_ARABIC_DIGITS: Final[dict[int, int]] = str.maketrans(PERSIAN_DIGITS, ARABIC_DIGITS)

# --- letters -------------------------------------------------------------

PERSIAN_LETTERS: Final[str] = "آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"
"""Persian alphabet without the Arabic-only letters and without variants."""

ARABIC_VARIANTS: Final[dict[str, str]] = {
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
    "ة": "ه",
    "ۀ": "ه",
    "ؤ": "و",
    "إ": "ا",
    "أ": "ا",
    "ٱ": "ا",
    "ٲ": "ا",
    "ئ": "ئ",
}

DIACRITICS: Final[frozenset[str]] = frozenset("ًٌٍَُِّْـ")
"""Arabic/Persian combining marks (tashdid, harakat, tatweel)."""

PERSIAN_VOWELS_AND_MARKS: Final[frozenset[str]] = frozenset("اًهیو")

# --- punctuation ---------------------------------------------------------

PERSIAN_PUNCTUATION: Final[str] = "؟!،؛«»ـ٪٫"
ASCII_PUNCTUATION: Final[str] = "?!,;.()[]{}<>\"'/:*+-=_@#$%&|\\~^`"

PUNCTUATION: Final[frozenset[str]] = frozenset(PERSIAN_PUNCTUATION + ASCII_PUNCTUATION)

PUNCTUATION_EQUIVALENTS: Final[dict[str, str]] = {
    ",": "،",
    ";": "؛",
    "?": "؟",
    "٪": "٪",
    "%": "٪",
}

# --- character classes used by the reference implementations -------------

WORD_REGEX: Final[str] = (
    "[\u200c\u0622\u0627\u0628\u067e\u062a-\u062c\u0686\u062d-\u0632\u0698"
    "\u0633-\u063a\u0641\u0642\u06a9\u06af\u0644-\u0648\u06ccA-Za-z0-9\u06f0-\u06f9]"
)
NON_WORD_REGEX: Final[str] = "[^" + WORD_REGEX[1:]

VALID_TEXT_CHARS: Final[frozenset[str]] = frozenset(
    ' ۱۲۳۴۵۶۷۸۹۰٪-+_ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی!؟:.،؛«»"{}()[],آة|/ء<>*'
)
VALID_TEXT_CHARS_LATIN: Final[frozenset[str]] = VALID_TEXT_CHARS | frozenset(
    "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ;'"
)

# --- morphology ----------------------------------------------------------

NOUN_SUFFIXES: Final[tuple[str, ...]] = (
    "هایمان",
    "هایتان",
    "هایشان",
    "هایی",
    "هایش",
    "هایم",
    "هایت",
    "های",
    "های",
    "مان",
    "تان",
    "شان",
    "های",
    "ترین",
    "ها",
    "ات",
    "ان",
    "ین",
    "ام",
    "ای",
    "یم",
    "ید",
    "اند",
    "م",
    "ت",
    "ش",
    "ی",
)
"""Plural / possessive / superlative suffixes stripped by :mod:`farsi_textutils.stem`."""

VERB_PREFIXES: Final[tuple[str, ...]] = (
    "می",
    "نمی",
    "همی",
    "خواه",
    "نخواه",
    "ب",
    "ن",
    "در",
    "بر",
    "پر",
    "باز",
    "ور",
    "فرو",
    "فرا",
    "وا",
)
VERB_SUFFIXES: Final[tuple[str, ...]] = ("یم", "ید", "ند", "م", "ی", "د", "")

PRONOUN_SUFFIXES: Final[tuple[str, ...]] = (
    "ام",
    "ای",
    "است",
    "ایم",
    "اید",
    "اند",
)

SPACE_MERGE_PREFIXES: Final[tuple[str, ...]] = ("بی", "می", "نمی")
SPACE_MERGE_SUFFIXES: Final[tuple[str, ...]] = (
    "هایی",
    "ها",
    "های",
    "هایی",
    "هایم",
    "هایت",
    "هایش",
    "هایمان",
    "هایتان",
    "هایشان",
    "ات",
    "ان",
    "ین",
    "انی",
    "بان",
    "ام",
    "ای",
    "یم",
    "ید",
    "اید",
    "اند",
    "ست",
    "تر",
    "ترین",
    "شده",
    "نشده",
    "گر",
    "گری",
    "گرا",
    "گرایان",
    "شناس",
    "شناسی",
    "شناسان",
    "گذار",
    "گذاری",
    "گذاران",
    "گیری",
    "پذیری",
    "پذیر",
    "بندی",
    "آوری",
    "سازی",
    "کننده",
    "کنندگان",
    "پرداز",
    "پردازی",
    "پردازان",
    "آمیز",
    "سنجی",
    "ریزی",
    "داری",
    "دار",
    "دهنده",
    "رسانی",
    "یاب",
    "یابی",
    "گانه",
    "انگاری",
    "بند",
)

AGENT_SUFFIXES: Final[tuple[str, ...]] = (
    "کنندگان",
    "کننده",
    "گذاران",
    "گذاری",
    "گذار",
    "شناسان",
    "شناسی",
    "شناس",
    "پردازان",
    "پردازی",
    "پرداز",
    "دهنده",
    "رسانی",
    "انگاری",
    "پذیری",
    "پذیر",
    "گیری",
    "سنجی",
    "یابی",
    "یاب",
    "بندی",
    "آوری",
    "سازی",
    "ریزی",
    "آمیز",
    "گانه",
    "گرایان",
    "گری",
    "گرا",
    "گر",
    "بند",
    "داری",
    "دار",
)
"""Agent-noun suffixes joined with a half-space (``کتاب‌ساز``, ``زبان‌شناس``)."""

# --- calendar ------------------------------------------------------------

MONTHS: Final[dict[str, int]] = {
    "فروردین": 1,
    "اردیبهشت": 2,
    "خرداد": 3,
    "تیر": 4,
    "مرداد": 5,
    "شهریور": 6,
    "مهر": 7,
    "آبان": 8,
    "آذر": 9,
    "دی": 10,
    "بهمن": 11,
    "اسفند": 12,
}

MONTH_NAMES: Final[tuple[str, ...]] = tuple(MONTHS)

WEEKDAYS: Final[tuple[str, ...]] = (
    "شنبه",
    "یکشنبه",
    "دوشنبه",
    "سه‌شنبه",
    "چهارشنبه",
    "پنجشنبه",
    "جمعه",
)

# --- sentence segmentation ----------------------------------------------

ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "مث",
        "مثل",
        "مثلا",
        "مثلاً",
        "ق",
        "قرن",
        "ص",
        "ش",
        "ه‍",
        "ه",
        "دکتر",
        "مهندس",
        "استاد",
        "آقای",
        "خانم",
        "جناب",
        "سرکار",
        "سرهنگ",
        "سرلشکر",
        "تیمسار",
        "ریاست",
        "مدیر",
        "شرکت",
        "سازمان",
        "اداره",
        "خیابان",
        "کوچه",
        "پلاک",
        "طبقه",
        "شماره",
        "تلفن",
        "کد",
        "پ.ن",
        "ع.پ",
    }
)
"""Tokens after which a sentence-final dot does *not* end the sentence."""

SENTENCE_DELIMITERS: Final[str] = ".؟?!؛;"

# --- regular expressions (pre-compiled at import time) -------------------

RE_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"[ \t\f\v\r]+")
RE_MULTI_NEWLINE: Final[re.Pattern[str]] = re.compile(r"\n{2,}")
RE_ALL_SPACE: Final[re.Pattern[str]] = re.compile(r"\s+")
RE_URL: Final[re.Pattern[str]] = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL: Final[re.Pattern[str]] = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
RE_NUMBER: Final[re.Pattern[str]] = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
RE_PERSIAN_NUMBER: Final[re.Pattern[str]] = re.compile(r"[-+]?[۰-۹٠-٩]*[.٫]?[۰-۹٠-٩]+")
RE_LATIN_WORD: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
RE_PERSIAN_WORD: Final[re.Pattern[str]] = re.compile(r"[" + PERSIAN_LETTERS + "\u200c]+")
RE_TOKEN: Final[re.Pattern[str]] = re.compile(
    RE_URL.pattern
    + "|"
    + RE_EMAIL.pattern
    + "|"
    + RE_NUMBER.pattern
    + r"|\w+(?:['’\-]\w+)*|["
    + re.escape(PERSIAN_PUNCTUATION + ASCII_PUNCTUATION)
    + "]",
    re.UNICODE,
)
RE_SEQUENCE_OF_SPACES: Final[re.Pattern[str]] = re.compile(r" {2,}")

__all__ = [
    "ABBREVIATIONS",
    "AGENT_SUFFIXES",
    "ARABIC_DIGITS",
    "ARABIC_VARIANTS",
    "ASCII_DIGITS",
    "ASCII_PUNCTUATION",
    "ASCII_TO_ARABIC_DIGITS",
    "ASCII_TO_PERSIAN_DIGITS",
    "DIACRITICS",
    "DIGIT_TRANSLATION",
    "HALF_SPACE",
    "MONTHS",
    "MONTH_NAMES",
    "NON_WORD_REGEX",
    "NOUN_SUFFIXES",
    "PERSIAN_DIGITS",
    "PERSIAN_LETTERS",
    "PERSIAN_PUNCTUATION",
    "PERSIAN_TO_ARABIC_DIGITS",
    "PRONOUN_SUFFIXES",
    "PUNCTUATION",
    "PUNCTUATION_EQUIVALENTS",
    "RE_ALL_SPACE",
    "RE_EMAIL",
    "RE_LATIN_WORD",
    "RE_MULTI_NEWLINE",
    "RE_NUMBER",
    "RE_PERSIAN_NUMBER",
    "RE_PERSIAN_WORD",
    "RE_TOKEN",
    "RE_URL",
    "RE_WHITESPACE",
    "SENTENCE_DELIMITERS",
    "SPACE_MERGE_PREFIXES",
    "SPACE_MERGE_SUFFIXES",
    "VALID_TEXT_CHARS",
    "VALID_TEXT_CHARS_LATIN",
    "VERB_PREFIXES",
    "VERB_SUFFIXES",
    "WEEKDAYS",
    "WORD_REGEX",
    "ZERO_WIDTH_CHARS",
]
