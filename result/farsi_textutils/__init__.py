"""``farsi_textutils`` -- Persian (Farsi) text utilities for Linux/Python 3.12+.

The package is a consolidation of the Persian material of several reference
projects (``parsivar``, ``persian_tools``, ``farsi_tools``, ``farsitools``,
``fakefarsi``) plus the language-agnostic algorithms borrowed from ``camel_tools``,
``janome``, ``snownlp`` and ``soynlp``.  See ``README.md`` for the provenance
table and for the list of things that were dropped (Windows/macOS support,
Python 2, torch/ONNX/sentencepiece models, Wapiti/Stanford Java taggers, NLTK).

Quick start::

    >>> from farsi_textutils import normalize, word_tokenize, number_to_words
    >>> normalize("مي روم به كتاب خانه ها")
    'می‌روم به کتاب خانه‌ها'
    >>> word_tokenize("سلام، دنیا!")
    ['سلام', '،', 'دنیا', '!']
    >>> number_to_words(2030)
    'دو هزار و سی'

The public API is grouped by task; import the sub-module for anything not
re-exported here (``farsi_textutils.validators``, ``farsi_textutils.cli`` ...).
"""

from __future__ import annotations

from typing import Final

from .chunk import Chunk, RegexpChunker, chunk_tokens, to_bracket_string
from .corpus import CorpusPipeline
from .dates import DateNormalizer, normalize_dates, normalize_numbers
from .digits import (
    Language,
    add_ordinal_suffix,
    add_separator,
    convert_digits,
    number_to_words,
    remove_ordinal_suffix,
    words_to_number,
)
from .exceptions import (
    ConfigurationError,
    FarsiTextUtilsError,
    InvalidCardNumber,
    InvalidLegalId,
    InvalidNationalId,
    InvalidPhoneNumber,
    InvalidPlate,
    InvalidShebaNumber,
    InvalidValueError,
    ModelNotTrainedError,
)
from .fake import PersianFaker, fake_profile
from .keywords import WordExtractor, extract_keywords
from .logging_setup import configure_logging, get_logger, logger
from .normalize import (
    Normalizer,
    NormalizerConfig,
    SpaceCorrector,
    check_standard_persian_text,
    clean_text,
    convert_digits_to_ascii,
    dediacritize,
    fix_half_spaces,
    normalize,
    remove_extra_spaces,
    standardize,
    unify_letters,
)
from .parallel import DEFAULT_WORKERS, ParallelConfig, parallel_map
from .paths import iter_files, read_text, write_text
from .pos import POSTagger, tag_tokens
from .similarity import BM25, TfidfVectorizer, WordContextVectorizer
from .spell import NgramModel, SpellChecker
from .stem import Stemmer
from .stopwords import STOP_WORDS, content_words, remove_stop_words
from .tokenize import LTokenizer, MaxScoreTokenizer, Tokenizer, sentence_tokenize, word_tokenize
from .transliterate import slugify, to_latin, to_persian
from .validators import bank_card, bill, legal_id, national_id, phone, plate, sheba

__version__: Final[str] = "1.0.0"
__author__: Final[str] = "farsi_textutils contributors"
__license__: Final[str] = "MIT"

__all__ = [  # noqa: RUF022 - grouped by task, not alphabetically
    "__version__",
    # normalisation & tokenisation
    "Normalizer",
    "NormalizerConfig",
    "SpaceCorrector",
    "Tokenizer",
    "LTokenizer",
    "MaxScoreTokenizer",
    "check_standard_persian_text",
    "clean_text",
    "convert_digits_to_ascii",
    "content_words",
    "dediacritize",
    "fix_half_spaces",
    "normalize",
    "remove_extra_spaces",
    "remove_stop_words",
    "sentence_tokenize",
    "standardize",
    "unify_letters",
    "word_tokenize",
    # digits & dates
    "DateNormalizer",
    "Language",
    "add_ordinal_suffix",
    "add_separator",
    "convert_digits",
    "normalize_dates",
    "normalize_numbers",
    "number_to_words",
    "remove_ordinal_suffix",
    "words_to_number",
    # morphology
    "Stemmer",
    "POSTagger",
    "RegexpChunker",
    "Chunk",
    "chunk_tokens",
    "tag_tokens",
    "to_bracket_string",
    # spelling & statistics
    "NgramModel",
    "SpellChecker",
    "BM25",
    "TfidfVectorizer",
    "WordContextVectorizer",
    "WordExtractor",
    "extract_keywords",
    "STOP_WORDS",
    # transliteration & fake data
    "slugify",
    "to_latin",
    "to_persian",
    "PersianFaker",
    "fake_profile",
    # validators
    "bank_card",
    "bill",
    "legal_id",
    "national_id",
    "phone",
    "plate",
    "sheba",
    # infrastructure
    "CorpusPipeline",
    "DEFAULT_WORKERS",
    "ParallelConfig",
    "configure_logging",
    "get_logger",
    "iter_files",
    "logger",
    "parallel_map",
    "read_text",
    "write_text",
    # exceptions
    "ConfigurationError",
    "FarsiTextUtilsError",
    "InvalidCardNumber",
    "InvalidLegalId",
    "InvalidNationalId",
    "InvalidPhoneNumber",
    "InvalidPlate",
    "InvalidShebaNumber",
    "InvalidValueError",
    "ModelNotTrainedError",
]
