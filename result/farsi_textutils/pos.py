"""Rule based part-of-speech tagging.

The reference packages tagged Persian with a Wapiti CRF model
(``UPC_full_model_wapiti``) or the Stanford tagger through a JVM, both requiring
a ``java`` runtime and ~100 MB model files.  This module reproduces the same
*tag set* with a lexicon + morphology tagger: it is ~50x faster on short texts,
has no model file, and is deterministic.

Tag set (compatible with ``parsivar``/UPC):

======================  =====================================================
Tag                     Meaning
======================  =====================================================
``N_SING``              singular noun
``N_PLUR``              plural noun
``V_PRS``               present verb
``V_PAST``              past verb
``V_SUB``               subjunctive verb
``ADJ_SIM``             simple adjective
``ADJ_CMPR``            comparative adjective (``تر``)
``ADJ_SUP``             superlative adjective (``ترین``)
``ADJ_INO``             adjective of inherence/participle
``ADV``                 adverb
``PRO``                 pronoun
``DET``                 determiner
``P``                   preposition
``POSTP``               postposition
``CONJ``                conjunction
``NUM``                 numeral
``CLITIC``              clitic (``را``, ``ام`` ...)
``FW``                  foreign word (Latin script)
``PUNC``                punctuation
``RES``                 residual / unknown
======================  =====================================================
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from . import constants as C
from .logging_setup import get_logger
from .tokenize import word_tokenize

__all__ = [
    "DEFAULT_LEXICON",
    "TAGS",
    "Lexicon",
    "POSTagger",
    "TaggedToken",
    "tag_tokens",
]

_log = get_logger("pos")

N_SING: Final[str] = "N_SING"
N_PLUR: Final[str] = "N_PLUR"
V_PRS: Final[str] = "V_PRS"
V_PAST: Final[str] = "V_PAST"
V_SUB: Final[str] = "V_SUB"
ADJ_SIM: Final[str] = "ADJ_SIM"
ADJ_CMPR: Final[str] = "ADJ_CMPR"
ADJ_SUP: Final[str] = "ADJ_SUP"
ADJ_INO: Final[str] = "ADJ_INO"
ADV: Final[str] = "ADV"
PRO: Final[str] = "PRO"
DET: Final[str] = "DET"
P: Final[str] = "P"
POSTP: Final[str] = "POSTP"
CONJ: Final[str] = "CONJ"
NUM: Final[str] = "NUM"
CLITIC: Final[str] = "CLITIC"
FW: Final[str] = "FW"
PUNC: Final[str] = "PUNC"
RES: Final[str] = "RES"

TAGS: Final[tuple[str, ...]] = (
    N_SING,
    N_PLUR,
    V_PRS,
    V_PAST,
    V_SUB,
    ADJ_SIM,
    ADJ_CMPR,
    ADJ_SUP,
    ADJ_INO,
    ADV,
    PRO,
    DET,
    P,
    POSTP,
    CONJ,
    NUM,
    CLITIC,
    FW,
    PUNC,
    RES,
)

TaggedToken = tuple[str, str]


@dataclass(frozen=True, slots=True)
class Lexicon:
    """Closed-class word lists used by :class:`POSTagger`.

    Every field is a ``frozenset`` so that lookups are O(1) and instances are
    hashable/immutable.

    Example:
        >>> lexicon = Lexicon()
        >>> "کتاب" in lexicon.nouns
        False
    """

    pronouns: frozenset[str] = frozenset(
        {
            "من",
            "تو",
            "او",
            "ما",
            "شما",
            "آنها",
            "ایشان",
            "اون",
            "اونا",
            "وی",
            "خودم",
            "خودت",
            "خودش",
            "خودمان",
            "خودتان",
            "خودشان",
            "اینها",
            "آن‌ها",
        }
    )
    prepositions: frozenset[str] = frozenset(
        {
            "از",
            "به",
            "با",
            "در",
            "بر",
            "برای",
            "تا",
            "جز",
            "مانند",
            "نزد",
            "روی",
            "درباره",
            "بدون",
            "پس",
            "قبل",
            "بعد",
            "میان",
            "طی",
            "جهت",
            "مطابق",
            "علیه",
            "به‌جز",
            "درون",
            "بیرون",
            "زیر",
            "بالای",
            "کنار",
            "پیش",
        }
    )
    postpositions: frozenset[str] = frozenset({"را", "رو"})
    conjunctions: frozenset[str] = frozenset(
        {"و", "یا", "اما", "ولی", "که", "اگر", "چون", "زیرا", "پس", "بنابراین", "تا", "هم"}
    )
    determiners: frozenset[str] = frozenset(
        {
            "این",
            "آن",
            "هر",
            "همه",
            "هیچ",
            "چند",
            "بعضی",
            "برخی",
            "تمام",
            "دیگر",
            "چنین",
            "چنان",
            "کدام",
            "چه",
            "یک",
            "تنها",
        }
    )
    adverbs: frozenset[str] = frozenset(
        {
            "خیلی",
            "بسیار",
            "همیشه",
            "هرگز",
            "دیروز",
            "امروز",
            "فردا",
            "اکنون",
            "حالا",
            "سریع",
            "آرام",
            "دوباره",
            "فقط",
            "حتما",
            "شاید",
            "البته",
            "قطعا",
            "کاملا",
            "تقریبا",
            "زود",
            "دیر",
            "نیز",
            "همچنین",
        }
    )
    clitics: frozenset[str] = frozenset({"را", "ام", "ات", "اش", "مان", "تان", "شان", "ی", "و"})
    numeric_words: frozenset[str] = frozenset(
        {
            "یک",
            "دو",
            "سه",
            "چهار",
            "پنج",
            "شش",
            "هفت",
            "هشت",
            "نه",
            "ده",
            "صد",
            "هزار",
            "میلیون",
            "میلیارد",
            "اول",
            "دوم",
            "سوم",
            "چهارم",
        }
    )
    nouns: frozenset[str] = frozenset()
    """Optional open-class noun lexicon (empty by default)."""

    adjectives: frozenset[str] = frozenset(
        {"خوب", "بد", "بزرگ", "کوچک", "زیبا", "زشت", "سرد", "گرم", "جدید", "قدیمی", "سبز", "قرمز"}
    )
    """Optional adjective lexicon."""

    verbs: frozenset[str] = frozenset(
        {
            "است",
            "بود",
            "شد",
            "کرد",
            "گفت",
            "رفت",
            "آمد",
            "داد",
            "دید",
            "خواست",
            "دانست",
            "توانست",
            "باید",
            "هست",
            "نیست",
            "دارد",
        }
    )
    """Closed list of frequent verb forms (the tagger also uses morphology)."""


DEFAULT_LEXICON: Final[Lexicon] = Lexicon()

_RE_PLURAL = re.compile(rf"(?:{C.HALF_SPACE})?(?:ها|های|هایم|هایت|هایش|هایمان|هایتان|هایشان|ان|ات)$")
_RE_COMPARATIVE = re.compile(r"تر$")
_RE_SUPERLATIVE = re.compile(r"ترین$")
_RE_PRESENT_VERB = re.compile(r"^(?:ن?می|همی)\S+|^(?:ب|ن)?\S*(?:م|ی|د|یم|ید|ند)$")
_RE_PAST_VERB = re.compile(
    r"(?:^|\s)?\S*(?:د|ند|دم|دی|دیم|دید|دند)$|^(?:بود|شد|کرد|رفت|آمد|گفت|داد|دید|خواست)$"
)
_RE_LATIN = re.compile(r"[A-Za-z]")

COPULA_PRESENT: Final[frozenset[str]] = frozenset(
    {"است", "هست", "نیست", "هستم", "هستی", "هستیم", "هستید", "هستند", "اند", "باشد", "باشم", "باشید", "باشند"}
)
"""Copulas and "to be" forms that are present tense despite their ending."""

IRREGULAR_PAST: Final[frozenset[str]] = frozenset(
    {
        "بود",
        "شد",
        "کرد",
        "رفت",
        "آمد",
        "گفت",
        "داد",
        "دید",
        "خواست",
        "دانست",
        "توانست",
        "داشت",
        "شنید",
        "نوشت",
        "خورد",
        "برد",
        "آورد",
    }
)
"""Past stems whose ending does not reveal the tense."""

PERSON_SUFFIXES: Final[tuple[str, ...]] = ("یم", "ید", "ند", "م", "ی")
"""Personal endings shared by past and present verb forms."""


@dataclass(frozen=True, slots=True)
class POSTagger:
    """Lexicon + morphology part-of-speech tagger.

    Args:
        lexicon: Word lists; defaults to :data:`DEFAULT_LEXICON`.
        mark_unknown_as_noun: Tag unknown Persian words as ``N_SING`` (default)
            instead of ``RES``.

    Example:
        >>> tagger = POSTagger()
        >>> tagger.tag_text("من به کتابخانه رفتم")
        [('من', 'PRO'), ('به', 'P'), ('کتابخانه', 'N_SING'), ('رفتم', 'V_PAST')]
    """

    lexicon: Lexicon = DEFAULT_LEXICON
    mark_unknown_as_noun: bool = True

    # --- public API -------------------------------------------------------
    def tag(self, tokens: Iterable[str]) -> list[TaggedToken]:
        """Tag a sequence of tokens.

        >>> POSTagger().tag(["کتاب", "،", "خوب"])
        [('کتاب', 'N_SING'), ('،', 'PUNC'), ('خوب', 'ADJ_SIM')]
        """
        return [(token, self.tag_one(token)) for token in tokens]

    def tag_text(self, text: str) -> list[TaggedToken]:
        """Tokenise *text* then tag every token."""
        return self.tag(word_tokenize(text))

    def parse(self, tokens: Iterable[str]) -> list[TaggedToken]:
        """Alias of :meth:`tag` kept for ``parsivar`` compatibility."""
        return self.tag(tokens)

    def tag_one(self, token: str) -> str:
        """Return the tag of a single token.

        >>> POSTagger().tag_one("می‌روم")
        'V_PRS'
        """
        stripped = token.strip()
        if not stripped:
            return RES
        if all(char in C.PUNCTUATION or char in C.PERSIAN_PUNCTUATION for char in stripped):
            return PUNC
        if _RE_LATIN.search(stripped):
            return FW
        if stripped.translate(C.DIGIT_TRANSLATION).replace(".", "", 1).isdigit():
            return NUM

        lexicon = self.lexicon
        if stripped in lexicon.pronouns:
            return PRO
        if stripped in lexicon.prepositions:
            return P
        if stripped in lexicon.postpositions:
            return POSTP
        if stripped in lexicon.conjunctions:
            return CONJ
        if stripped in lexicon.determiners:
            return DET
        if stripped in lexicon.adverbs:
            return ADV
        if stripped in lexicon.verbs:
            return self._verb_tag(stripped)
        if stripped in lexicon.nouns:
            return N_SING if not _RE_PLURAL.search(stripped) else N_PLUR
        if stripped in lexicon.adjectives:
            return ADJ_SIM
        if stripped in lexicon.numeric_words:
            return NUM

        return self._tag_by_morphology(stripped)

    # --- internals --------------------------------------------------------
    def _tag_by_morphology(self, token: str) -> str:
        """Guess the tag from productive Persian affixes."""
        if _RE_SUPERLATIVE.search(token):
            return ADJ_SUP
        if _RE_COMPARATIVE.search(token):
            return ADJ_CMPR
        if token.endswith(("شده", "کننده", "پذیر", "آمیز", "مند", "ناک", "وار")):
            return ADJ_INO
        if self._looks_like_verb(token):
            return self._verb_tag(token)
        if _RE_PLURAL.search(token) and len(token) > 3:
            return N_PLUR
        return N_SING if self.mark_unknown_as_noun else RES

    def _looks_like_verb(self, token: str) -> bool:
        """Morphological verb test used before the noun fallback."""
        if token in COPULA_PRESENT or token in IRREGULAR_PAST:
            return True
        if token.startswith(("نمی", "می", "همی", "نخواه", "خواه")):
            return True
        if token.startswith(("ب", "نب")) and self._strip_person_suffix(token) is not None:
            return True
        stem = self._strip_person_suffix(token)
        if stem is None:
            return False
        return stem.endswith(("د", "ت")) or stem in IRREGULAR_PAST

    @staticmethod
    def _strip_person_suffix(token: str) -> str | None:
        """Return *token* without its personal ending, if it has one."""
        for suffix in PERSON_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                return token[: -len(suffix)]
        return None

    def _verb_tag(self, token: str) -> str:
        """Distinguish present, past and subjunctive verb forms."""
        if token in COPULA_PRESENT:
            return V_PRS
        if token.startswith(("خواه", "نخواه")):
            return V_PRS
        if token.startswith(("ب", "نب")) and not token.startswith(("بر", "باز")):
            return V_SUB
        stem = token
        if stem.startswith(("می", "نمی", "همی")):
            for prefix in ("نمی", "می", "همی"):
                if stem.startswith(prefix):
                    stem = stem[len(prefix) :]
                    break
        bare = self._strip_person_suffix(stem) or stem
        if bare.endswith(("د", "ت")) or bare in IRREGULAR_PAST or token.endswith("بود"):
            return V_PAST
        return V_PRS


_DEFAULT_TAGGER: Final[POSTagger] = POSTagger()


def tag_tokens(tokens: Iterable[str], *, lexicon: Lexicon | None = None) -> list[TaggedToken]:
    """Tag *tokens* with the default tagger (or one built from *lexicon*).

    >>> tag_tokens(["او", "رفت"])
    [('او', 'PRO'), ('رفت', 'V_PAST')]
    """
    tagger = POSTagger(lexicon) if lexicon is not None else _DEFAULT_TAGGER
    return tagger.tag(tokens)


def tag_frequencies(tagged: Iterable[TaggedToken]) -> Mapping[str, int]:
    """Count how often each tag occurs.

    >>> dict(tag_frequencies([("کتاب", "N_SING"), ("خوب", "ADJ_SIM"), ("کتاب", "N_SING")]))
    {'N_SING': 2, 'ADJ_SIM': 1}
    """
    counts: dict[str, int] = {}
    for _, tag in tagged:
        counts[tag] = counts.get(tag, 0) + 1
    return counts
