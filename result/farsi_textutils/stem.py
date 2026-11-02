"""Persian stemming (ریشه‌یابی) and morphological stripping.

Ported from ``parsivar.stemmer.FindStems``.  The reference implementation
loaded four pickles produced from the Persian Dependency Treebank and fell back
to rebuilding them from a corpus directory; the pickles are not part of the
reference checkout, so this module:

* keeps the *exact* affix analysis order of the original (verb patterns first,
  then noun patterns) as declarative rule tables instead of deep nesting;
* works without any data file -- a :class:`Stemmer` with no lexicons applies
  affix stripping only;
* accepts plain-text or JSON lexicons, loaded with ``pathlib`` (no pickle, so no
  arbitrary-code-execution risk from model files).

The public entry point is :meth:`Stemmer.stem`, with :meth:`Stemmer.convert_to_stem`
kept as a ``parsivar`` compatible alias.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from . import constants as C
from .logging_setup import get_logger
from .paths import read_lines

__all__ = [
    "AffixRules",
    "Stemmer",
    "StemmerConfig",
    "load_lexicon",
    "load_verb_tense_map",
]

_log = get_logger("stem")


def load_lexicon(path: Path | str) -> frozenset[str]:
    """Read a word list (one entry per line) for the stemmer lexicons."""
    return frozenset(read_lines(path))


def load_verb_tense_map(path: Path | str) -> dict[str, str]:
    """Read a ``present<tab>past`` verb stem mapping in JSON or TSV form."""
    file_path = Path(path).expanduser()
    if file_path.suffix == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return {str(key): str(value) for key, value in data.items()}
    mapping: dict[str, str] = {}
    for line in read_lines(file_path):
        present, _, past = line.partition("\t")
        if present and past:
            mapping[present] = past
    return mapping


@dataclass(frozen=True, slots=True)
class AffixRules:
    """Declarative affix tables used by :class:`Stemmer`.

    Defaults reproduce the literal lists of ``FindStems.convert_to_stem``.
    """

    prefixes: tuple[str, ...] = C.VERB_PREFIXES
    verb_suffixes: tuple[str, ...] = C.VERB_SUFFIXES
    pronoun_suffixes: tuple[str, ...] = C.PRONOUN_SUFFIXES
    noun_suffixes: tuple[str, ...] = C.NOUN_SUFFIXES
    plural_suffixes: tuple[str, ...] = ("ها", "ات", "های", "ان", "هایی", "ین")
    past_auxiliaries: tuple[str, ...] = ("بود", "باشد", "باش")
    progressive_prefixes: tuple[str, ...] = ("داشتم", "داشتی", "داشت", "داشتیم", "داشتید", "داشتند")
    possession_prefixes: tuple[str, ...] = ("دارم", "داری", "دارد", "داریم", "دارید", "دارند")
    future_prefixes: tuple[str, ...] = ("خواه", "نخواه")
    """Prefix tables in the original evaluation order."""


@dataclass(frozen=True, slots=True)
class StemmerConfig:
    """Options for :class:`Stemmer`."""

    rules: AffixRules = AffixRules()
    strip_nouns: bool = True
    strip_verbs: bool = True
    keep_verb_pair: bool = True
    """When ``True`` verbs are returned as ``present&past`` (``parsivar`` style)."""


@dataclass(slots=True)
class Stemmer:
    """Rule based Persian stemmer with optional lexicons.

    Args:
        config: Behaviour flags and affix tables.
        nouns: Known noun stems; enables the "is this a real word?" checks.
        verbs: Known verb stems.
        verb_map: ``present -> past`` mapping used to pair verb stems.
        irregular_nouns: ``broken plural -> singular`` mapping.

    Example:
        >>> stemmer = Stemmer(nouns={"کتاب"}, verbs={"رو"}, verb_map={"رو": "رفت"})
        >>> stemmer.stem("کتاب‌ها")
        'کتاب'
        >>> stemmer.stem("می‌روم")
        'رو&رفت'
    """

    config: StemmerConfig = field(default_factory=StemmerConfig)
    nouns: frozenset[str] = frozenset()
    verbs: frozenset[str] = frozenset()
    verb_map: Mapping[str, str] = field(default_factory=dict)
    irregular_nouns: Mapping[str, str] = field(default_factory=dict)

    # --- public API -------------------------------------------------------
    def stem(self, word: str, pos: str | None = None) -> str:
        """Return the stem of *word*.

        Args:
            word: A single Persian word.
            pos: Optional part of speech (``"N"`` or ``"V"``) restricting the
                analysis pass; ``None`` tries both, verbs first.

        Returns:
            The stem, unchanged when no rule applies.
        """
        normalized = word.strip(C.HALF_SPACE)
        if not normalized:
            return word
        if normalized in self.verb_map:
            return self._verb_pair(normalized) if normalized in self.verbs else self.verb_map[normalized]

        if pos in (None, "N") and normalized in self.nouns:
            return self._irregular(normalized)
        if pos in (None, "V") and normalized in self.verbs:
            return self._verb_pair(normalized)

        if self.config.strip_verbs and pos in (None, "V"):
            verb = self._verb_stem(normalized)
            if verb is not None:
                return verb

        if self.config.strip_nouns and pos in (None, "N"):
            noun = self._noun_stem(normalized)
            if noun is not None:
                return noun

        return self._fallback(normalized, pos)

    def convert_to_stem(self, word: str, pos: str | None = None) -> str:
        """``parsivar`` compatible alias of :meth:`stem`."""
        return self.stem(word, pos)

    def stem_all(self, words: Iterable[str], pos: str | None = None) -> list[str]:
        """Stem every word of *words*."""
        return [self.stem(word, pos) for word in words]

    # --- verb analysis ----------------------------------------------------
    def _verb_pair(self, verb: str) -> str:
        """Return ``present&past`` for a known verb stem."""
        if not self.config.keep_verb_pair:
            return verb
        if verb in self.verb_map:
            return f"{verb}&{self.verb_map[verb]}"
        return verb

    def _verb_stem(self, word: str) -> str | None:
        """Try every verb pattern of the reference implementation."""
        for candidate in self._strip(word, self.config.rules.progressive_prefixes, "prefix"):
            for removed in self._strip(candidate, ("می",), "prefix"):
                for suffixless in self._strip(removed, self.config.rules.verb_suffixes, "suffix"):
                    if suffixless in self.verbs:
                        return self._verb_pair(suffixless)

        for candidate in self._strip(word, self.config.rules.possession_prefixes, "prefix"):
            for without_mi in self._strip(candidate, ("می",), "prefix"):
                for suffixless in self._strip(without_mi, self.config.rules.verb_suffixes, "suffix"):
                    if suffixless in self.verbs:
                        return self._verb_pair(suffixless)

        for prefix in (("می", "نمی", "همی"), self.config.rules.prefixes):
            for candidate in self._strip(word, prefix, "prefix"):
                for suffixless in self._strip(candidate, self.config.rules.verb_suffixes, "suffix"):
                    if suffixless in self.verbs:
                        return self._verb_pair(suffixless)

        for candidate in self._strip(word, self.config.rules.verb_suffixes, "suffix"):
            for auxiliary_less in self._strip(candidate, self.config.rules.past_auxiliaries, "suffix"):
                for without_heh in self._strip(auxiliary_less, ("ه",), "suffix"):
                    for prefixless in self._strip(without_heh, ("ن", ""), "prefix"):
                        if prefixless in self.verbs:
                            return self._verb_pair(prefixless)

        for candidate in self._strip(word, self.config.rules.pronoun_suffixes, "suffix"):
            for without_heh in self._strip(candidate, ("ه",), "suffix"):
                for prefixless in self._strip(without_heh, ("ن", ""), "prefix"):
                    if prefixless in self.verbs:
                        return self._verb_pair(prefixless)

        for candidate in self._strip(word, self.config.rules.future_prefixes, "prefix"):
            for suffixless in self._strip(candidate, self.config.rules.verb_suffixes, "suffix"):
                if suffixless in self.verbs:
                    return self._verb_pair(suffixless)

        for candidate in self._strip(word, ("ب", "ن", ""), "prefix"):
            for suffixless in self._strip(candidate, self.config.rules.verb_suffixes, "suffix"):
                adjusted = self._repair_ya(suffixless)
                if adjusted in self.verbs:
                    return self._verb_pair(adjusted)
        return None

    @staticmethod
    def _repair_ya(word: str) -> str:
        """Undo the ``یا``/``آ`` and ``ی``/``ا`` spelling alternations."""
        if word.startswith("یا") and len(word) > 2:
            return "آ" + word[2:]
        if word.startswith("ی") and len(word) > 1:
            return "ا" + word[1:]
        if word.endswith(("آی", "ای")) and len(word) > 2:
            return word[:-1]
        return word

    # --- noun analysis ----------------------------------------------------
    def _noun_stem(self, word: str) -> str | None:
        """Try the noun patterns of the reference implementation."""
        rules = self.config.rules
        for candidate in self._strip(word, self._possessive_suffixes(), "suffix"):
            if candidate in self.nouns:
                return self._irregular(candidate)
            for plural in self._strip(candidate, rules.plural_suffixes, "suffix"):
                if plural in self.nouns:
                    return self._irregular(plural)
                for with_heh in self._strip(plural, ("گ",), "suffix"):
                    if f"{with_heh}ه" in self.nouns:
                        return self._irregular(f"{with_heh}ه")
                for generic in self._strip(plural, C.AGENT_SUFFIXES, "suffix"):
                    if generic in self.nouns:
                        return self._irregular(generic)
                    for prefixless in self._strip(generic, C.SPACE_MERGE_PREFIXES, "prefix"):
                        if prefixless in self.nouns:
                            return self._irregular(prefixless)
        return None

    @staticmethod
    def _possessive_suffixes() -> tuple[str, ...]:
        """Possessive/subject suffixes stripped before the plural pass."""
        return (
            "یمان",
            "یتان",
            "یشان",
            "مان",
            "تان",
            "شان",
            "یم",
            "یت",
            "یش",
            "ام",
            "ات",
            "اش",
            "م",
            "ت",
            "ش",
        )

    def _irregular(self, word: str) -> str:
        """Map a broken plural onto its singular when the table knows it."""
        return self.irregular_nouns.get(word, word)

    def _fallback(self, word: str, pos: str | None) -> str:
        """Affix stripping without any lexicon.

        Used when a :class:`Stemmer` is built without data files, so that the
        library stays useful out of the box (``کتاب‌ها`` -> ``کتاب``).  Nouns are
        tried before verbs because un-annotated Persian text is noun heavy.
        """
        verb_prefix = word.startswith(("نمی", "می", "همی", "نخواه", "خواه"))
        if pos in (None, "N") and not verb_prefix and self.config.strip_nouns:
            noun = self._strip_any(word, self._noun_fallback_suffixes())
            if noun is not None:
                return noun
        if pos in (None, "V") and self.config.strip_verbs:
            bare = self._strip_any(word, ("نمی", "می", "همی", "نخواه", "خواه", "ب", "ن"), prefix=True)
            if bare is not None and len(bare) > 1:
                return bare
            verb = self._strip_any(word, ("یم", "ید", "ند", "م", "ی", "د"))
            if verb is not None and len(verb) > 1:
                return verb
        return word

    def _noun_fallback_suffixes(self) -> tuple[str, ...]:
        """Suffixes considered by the lexicon free fallback, longest first."""
        suffixes = (
            set(self.config.rules.noun_suffixes)
            | set(self.config.rules.plural_suffixes)
            | set(C.AGENT_SUFFIXES)
        )
        return tuple(sorted(suffixes, key=len, reverse=True))

    @staticmethod
    def _strip_any(word: str, affixes: Sequence[str], *, prefix: bool = False) -> str | None:
        """Strip the first matching affix of *affixes* (longest first).

        A minimum stem length is enforced so that the lexicon free fallback does
        not turn ``سلام`` into ``سل`` just because it ends with ``ام``.
        """
        for affix in affixes:
            if not affix:
                continue
            minimum = 4 if len(affix) == 1 else 3
            if prefix and word.startswith(affix) and len(word) - len(affix) >= minimum:
                return word[len(affix) :].strip(C.HALF_SPACE)
            if not prefix and word.endswith(affix) and len(word) - len(affix) >= minimum:
                return word[: -len(affix)].strip(C.HALF_SPACE)
        return None

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _strip(word: str, affixes: Sequence[str], mode: str) -> list[str]:
        """Return every word obtained by removing one *affix*.

        >>> Stemmer._strip("می‌روم", ("می", "نمی"), "prefix")
        ['روم']
        """
        results: list[str] = []
        source = word.strip(C.HALF_SPACE)
        for affix in affixes:
            if mode == "prefix" and affix and source.startswith(affix):
                results.append(source[len(affix) :].strip(C.HALF_SPACE))
            elif mode == "prefix" and not affix:
                results.append(source)
            elif mode == "suffix" and affix and source.endswith(affix):
                results.append(source[: -len(affix)].strip(C.HALF_SPACE))
            elif mode == "suffix" and not affix:
                results.append(source)
        return [result for result in results if result]

    @classmethod
    def with_lexicons(
        cls,
        *,
        noun_lexicon: Path | str | None = None,
        verb_lexicon: Path | str | None = None,
        verb_map: Path | str | None = None,
        config: StemmerConfig | None = None,
    ) -> Stemmer:
        """Build a stemmer from plain text/JSON lexicon files.

        >>> Stemmer.with_lexicons() .nouns
        frozenset()
        """
        return cls(
            config=config or StemmerConfig(),
            nouns=load_lexicon(noun_lexicon) if noun_lexicon else frozenset(),
            verbs=load_lexicon(verb_lexicon) if verb_lexicon else frozenset(),
            verb_map=load_verb_tense_map(verb_map) if verb_map else {},
        )


_DEFAULT_STEMMER: Final[Stemmer] = Stemmer()


def stem(word: str, pos: str | None = None) -> str:
    """Stem *word* with the default (lexicon free) stemmer.

    >>> stem("کتاب‌ها")
    'کتاب'
    >>> stem("می‌روم", pos="V")
    'روم'
    """
    return _DEFAULT_STEMMER.stem(word, pos)
