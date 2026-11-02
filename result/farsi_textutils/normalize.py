"""Persian text normalisation.

This module merges the three normalisers of the reference packages:

* ``parsivar.normalizer.Normalizer`` -- orthography unification
  (:meth:`~Normalizer.sub_alphabets`), rule based half-space repair
  (:meth:`~Normalizer.space_correction`) and the optional statistical /
  date / pinglish stages;
* ``parsivar.data_helper.DataHelper.clean_text`` -- punctuation spacing and
  removal of characters outside the Persian alphabet;
* ``farsi_tools.standardize_persian_text`` -- the pipeline flavour with
  explicit options (digit styles, newline handling, validity check).

Everything is expressed as small, individually testable steps that compose.

Examples:
    >>> normalize("كتاب‌هاي  من ، خوب است")
    'کتاب‌های من، خوب است'
    >>> normalize("مي روم", fix_space=True)
    'می‌روم'
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

from . import constants as C
from .data.orthography import ORTHOGRAPHY_MAP
from .exceptions import ConfigurationError
from .logging_setup import get_logger
from .mapping import CharFilter, CharFilterChain, CharMapper, FunctionFilter

__all__ = [
    "Normalizer",
    "NormalizerConfig",
    "SpaceCorrector",
    "StandardizeOptions",
    "check_standard_persian_text",
    "clean_text",
    "collapse_whitespace",
    "convert_digits_to_ascii",
    "dediacritize",
    "fix_half_spaces",
    "normalize",
    "remove_extra_spaces",
    "standardize",
    "unify_letters",
]

_log = get_logger("normalize")

_ORTHOGRAPHY: Final[CharMapper] = CharMapper(
    {source: target for source, target in ORTHOGRAPHY_MAP.items() if not source.isdigit()},
    name="persian-orthography",
)
"""Unifies Arabic letter variants and presentation forms.

Digits are mapped separately (:func:`convert_digits_to_ascii`) so that the
``digits_to_ascii`` option of :class:`NormalizerConfig` stays meaningful; the
reference implementation always converted them.
"""

_NUMBER_PLACEHOLDER: Final[str] = "\ue000"
_PROTECTED_TOKENS: Final[re.Pattern[str]] = re.compile(
    r"\d+(?:[.,\u066b]\d+)?(?:[/\-\u060c]\d+(?:[.,\u066b]\d+)?)*"
)
"""Numbers and numeric dates, protected from punctuation spacing.

``3.14`` stays a single token, and so does ``1402/05/12`` -- the reference
implementation spaced the slashes out (``1402 / 05 / 12``)."""


# --------------------------------------------------------------------------
# primitive steps
# --------------------------------------------------------------------------
def dediacritize(text: str) -> str:
    """Remove Arabic/Persian diacritics (harakat, tatweel).

    Ported from ``camel_tools.utils.dediac`` and specialised for Persian.

    >>> dediacritize("كِتَابْ")
    'كتاب'
    """
    return "".join(char for char in text if char not in C.DIACRITICS)


def unify_letters(text: str) -> str:
    """Map Arabic/presentation-form letters and digits onto Persian ones.

    This is the whole of ``Normalizer.sub_alphabets`` reduced to one pass over
    the string (the reference implementation ran ~50 regex substitutions).

    >>> unify_letters("كتاب ٤٢")
    'کتاب ٤٢'
    """
    return _ORTHOGRAPHY.map_string(text)


def convert_digits_to_ascii(text: str) -> str:
    """Convert Persian/Arabic-Indic digits to ASCII digits.

    >>> convert_digits_to_ascii("۱۲۳ ٤٥٦")
    '123 456'
    """
    return text.translate(C.DIGIT_TRANSLATION)


def remove_extra_spaces(text: str) -> str:
    r"""Strip the text and collapse every whitespace run into a single space.

    >>> remove_extra_spaces("  یک   دو \n سه ")
    'یک دو سه'
    """
    return C.RE_ALL_SPACE.sub(" ", text).strip()


def collapse_whitespace(text: str, *, keep_newlines: bool = True) -> str:
    r"""Collapse spaces/tabs, optionally preserving single newlines.

    >>> collapse_whitespace("یک\n\n\nدو")
    'یک\nدو'
    """
    if not keep_newlines:
        return remove_extra_spaces(text)
    text = C.RE_WHITESPACE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    return C.RE_MULTI_NEWLINE.sub("\n", text).strip()


def check_standard_persian_text(text: str, *, includes_latin: bool = False) -> bool | str:
    """Validate that *text* only uses characters of standard Persian.

    Returns ``True`` when the text is clean, otherwise the first offending
    character.

    >>> check_standard_persian_text("سلام دنیا")
    True
    >>> check_standard_persian_text("سلام 😀")
    '😀'
    """
    allowed = C.VALID_TEXT_CHARS_LATIN if includes_latin else C.VALID_TEXT_CHARS
    for char in text:
        if char not in allowed and char not in C.PERSIAN_LETTERS and char != C.HALF_SPACE:
            return char
    return True


# --------------------------------------------------------------------------
# half-space (نیم‌فاصله) repair
# --------------------------------------------------------------------------
def _alternation(items: Iterable[str]) -> str:
    """Build a regex alternation, longest alternative first."""
    ordered = sorted({item for item in items if item}, key=len, reverse=True)
    return "|".join(re.escape(item) for item in ordered)


_RE_LEADING_PREFIX = re.compile(rf"^({_alternation(C.SPACE_MERGE_PREFIXES)}) +", re.MULTILINE)
_RE_INNER_PREFIX = re.compile(rf"( )({_alternation(C.SPACE_MERGE_PREFIXES)}) +")
_RE_INNER_SUFFIX = re.compile(rf"( +)({_alternation(C.SPACE_MERGE_SUFFIXES)})(?=[ ,.!?؟،؛:\n]|$)")
_RE_PARTICIPLE = re.compile(r"( +)(نشده|شده)( +)")
_RE_AGENT = re.compile(rf"( +)({_alternation(C.AGENT_SUFFIXES)})(?=[ ,.!?؟،؛:\n]|$)")
_RE_SPACE_BEFORE_HALFSPACE = re.compile(rf" +{C.HALF_SPACE}")
_RE_SPACE_AFTER_HALFSPACE = re.compile(rf"{C.HALF_SPACE} +")
_RE_DOUBLE_HALFSPACE = re.compile(rf"(?:{C.HALF_SPACE})+")
_RE_LEADING_HALFSPACE = re.compile(rf"^{C.HALF_SPACE}", re.MULTILINE)
_RE_HALFSPACE_BEFORE_NONWORD = re.compile(rf"{C.HALF_SPACE}(?=[^\w{C.HALF_SPACE}]|$)")


def fix_half_spaces(text: str) -> str:
    """Attach prefixes/suffixes that were written with a plain space.

    Rules are the ones from ``Normalizer.space_correction``: a separated ``می``,
    ``نمی`` or ``بی`` prefix and a separated plural / possessive / agent suffix
    are joined with a ZWNJ instead of a space.

    >>> fix_half_spaces("می روم به کتاب خانه ها")
    'می‌روم به کتاب خانه‌ها'
    """
    text = _RE_LEADING_PREFIX.sub(rf"\1{C.HALF_SPACE}", text)
    text = _RE_INNER_PREFIX.sub(rf"\1\2{C.HALF_SPACE}", text)
    text = _RE_INNER_SUFFIX.sub(rf"{C.HALF_SPACE}\2", text)
    text = _RE_PARTICIPLE.sub(rf"{C.HALF_SPACE}\2{C.HALF_SPACE}", text)
    text = _RE_AGENT.sub(rf"{C.HALF_SPACE}\2", text)
    return tidy_half_spaces(text)


def tidy_half_spaces(text: str) -> str:
    """Remove spaces around ZWNJs and drop redundant/leading ZWNJs.

    >>> tidy_half_spaces("کتاب ‌ های ‌هم")
    'کتاب‌های‌هم'
    """
    text = _RE_SPACE_BEFORE_HALFSPACE.sub(C.HALF_SPACE, text)
    text = _RE_SPACE_AFTER_HALFSPACE.sub(C.HALF_SPACE, text)
    text = _RE_DOUBLE_HALFSPACE.sub(C.HALF_SPACE, text)
    text = _RE_LEADING_HALFSPACE.sub("", text)
    return _RE_HALFSPACE_BEFORE_NONWORD.sub("", text)


# --------------------------------------------------------------------------
# punctuation aware cleaning (``DataHelper.clean_text``)
# --------------------------------------------------------------------------
_SPACED_PUNCTUATION: Final[re.Pattern[str]] = re.compile(
    "[" + re.escape(")(}{:؟!،؛»«./<>?.,;%" + C.ASCII_PUNCTUATION.replace("@", "")) + "]+"
)
_DISALLOWED_CHARS: Final[re.Pattern[str]] = re.compile(
    "[^"
    + re.escape(")(}{:؟!-،؛»«.@$&%/<>?.,:;")
    + "a-zA-Z0-9"
    + "آ-ی"
    + C.HALF_SPACE
    + r"\d\s:"
    + re.escape(_NUMBER_PLACEHOLDER)
    + "]"
)


_RE_SPACE_BEFORE_PUNCT: Final[re.Pattern[str]] = re.compile(r"\s+([،؛!؟.:»\)\]}])")
_RE_SPACE_AFTER_OPEN: Final[re.Pattern[str]] = re.compile(r"([«\(\[{])\s+")


def clean_text(text: str, *, new_line_elimination: bool = False) -> str:
    """Normalise punctuation spacing and drop non-Persian characters.

    Ported from ``parsivar.data_helper.DataHelper.clean_text``; numbers are
    protected so that ``3.14`` is not split around its decimal point.

    >>> clean_text("او گفت(سلام)!")
    'او گفت (سلام)!'
    """
    numbers = _PROTECTED_TOKENS.findall(text)
    placeholder_text = _PROTECTED_TOKENS.sub(_NUMBER_PLACEHOLDER, text)
    placeholder_text = _SPACED_PUNCTUATION.sub(lambda m: f" {m.group()} ", placeholder_text)
    placeholder_text = _DISALLOWED_CHARS.sub("", placeholder_text)
    placeholder_text = C.RE_WHITESPACE.sub(" ", placeholder_text)
    if new_line_elimination:
        placeholder_text = C.RE_ALL_SPACE.sub(" ", placeholder_text)
    placeholder_text = C.RE_MULTI_NEWLINE.sub("\n", placeholder_text)

    restored = []
    index = 0
    for char in placeholder_text:
        if char == _NUMBER_PLACEHOLDER and index < len(numbers):
            restored.append(numbers[index])
            index += 1
        else:
            restored.append(char)
    cleaned = "".join(restored)
    cleaned = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    cleaned = _RE_SPACE_AFTER_OPEN.sub(r"\1", cleaned)
    return cleaned.strip()


# --------------------------------------------------------------------------
# statistical space correction (replaces the pickled chunk parser)
# --------------------------------------------------------------------------
@dataclass(slots=True)
class SpaceCorrector:
    """Merge/split mis-spaced words using a word-frequency lexicon.

    The reference implementation trained a ``ClassifierBasedTagger`` (NLTK) and
    pickled it to ``TokenMerger.pckl`` (~50 MB, NLTK dependency).  This class
    reaches comparable quality for the ``می‌روم`` / ``کتاب‌ها`` cases with a
    dynamic-programming segmentation over a plain ``dict[str, int]`` lexicon.

    Args:
        lexicon: Word frequencies; words missing from the lexicon score ``-inf``.
        max_join: Maximum number of whitespace separated tokens joined at once.
        unknown_penalty: Cost of an unknown word; higher means fewer merges.
    """

    lexicon: Mapping[str, int]
    max_join: int = 3
    unknown_penalty: float = 12.0
    half_space: str = C.HALF_SPACE
    _total: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        if self.max_join < 1:
            raise ConfigurationError("max_join must be >= 1")
        self._total = sum(self.lexicon.values()) or 1

    def _score(self, word: str) -> float:
        """Negative log frequency of *word*, or a fixed penalty when unknown."""
        import math

        count = self.lexicon.get(word, 0)
        if count <= 0:
            return -self.unknown_penalty
        return math.log(count / self._total)

    def correct(self, text: str) -> str:
        """Repair spacing of every whitespace separated group in *text*.

        >>> corrector = SpaceCorrector({"می‌روم": 5, "کتاب‌ها": 7, "خانه": 3})
        >>> corrector.correct("می روم کتاب ها")
        'می‌روم کتاب‌ها'
        """
        sentences: list[str] = []
        for line in text.split("\n"):
            tokens = line.split()
            words: list[str] = []
            index = 0
            while index < len(tokens):
                window = tokens[index : index + self.max_join]
                best, consumed = window[0], 1
                best_score = self._score(window[0])
                for size in range(2, len(window) + 1):
                    candidate = self._join(window[:size])
                    score = self._score(candidate) - 0.5 * (size - 1)
                    if score > best_score:
                        best, best_score, consumed = candidate, score, size
                words.append(best)
                index += consumed
            sentences.append(" ".join(words))
        return "\n".join(sentences)

    def _join(self, tokens: Sequence[str]) -> str:
        """Join *tokens* with the half-space, keeping an existing ZWNJ."""
        joined = self.half_space.join(part for part in tokens if part)
        return joined.strip(self.half_space)


# --------------------------------------------------------------------------
# the normaliser
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class NormalizerConfig:
    """Options of :class:`Normalizer`.

    Attributes:
        unify_orthography: Map Arabic variants / presentation forms to Persian.
        remove_diacritics: Drop harakat and tatweel.
        fix_space: Rule based half-space repair (``می روم`` -> ``می‌روم``).
        statistical_space_correction: Run :class:`SpaceCorrector` as well; needs
            a trained corrector to be supplied to the normaliser.
        date_normalization: Convert dates and number words to a canonical form.
        pinglish_conversion: Convert Persian written in Latin script (pinglish).
        new_line_elimination: Turn newlines into spaces during cleaning.
        keep_numbers: Protect and restore numeric tokens while cleaning.
        normalize_punctuation: Replace ASCII punctuation with Persian forms.
        digits_to_ascii: Convert Persian/Arabic digits to ASCII (off by default
            because outputs are usually meant for Persian readers).
    """

    unify_orthography: bool = True
    remove_diacritics: bool = True
    fix_space: bool = True
    statistical_space_correction: bool = False
    date_normalization: bool = False
    pinglish_conversion: bool = False
    new_line_elimination: bool = False
    keep_numbers: bool = True
    normalize_punctuation: bool = True
    digits_to_ascii: bool = False

    def with_options(self, **overrides: object) -> NormalizerConfig:
        """Return a copy of the config with *overrides* applied.

        >>> NormalizerConfig().with_options(fix_space=False).fix_space
        False
        """
        unknown = set(overrides) - set(self.__dataclass_fields__)
        if unknown:
            raise ConfigurationError(f"unknown normaliser option(s): {sorted(unknown)}")
        return replace(self, **overrides)  # type: ignore[arg-type]


_PUNCTUATION_MAP: Final[CharMapper] = CharMapper(C.PUNCTUATION_EQUIVALENTS, name="punctuation")


class Normalizer:
    """Configurable Persian normaliser.

    Args:
        config: Behaviour flags; defaults to :class:`NormalizerConfig`.
        space_corrector: Required for ``statistical_space_correction``.

    Example:
        >>> normalizer = Normalizer()
        >>> normalizer.normalize("كتاب هاي  من")
        'کتاب‌های من'
    """

    __slots__ = ("_config", "_dates", "_pinglish", "_pipeline", "_space_corrector", "_tokenizer")

    def __init__(
        self,
        config: NormalizerConfig | None = None,
        *,
        space_corrector: SpaceCorrector | None = None,
    ) -> None:
        self._config = config or NormalizerConfig()
        if self._config.statistical_space_correction and space_corrector is None:
            raise ConfigurationError("statistical_space_correction requires a SpaceCorrector instance")
        self._space_corrector = space_corrector
        self._tokenizer = None
        self._dates = None
        self._pinglish = None

        filters: list[CharFilter] = []
        if self._config.unify_orthography:
            filters.append(FunctionFilter(self.sub_alphabets, name="orthography"))
        if self._config.remove_diacritics:
            filters.append(FunctionFilter(dediacritize, name="diacritics"))
        if self._config.normalize_punctuation:
            filters.append(FunctionFilter(_PUNCTUATION_MAP.map_string, name="punctuation"))
        self._pipeline = CharFilterChain(filters)

    # --- properties -------------------------------------------------------
    @property
    def config(self) -> NormalizerConfig:
        """The immutable configuration of this normaliser."""
        return self._config

    @property
    def pipeline(self) -> CharFilterChain:
        """The character-filter chain applied before the word-level rules."""
        return self._pipeline

    # --- compatibility shims ---------------------------------------------
    def sub_alphabets(self, text: str) -> str:
        """Unify the alphabet (name kept for parity with ``parsivar``)."""
        return unify_letters(text)

    def space_correction(self, text: str) -> str:
        """Rule based half-space repair (name kept for parity with ``parsivar``)."""
        return fix_half_spaces(text)

    def clean_text(self, text: str, *, new_line_elimination: bool | None = None) -> str:
        """Apply :func:`clean_text` using the configured newline policy."""
        if new_line_elimination is None:
            new_line_elimination = self._config.new_line_elimination
        return clean_text(text, new_line_elimination=new_line_elimination)

    # --- main entry points -----------------------------------------------
    def normalize(self, text: str, *, new_line_elimination: bool | None = None) -> str:
        """Normalise *text* end to end.

        Args:
            text: Input document.
            new_line_elimination: Overrides ``config.new_line_elimination``;
                the reference signature kept this as a per-call flag.

        Returns:
            The normalised document.
        """
        if not text:
            return ""
        result = self._pipeline.apply(text)
        elimination = bool(new_line_elimination or self._config.new_line_elimination)
        result = clean_text(result, new_line_elimination=elimination)

        if self._config.statistical_space_correction and self._space_corrector is not None:
            result = self._space_corrector.correct(result)
            result = clean_text(result, new_line_elimination=self._config.new_line_elimination)

        if self._config.fix_space:
            result = fix_half_spaces(result)

        if self._config.pinglish_conversion:
            result = self._convert_pinglish(result)

        if self._config.date_normalization:
            result = self._normalize_dates(result)

        if self._config.digits_to_ascii:
            result = convert_digits_to_ascii(result)

        return collapse_whitespace(result, keep_newlines=not self._config.new_line_elimination)

    def __call__(self, text: str) -> str:
        """Normalise *text* (same as :meth:`normalize`)."""
        return self.normalize(text)

    # --- lazy optional stages --------------------------------------------
    def _convert_pinglish(self, text: str) -> str:
        from .transliterate import to_persian

        return to_persian(text)

    def _normalize_dates(self, text: str) -> str:
        from .dates import DateNormalizer, normalize_numbers
        from .tokenize import word_tokenize

        tokens = normalize_numbers(word_tokenize(text))
        return DateNormalizer().normalize_dates(tokens)


def normalize(text: str, *, config: NormalizerConfig | None = None, **overrides: object) -> str:
    """Convenience wrapper around :class:`Normalizer`.

    >>> normalize("سلام   دنيا!")
    'سلام دنیا!'
    >>> normalize("مي روم", fix_space=True)
    'می‌روم'
    """
    resolved = (config or NormalizerConfig()).with_options(**overrides)
    return Normalizer(resolved).normalize(text)


# --------------------------------------------------------------------------
# ``farsi_tools`` style pipeline
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StandardizeOptions:
    """Options of :func:`standardize` (subset of ``standardize_persian_text``)."""

    may_include_latin_chars: bool = False
    remove_spaces: bool = True
    check_persian: bool = False
    replace_ascii_digits: bool = True
    remove_newlines: bool = False
    remove_double_newlines: bool = False
    fix_half_spaces: bool = True
    remove_diacritics: bool = True
    unify_orthography: bool = True


def standardize(
    text: str | None,
    *,
    options: StandardizeOptions | None = None,
    **overrides: object,
) -> str | None:
    """Standardise Persian text the way ``farsi_tools`` did.

    Args:
        text: Input text; ``None`` short-circuits to ``None``.
        options: Pipeline options.
        **overrides: Field overrides for *options*.

    Returns:
        The standardised text, or ``None`` when *text* is ``None``.

    Raises:
        InvalidValueError: When ``check_persian`` is enabled and an invalid
            character is found.

    >>> standardize("كتابهاي  من", replace_ascii_digits=False)
    'کتابهای من'
    """
    if text is None:
        return None
    opts = options or StandardizeOptions()
    if overrides:
        unknown = set(overrides) - set(StandardizeOptions.__dataclass_fields__)
        if unknown:
            raise ConfigurationError(f"unknown standardize option(s): {sorted(unknown)}")
        opts = replace(opts, **overrides)  # type: ignore[arg-type]

    result = text
    if opts.unify_orthography:
        result = unify_letters(result)
    if opts.remove_diacritics:
        result = dediacritize(result)
    if opts.fix_half_spaces:
        result = tidy_half_spaces(fix_half_spaces(result))

    if opts.remove_spaces:
        if opts.remove_newlines:
            result = C.RE_ALL_SPACE.sub(" ", result)
        else:
            result = C.RE_WHITESPACE.sub(" ", result)
            result = re.sub(r" *\n *", "\n", result)
        result = result.strip()
        if opts.remove_double_newlines:
            result = C.RE_MULTI_NEWLINE.sub("\n", result)

    if opts.replace_ascii_digits:
        result = result.translate(C.ASCII_TO_PERSIAN_DIGITS)

    if opts.check_persian:
        verdict = check_standard_persian_text(result, includes_latin=opts.may_include_latin_chars)
        if verdict is not True:
            from .exceptions import InvalidValueError

            name = unicodedata.name(str(verdict), "UNKNOWN")
            raise InvalidValueError(
                f"invalid character {verdict!r} (U+{ord(str(verdict)):04X}, {name}) in text {result!r}"
            )
    return result
