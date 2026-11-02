"""Reusable character-mapping primitives.

Extracted from ``camel_tools.utils.charmap.CharMapper`` and
``camel_tools.utils.transliterate.Transliterator``: instead of a mapping object
that carries an ``@@IGNORE@@`` marker convention, we expose a small
:class:`CharMapper` plus a :class:`CharFilter` chain (the pattern comes from
``janome.charfilter``).  The chain keeps every normalisation step a composable,
testable callable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Final, Literal

__all__ = [
    "CharFilter",
    "CharFilterChain",
    "CharMapper",
    "FunctionFilter",
    "UnicodeNormalizeFilter",
]


class CharMapper(Mapping[str, str]):
    """An immutable, translation-table backed character mapper.

    >>> mapper = CharMapper({"ي": "ی", "ك": "ک"})
    >>> mapper.map_string("كتاب")
    'کتاب'
    """

    __slots__ = ("_mapping", "_table", "name")

    def __init__(self, mapping: Mapping[str, str], *, name: str | None = None) -> None:
        if not isinstance(mapping, Mapping):
            raise TypeError("mapping must implement the Mapping protocol")
        for source, target in mapping.items():
            if len(source) != 1:
                raise ValueError(f"source keys must be single characters, got {source!r}")
            if not isinstance(target, str):
                raise TypeError(f"target value for {source!r} must be str, got {type(target)!r}")
        self._mapping: Final[Mapping[str, str]] = dict(mapping)
        self._table: Final[dict[int, str]] = str.maketrans(self._mapping)
        self.name = name or "char-mapper"

    # --- Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> str:
        return self._mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    # --- API --------------------------------------------------------------
    def map_string(self, text: str) -> str:
        """Apply the mapping to *text* in a single pass."""
        return text.translate(self._table)

    def map_chars(self, text: str) -> list[str]:
        """Return the mapped characters of *text* as a list (unmapped chars kept)."""
        return [self._mapping.get(char, char) for char in text]

    def concatenated(self, other: CharMapper) -> CharMapper:
        """Return a mapper equivalent to ``self`` applied before *other*."""
        combined = dict(self._mapping)
        combined.update({key: other.map_string(value) for key, value in combined.items()})
        for key, value in other._mapping.items():
            combined.setdefault(key, value)
        return CharMapper(combined, name=f"{self.name}+{other.name}")

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, str]], *, name: str | None = None) -> CharMapper:
        """Build a mapper from ``(source, target)`` pairs."""
        return cls(dict(pairs), name=name)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"CharMapper(name={self.name!r}, entries={len(self._mapping)})"


class CharFilter(ABC):
    """A text-to-text transformation usable inside a :class:`CharFilterChain`."""

    @abstractmethod
    def apply(self, text: str) -> str:
        """Transform *text*."""

    def __call__(self, text: str) -> str:
        """Make every filter callable (delegates to :meth:`apply`)."""
        return self.apply(text)

    def __or__(self, other: CharFilter) -> CharFilterChain:
        return CharFilterChain((self, other))


class FunctionFilter(CharFilter):
    """Adapter that turns any ``Callable[[str], str]`` into a :class:`CharFilter`."""

    __slots__ = ("_function", "name")

    def __init__(self, function: Callable[[str], str], *, name: str | None = None) -> None:
        self._function = function
        self.name = name or getattr(function, "__name__", "function-filter")

    def apply(self, text: str) -> str:
        """Call the wrapped function."""
        return self._function(text)


class MappingFilter(CharFilter):  # pragma: no cover - thin wrapper, covered indirectly
    """A :class:`CharFilter` backed by a :class:`CharMapper`."""

    __slots__ = ("mapper",)

    def __init__(self, mapper: CharMapper) -> None:
        self.mapper = mapper

    def apply(self, text: str) -> str:
        return self.mapper.map_string(text)


class UnicodeNormalizeFilter(CharFilter):
    """Apply a Unicode normal form (``NFKC`` by default)."""

    __slots__ = ("form",)

    def __init__(self, form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC") -> None:
        self.form = form

    def apply(self, text: str) -> str:
        """Normalise *text* to :attr:`form`."""
        import unicodedata

        return unicodedata.normalize(self.form, text)


class CharFilterChain(CharFilter):
    """Run several filters in order.

    >>> chain = CharFilterChain((FunctionFilter(str.strip), FunctionFilter(str.lower)))
    >>> chain("  BOOK  ")
    'book'
    """

    __slots__ = ("filters",)

    def __init__(self, filters: Iterable[CharFilter] = ()) -> None:
        self.filters: tuple[CharFilter, ...] = tuple(filters)

    def apply(self, text: str) -> str:
        """Run every filter of the chain in order."""
        for char_filter in self.filters:
            text = char_filter(text)
        return text

    def then(self, other: CharFilter) -> CharFilterChain:
        """Return a new chain with *other* appended."""
        return CharFilterChain((*self.filters, other))

    def __iter__(self) -> Iterator[CharFilter]:
        return iter(self.filters)

    def __len__(self) -> int:
        return len(self.filters)
