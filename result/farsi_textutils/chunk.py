"""Shallow parsing (chunking) with a miniature regex grammar engine.

``parsivar.chunker.FindChunks`` relied on ``nltk.RegexpParser`` and
``parsivar.token_merger.ClassifierChunkParser`` on ``nltk.ClassifierBasedTagger``
plus a pickled model.  Both dependencies are gone here: this module implements
the small subset of the NLTK chunk-grammar language that the Persian grammar
actually uses --

* terminals are tag names, ``*`` wildcards (``V_.*``) and ``.`` (any tag);
* quantifiers ``?``, ``*`` and ``+``;
* rules are applied in order, left to right, without overlap;
* the label of a matched span can be reused by later rules (``{<DET><NP>}``).

Example:
    >>> chunker = RegexpChunker()
    >>> chunker.chunk_text("کتاب خوب است")
    '[کتاب/N_SING NP] [خوب/ADJ_SIM است/V_PRS VP]'
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

from .logging_setup import get_logger
from .pos import POSTagger, TaggedToken

__all__ = [
    "DEFAULT_PERSIAN_GRAMMAR",
    "Chunk",
    "ChunkParser",
    "GrammarRule",
    "Leaf",
    "Node",
    "RegexpChunker",
    "chunk_tokens",
    "to_bracket_string",
]

_log = get_logger("chunk")


@dataclass(frozen=True, slots=True)
class Leaf:
    """A tagged token (a grammar terminal).

    >>> Leaf("کتاب", "N_SING").label
    'N_SING'
    """

    token: str
    tag: str

    @property
    def label(self) -> str:
        """Label used by the grammar engine: the POS tag."""
        return self.tag


@dataclass(frozen=True, slots=True)
class Chunk:
    """A chunk node produced by a grammar rule.

    >>> Chunk("NP", (Leaf("کتاب", "N_SING"),)).label
    'NP'
    """

    label: str
    children: tuple[Node, ...]

    def tokens(self) -> list[str]:
        """Surface tokens of the chunk, in order.

        >>> Chunk("NP", (Leaf("کتاب", "N_SING"),)).tokens()
        ['کتاب']
        """
        result: list[str] = []
        for child in self.children:
            if isinstance(child, Leaf):
                result.append(child.token)
            else:
                result.extend(child.tokens())
        return result

    def text(self) -> str:
        """The chunk as a whitespace joined string."""
        return " ".join(self.tokens())


Node = Leaf | Chunk
"""A parse node: either a tagged :class:`Leaf` or a :class:`Chunk`."""


@dataclass(frozen=True, slots=True)
class GrammarRule:
    """One ``LABEL: {<TAG><TAG>?}`` rule.

    Attributes:
        label: Chunk label (``NP``, ``VP``, ``PP`` ...).
        pattern: Text between the braces, e.g. ``<DET><NP>``.
    """

    label: str
    pattern: str
    _parts: tuple[tuple[re.Pattern[str], str, str], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        parts = tuple(_parse_pattern(self.pattern))
        if not parts:
            raise ValueError(f"empty grammar pattern for {self.label!r}")
        object.__setattr__(self, "_parts", parts)

    @property
    def parts(self) -> tuple[tuple[re.Pattern[str], str, str], ...]:
        """The compiled ``(tag-regex, quantifier, raw-tag)`` sequence."""
        return self._parts


def _parse_pattern(pattern: str) -> Iterable[tuple[re.Pattern[str], str, str]]:
    """Turn ``<TAG>?</TAG>`` markup into ``(regex, quantifier, raw_tag)`` triples."""
    for match in re.finditer(r"<([^>]+)>\s*([?*+]?)", pattern):
        tag, quantifier = match.group(1), match.group(2)
        yield re.compile(rf"^(?:{tag})$"), quantifier, tag


DEFAULT_PERSIAN_GRAMMAR: Final[tuple[GrammarRule, ...]] = (
    GrammarRule("VP", "<ADJ_SIM><V_PRS>"),
    GrammarRule("VP", "<ADJ_INO><V.*>"),
    GrammarRule("VP", "<V_PRS><N_SING><V_SUB>"),
    GrammarRule("NP", "<N_SING><ADJ.*><N_SING>"),
    GrammarRule("NP", "<N.*><PRO>"),
    GrammarRule("VP", "<N_SING><V_.*>"),
    GrammarRule("VP", "<V.*>+"),
    GrammarRule("NP", "<ADJ.*>?<N.*>+<ADJ.*>?"),
    GrammarRule("DNP", "<DET><NP>"),
    GrammarRule("PP", "<ADJ_CMPR><P>"),
    GrammarRule("PP", "<ADJ_SIM><P>"),
    GrammarRule("PP", "<P><N_SING>"),
    GrammarRule("PP", "<P>*"),
    GrammarRule("DDNP", "<NP><DNP>"),
    GrammarRule("NPP", "<PP><NP>+"),
)
"""The grammar of ``parsivar.chunker.FindChunks``, rule for rule."""

_MAX_QUANTIFIER: Final[dict[str, int]] = {"": 1, "?": 1, "*": 64, "+": 64}


def _node_matches(node: Node, pattern: re.Pattern[str], raw_tag: str) -> bool:
    """Match a grammar terminal against a node.

    A chunk node can only be referenced by its *exact* label (``<NP>``,
    ``<PP>``), mirroring NLTK: a wildcard terminal such as ``<V.*>`` must never
    match the ``VP`` chunk label produced by an earlier rule.
    """
    if isinstance(node, Chunk):
        return node.label == raw_tag
    return pattern.match(node.tag) is not None


class RegexpChunker:
    """Apply a chunk grammar to tagged tokens.

    Args:
        grammar: Rules applied in order; defaults to
            :data:`DEFAULT_PERSIAN_GRAMMAR`.
        tagger: Tagger used by :meth:`chunk_text`.

    Example:
        >>> chunker = RegexpChunker()
        >>> tree = chunker.parse([("کتاب", "N_SING"), ("خوب", "ADJ_SIM")])
        >>> to_bracket_string(tree)
        '[کتاب/N_SING خوب/ADJ_SIM NP]'
    """

    __slots__ = ("grammar", "tagger")

    def __init__(
        self,
        grammar: Sequence[GrammarRule] | None = None,
        *,
        tagger: POSTagger | None = None,
    ) -> None:
        self.grammar: tuple[GrammarRule, ...] = tuple(grammar or DEFAULT_PERSIAN_GRAMMAR)
        self.tagger = tagger or POSTagger()

    # --- public API -------------------------------------------------------
    def parse(self, tagged_tokens: Iterable[TaggedToken]) -> list[Node]:
        """Chunk a tagged sentence, returning the top level nodes."""
        nodes: list[Node] = [Leaf(token, tag) for token, tag in tagged_tokens]
        for rule in self.grammar:
            nodes = self._apply(rule, nodes)
        return nodes

    def chunk_text(self, text: str) -> str:
        """Tag, chunk and render *text* as a bracket string."""
        return to_bracket_string(self.parse(self.tagger.tag_text(text)))

    def chunks(self, tagged_tokens: Iterable[TaggedToken], label: str | None = None) -> list[Chunk]:
        """Return every :class:`Chunk` of the parse, optionally filtered by label.

        >>> chunker = RegexpChunker()
        >>> sorted({c.label for c in chunker.chunks([("به", "P"), ("خانه", "N_SING")])})
        ['NP', 'NPP', 'PP']
        """
        found: list[Chunk] = []

        def walk(nodes: Iterable[Node]) -> None:
            for node in nodes:
                if isinstance(node, Chunk):
                    if label is None or node.label == label:
                        found.append(node)
                    walk(node.children)

        walk(self.parse(tagged_tokens))
        return found

    # --- grammar engine ---------------------------------------------------
    def _apply(self, rule: GrammarRule, nodes: Sequence[Node]) -> list[Node]:
        """Apply one rule left to right, replacing matched spans by chunks."""
        output: list[Node] = []
        index = 0
        while index < len(nodes):
            end = self._match(rule, nodes, index)
            if end is None:
                output.append(nodes[index])
                index += 1
            else:
                output.append(Chunk(rule.label, tuple(nodes[index:end])))
                index = end
        return output

    def _match(self, rule: GrammarRule, nodes: Sequence[Node], start: int) -> int | None:
        """Return the exclusive end index of a match starting at *start*."""
        cursor = start
        for pattern, quantifier, raw_tag in rule.parts:
            matched = 0
            limit = _MAX_QUANTIFIER[quantifier]
            while matched < limit and cursor < len(nodes) and _node_matches(nodes[cursor], pattern, raw_tag):
                cursor += 1
                matched += 1
            minimum = 0 if quantifier in {"?", "*"} else 1
            if matched < minimum:
                return None
        return cursor if cursor > start else None


ChunkParser = RegexpChunker
"""Alias kept for readers used to ``nltk.chunk.ChunkParserI``."""


def chunk_tokens(
    tagged_tokens: Iterable[TaggedToken],
    *,
    grammar: Sequence[GrammarRule] | None = None,
) -> list[Node]:
    """Chunk *tagged_tokens* with the default grammar.

    >>> to_bracket_string(chunk_tokens([("به", "P"), ("خانه", "N_SING")]))
    '[[به/P PP] [خانه/N_SING NP] NPP]'
    """
    return RegexpChunker(grammar).parse(tagged_tokens)


def to_bracket_string(nodes: Iterable[Node]) -> str:
    """Render a parse as an NLTK-like bracket string.

    >>> to_bracket_string([Chunk("NP", (Leaf("کتاب", "N_SING"),))])
    '[کتاب/N_SING NP]'
    """
    pieces: list[str] = []
    for node in nodes:
        if isinstance(node, Leaf):
            pieces.append(f"{node.token}/{node.tag}")
        else:
            pieces.append(f"[{to_bracket_string(node.children)} {node.label}]")
    return " ".join(piece for piece in pieces if piece)


def to_plain_text(nodes: Iterable[Node]) -> str:
    """Render a parse as plain text without labels.

    >>> to_plain_text([Leaf("سلام", "N_SING"), Leaf("دنیا", "N_SING")])
    'سلام دنیا'
    """
    tokens: list[str] = []
    for node in nodes:
        tokens.extend([node.token] if isinstance(node, Leaf) else node.tokens())
    return " ".join(tokens)


def chunk_tags(tagged_tokens: Iterable[TaggedToken]) -> list[str]:
    """Return the chunk labels of a parse in surface order."""
    return [node.label for node in RegexpChunker().parse(tagged_tokens)]
