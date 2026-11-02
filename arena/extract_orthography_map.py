#!/usr/bin/env python3
"""Session helper: turn ``parsivar.normalizer.Normalizer.sub_alphabets`` into a
flat ``char -> char`` table.

The reference implementation applies ~80 chained ``re.sub`` calls, one per
letter, which is slow (80 full passes over the document) and hard to audit.
This script walks the method's AST, evaluates the literal patterns/replacements
and emits a single ``dict[str, str]`` that :mod:`farsi_textutils.constants`
stores as a translation table.

Usage:
    python3 arena/extract_orthography_map.py [--out result/_orthography.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import unicodedata
from pathlib import Path

PRESENTATION_RANGES = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF))


def collect_pairs(source: Path) -> list[tuple[str, str]]:
    """Return every (pattern, replacement) pair of ``sub_alphabets`` in order."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "sub_alphabets":
            method = node
            break
    if method is None:
        raise LookupError("sub_alphabets not found")

    values: dict[str, str] = {}
    pairs: list[tuple[str, str]] = []
    for node in method.body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        name, value = target.id, node.value
        # ``xN = r"..."`` literal assignments
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            values[name] = value.value
        # ``sub(naN, nbN, ...)`` call sites define the ordered substitution
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "sub"
            and len(value.args) >= 2
        ):
            a, b = value.args[0], value.args[1]
            if isinstance(a, ast.Name) and isinstance(b, ast.Name):
                pattern, replacement = values.get(a.id), values.get(b.id)
                if pattern is not None and replacement is not None:
                    pairs.append((pattern, replacement))
    return pairs


def flatten(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Expand ``a|b -> c`` alternations into individual character mappings."""
    table: dict[str, str] = {}
    for pattern, replacement in pairs:
        if pattern.startswith("(") and pattern.endswith(")+"):
            continue  # whitespace collapipers are handled separately
        for alternative in pattern.split("|"):
            if not alternative:
                continue
            try:
                char = re.sub(r"\\(.)", r"\1", alternative)
            except re.error:  # pragma: no cover - defensive
                continue
            if len(char) != 1:
                continue
            if len(replacement) > 1:
                continue
            table[char] = replacement
    return table


def _nfkc_base(char: str) -> str | None:
    """Return the canonical base letter of an Arabic presentation form."""
    code = ord(char)
    if not any(low <= code <= high for low, high in PRESENTATION_RANGES):
        return None
    base = unicodedata.normalize("NFKC", char)
    return base if len(base) == 1 and base != char else None


def resolve(table: dict[str, str]) -> dict[str, str]:
    """Make the table idempotent.

    Two upstream defects are repaired here:

    * presentation forms are resolved through their NFKC base letter first, so
      ``ﻧ`` (NOON INITIAL FORM) becomes ``ن`` instead of the ``ذ`` of the
      reference table, which grouped it with ``ﺫ|ﺬ`` by mistake;
    * chained mappings (``ﻱ -> ي -> ی``) are collapsed to a single hop so one
      ``str.translate`` pass is always enough.
    """
    def follow(char: str, target: str) -> str:
        seen = {char, target}
        while target in table and table[target] not in seen:
            target = table[target]
            seen.add(target)
        return target

    resolved = {char: follow(char, _nfkc_base(char) or target) for char, target in table.items()}
    # The reference table only lists a few of the ~700 Arabic presentation
    # forms, so complete the coverage from Unicode's own decomposition.
    for low, high in PRESENTATION_RANGES:
        for code in range(low, high + 1):
            char = chr(code)
            base = _nfkc_base(char)
            if base is None:
                continue
            resolved.setdefault(char, follow(char, base))
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("parsivar/normalizer.py"))
    parser.add_argument("--out", type=Path, default=Path("result/_orthography.json"))
    args = parser.parse_args()

    pairs = collect_pairs(args.source)
    table = resolve(flatten(pairs))
    args.out.write_text(json.dumps(table, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"{len(pairs)} substitution rules -> {len(table)} single-character mappings")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
