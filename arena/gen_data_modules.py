#!/usr/bin/env python3
"""Session helper: port the frozen data tables of the source packages into
``result/farsi_textutils/data/`` as typed, dependency-free Python modules.

The source packages keep their reference data as module level literals
(``persian_tools``, ``fakefarsi``) plus a few ``dict``s holding callables
(``persian_tools.bank.banks_code``).  This script reads those literals with
``ast``/``importlib`` and re-emits them as ``Final`` annotated tuples/mappings
so the result package never has to import the original projects.

Usage:
    python3 arena/gen_data_modules.py [--src .] [--out result/farsi_textutils/data]

Only the standard library is used on purpose: this script must run before the
result package exists.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import pprint
import sys
from pathlib import Path
from typing import Any, Iterable

HEADER = '''"""{doc}

Ported verbatim (data only) from the reference implementation; the surrounding
logic was rewritten for :mod:`farsi_textutils` -- see ``result/README.md`` for the
provenance table.
"""

from __future__ import annotations

from typing import Final
'''

pp = pprint.PrettyPrinter(indent=0, width=100, sort_dicts=False, compact=True)


def _render(value: Any) -> str:
    """Render a literal with a stable, readable layout."""
    text = pp.pformat(value)
    if len(text) > 100 or "\n" in text:
        text = pprint.PrettyPrinter(indent=4, width=96, sort_dicts=False).pformat(value)
    return text


def _literal_data(path: Path, name: str = "data") -> Any:
    """Extract a module level literal assignment without importing the module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise LookupError(f"{name!r} literal not found in {path}")


def _module_data(module: str, attribute: str = "data") -> Any:
    """Import a light module and read an attribute (used when it holds callables)."""
    return getattr(importlib.import_module(module), attribute)


def _banks(src: Path) -> tuple[dict[str, Any], ...]:
    """Bank records, with account-number calculators reduced to their names."""
    tree = ast.parse((src / "persian_tools/bank/banks_code.py").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
            continue
        if node.targets[0].id != "data":
            continue
        for element in node.value.elts:  # type: ignore[attr-defined]
            record: dict[str, Any] = {}
            for key, value_node in zip(element.keys, element.values, strict=True):
                key_str = ast.literal_eval(key)
                if isinstance(value_node, ast.Name):
                    record[key_str] = value_node.id
                else:
                    record[key_str] = ast.literal_eval(value_node)
            records.append(record)
    if not records:
        raise LookupError("no bank records extracted")
    return tuple(records)


def _callables(src: Path) -> tuple[str, int]:
    """Extract the account-number calculator bodies as source (for manual porting)."""
    text = (src / "persian_tools/bank/banks_code.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    return "\n".join(names), len(names)


PROVINCE_TYPOS: dict[str, str] = {
    # the upstream plate dataset misspells Isfahan
    "اصفحان": "اصفهان",
}


def fix_provinces(data: dict[str, dict[int, str]]) -> dict[str, dict[int, str]]:
    """Repair the known typos of the plate province table."""
    return {
        category: {code: PROVINCE_TYPOS.get(name, name) for code, name in table.items()}
        for category, table in data.items()
    }


def write(path: Path, doc: str, body: str) -> None:
    path.write_text(HEADER.format(doc=doc) + "\n" + body + "\n", encoding="utf-8")
    print(f"  wrote {path.relative_to(path.parents[2])} ({len(body.splitlines())} body lines)")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--out", type=Path, default=Path("result/farsi_textutils/data")
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    src: Path = args.src.resolve()
    out: Path = (src / args.out if not args.out.is_absolute() else args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(src))

    print(f"source: {src}")
    print(f"output: {out}")

    # --- bank metadata -----------------------------------------------------
    names, count = _callables(src)
    print(f"  bank calculators found: {count} -> {names}")
    write(
        out / "banks.py",
        "Iranian bank metadata: name, card prefix(es) and SHEBA (IBAN) code.",
        'BANK_RECORD_KEYS: Final[tuple[str, ...]] = (\n'
        '    "nickname",\n    "name",\n    "persian_name",\n    "card_prefix",\n'
        '    "sheba_code",\n    "account_number_calculator",\n)\n\n'
        f"BANKS: Final[tuple[dict[str, object], ...]] = {_render(_banks(src))}\n",
    )

    # --- national-id place codes ------------------------------------------
    write(
        out / "places.py",
        "National-ID registration place codes (city + province).",
        f"PLACES: Final[tuple[dict[str, str], ...]] = {_render(_literal_data(src / 'persian_tools/national_id/places_code.py'))}\n",
    )

    # --- mobile operators --------------------------------------------------
    write(
        out / "operators.py",
        "Iranian mobile numbering plan: prefixes, operator, province and SIM type.",
        f"OPERATOR_NAMES: Final[dict[str, str]] = {_render(_module_data('persian_tools.phone_number.operators', 'OPERATORS'))}\n\n"
        f"MOBILE_PREFIXES: Final[dict[str, dict[str, object]]] = {_render(_module_data('persian_tools.phone_number.operators', 'data'))}\n",
    )

    # --- licence plates ----------------------------------------------------
    plate_mod = importlib.import_module("persian_tools.plate.dataset")
    write(
        out / "plate.py",
        "Iranian licence plate reference data: provinces per plate type and categories.",
        "TYPE_CAR: Final[str] = "
        f"{plate_mod.TYPE_CAR!r}\n"
        "TYPE_MOTORCYCLE: Final[str] = "
        f"{plate_mod.TYPE_MOTORCYCLE!r}\n\n"
        f"PROVINCES: Final[dict[str, dict[int, str]]] = {_render(fix_provinces(plate_mod.provinces))}\n\n"
        f"CATEGORIES: Final[dict[str, str]] = {_render(plate_mod.categories)}\n",
    )

    # --- persian given/family names ---------------------------------------
    names_mod = importlib.import_module("fakefarsi.data")
    write(
        out / "names.py",
        "Persian given names and family names used by :mod:`farsi_textutils.fake`.",
        f"MALE_NAMES: Final[tuple[str, ...]] = {_render(tuple(names_mod.male_names))}\n\n"
        f"FEMALE_NAMES: Final[tuple[str, ...]] = {_render(tuple(names_mod.female_names))}\n\n"
        f"FAMILY_NAMES: Final[tuple[str, ...]] = {_render(tuple(names_mod.l_names))}\n",
    )

    # --- number words ------------------------------------------------------
    digits_mod = importlib.import_module("persian_tools.digits.strings")
    write(
        out / "numbers.py",
        "Persian number-word tables shared by :mod:`farsi_textutils.digits`.",
        "".join(
            f"{name}: Final[{annotation}] = "
            f"{_render(tuple(getattr(digits_mod, name)) if annotation.startswith('tuple') else getattr(digits_mod, name))}\n\n"
            for name, annotation in (
                ("SCALE", "tuple[str, ...]"),
                ("NUMBER_TEXT", "dict[int, str]"),
                ("UNITS", "dict[str, int]"),
                ("TEN", "dict[str, int]"),
                ("MAGNITUDE", "dict[str, int]"),
                ("TYPO_LIST", "dict[str, str]"),
                ("JOINERS", "tuple[str, ...]"),
                ("PREFIXES", "tuple[str, ...]"),
            )
        ).rstrip(),
    )

    # --- stop words --------------------------------------------------------
    farsi_tools = importlib.import_module("farsi_tools")
    words = tuple(dict.fromkeys(farsi_tools.stop_words()))
    write(
        out / "stopwords.py",
        "Persian stop words / punctuation set (from the ``farsi_tools`` reference package).",
        f"STOP_WORDS: Final[frozenset[str]] = frozenset({_render(sorted(words))})\n",
    )

    # --- orthography table -------------------------------------------------
    ortho_json = src / "result/_orthography.json"
    if ortho_json.exists():
        table = json.loads(ortho_json.read_text(encoding="utf-8"))
        extra = {
            "٥": "5",
            "۰": "0",
            "۱": "1",
            "۲": "2",
            "۳": "3",
            "۴": "4",
            "۵": "5",
            "۶": "6",
            "۷": "7",
            "۸": "8",
            "۹": "9",
        }
        table = {**table, **extra}
        write(
            out / "orthography.py",
            "Single-character orthography normalisation table "
            "(Persian/Arabic presentation forms, digits and diacritics).",
            "".join(
                f"{name}: Final[dict[str, str]] = {_render(table)}\n"
                for name in ("ORTHOGRAPHY_MAP",)
            ).rstrip(),
        )
    else:
        print(f"  ! {ortho_json} missing - run arena/extract_orthography_map.py first")

    # --- package init ------------------------------------------------------
    (out / "__init__.py").write_text(
        '''"""Static reference data ported from the source packages.

Every table here is plain, immutable Python data so that importing
:mod:`farsi_textutils.data` never pulls third-party dependencies.
"""

from __future__ import annotations

from .banks import BANK_RECORD_KEYS, BANKS
from .names import FAMILY_NAMES, FEMALE_NAMES, MALE_NAMES
from .numbers import (
    JOINERS,
    MAGNITUDE,
    NUMBER_TEXT,
    PREFIXES,
    SCALE,
    TEN,
    TYPO_LIST,
    UNITS,
)
from .operators import MOBILE_PREFIXES, OPERATOR_NAMES
from .orthography import ORTHOGRAPHY_MAP
from .places import PLACES
from .plate import CATEGORIES, PROVINCES, TYPE_CAR, TYPE_MOTORCYCLE
from .stopwords import STOP_WORDS

__all__ = [
    "BANK_RECORD_KEYS",
    "BANKS",
    "CATEGORIES",
    "FAMILY_NAMES",
    "FEMALE_NAMES",
    "JOINERS",
    "MAGNITUDE",
    "MALE_NAMES",
    "MOBILE_PREFIXES",
    "NUMBER_TEXT",
    "OPERATOR_NAMES",
    "ORTHOGRAPHY_MAP",
    "PLACES",
    "PREFIXES",
    "PROVINCES",
    "SCALE",
    "STOP_WORDS",
    "TEN",
    "TYPE_CAR",
    "TYPE_MOTORCYCLE",
    "TYPO_LIST",
    "UNITS",
]
''',
        encoding="utf-8",
    )
    print(f"  wrote {out / '__init__.py'}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
