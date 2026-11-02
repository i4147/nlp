"""``python -m farsi_textutils`` entry point.

Kept intentionally tiny: the CLI lives in :mod:`farsi_textutils.cli`, which is
also exposed as the ``farsi-textutils`` console script.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":  # pragma: no cover - module execution
    main()
