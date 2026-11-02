"""Validators for Iranian identifiers.

Each sub-module follows the same conventions:

* ``validate(value) -> bool`` -- never raises for a malformed value;
* ``normalize(value) -> str`` -- canonical form (ASCII digits, no separators);
* ``generate(...)`` / ``extract(text)`` where it makes sense;
* a ``*_data(value)`` lookup that raises :class:`~farsi_textutils.exceptions.InvalidValueError`
  for invalid input and returns ``None`` when the registry has no entry.

Example:
    >>> from farsi_textutils.validators import national_id, phone
    >>> national_id.validate("0499370899")
    True
    >>> phone.normalize("+989123456789")
    '09123456789'
"""

from __future__ import annotations

from . import bank_card, bill, legal_id, national_id, phone, plate, sheba

__all__ = ["bank_card", "bill", "legal_id", "national_id", "phone", "plate", "sheba"]
