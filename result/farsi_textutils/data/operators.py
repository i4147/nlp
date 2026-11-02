"""Iranian mobile numbering plan: prefixes, operator, province and SIM type.

Ported verbatim (data only) from the reference implementation; the surrounding
logic was rewritten for :mod:`farsi_textutils` -- see ``result/README.md`` for the
provenance table.
"""

from __future__ import annotations

from typing import Final

OPERATOR_NAMES: Final[dict[str, str]] = {
    "ShatelMobile": "شاتل موبایل",
    "MCI": "همراه اول",
    "Irancell": "ایرانسل",
    "Taliya": "تالیا",
    "RightTel": "رایتل",
    "TeleKish": "تله کیش",
    "ApTel": "آپتل",
    "SamanTel": "سامانتل",
}

MOBILE_PREFIXES: Final[dict[str, dict[str, object]]] = {
    "910": {"base": "کشوری", "province": [], "type": ["permanent", "credit"], "operator": "همراه اول"},
    "911": {
        "province": ["گلستان", "گیلان"],
        "base": "مازندران",
        "type": ["permanent", "credit"],
        "operator": "همراه اول",
    },
    "912": {
        "province": ["البرز", "زنجان", "سمنان", "قزوین", "قم", "برخی از شهرستان های استان مرکزی"],
        "base": "تهران",
        "type": ["permanent"],
        "operator": "همراه اول",
    },
    "913": {
        "province": ["یزد", "چهارمحال و بختیاری", "کرمان"],
        "base": "اصفهان",
        "type": ["permanent", "credit"],
        "operator": "همراه اول",
    },
    "914": {
        "province": ["آذربایجان شرقی", "اردبیل", "اصفهان"],
        "base": "آذربایجان غربی",
        "type": ["permanent", "credit"],
        "operator": "همراه اول",
    },
    "915": {
        "province": ["خراسان شمالی", "خراسان جنوبی", "سیستان و بلوچستان"],
        "base": "خراسان رضوی",
        "type": ["permanent", "credit"],
        "operator": "همراه اول",
    },
    "916": {
        "province": ["لرستان", "فارس", "اصفهان"],
        "base": "خوزستان",
        "type": ["permanent", "credit"],
        "operator": "همراه اول",
    },
    "917": {
        "province": ["بوشهر", "کهگیلویه و بویر احمد", "هرمزگان"],
        "base": "فارس",
        "type": ["permanent", "credit"],
        "operator": "همراه اول",
    },
    "918": {
        "province": ["کردستان", "ایلام", "همدان"],
        "base": "کرمانشاه",
        "type": ["permanent", "credit"],
        "operator": "همراه اول",
    },
    "919": {
        "province": ["البرز", "سمنان", "قم", "قزوین", "زنجان"],
        "base": "تهران",
        "type": ["credit"],
        "operator": "همراه اول",
    },
    "990": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "همراه اول"},
    "991": {"province": [], "base": "کشوری", "type": ["permanent", "credit"], "operator": "همراه اول"},
    "992": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "همراه اول"},
    "993": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "همراه اول"},
    "994": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "همراه اول"},
    "995": {"province": [], "base": "کشوری", "type": ["credit", "permanent"], "operator": "همراه اول"},
    "996": {"province": [], "base": "کشوری", "type": ["credit", "permanent"], "operator": "همراه اول"},
    "932": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "تالیا"},
    "920": {"province": [], "base": "کشوری", "type": ["permanent"], "operator": "رایتل"},
    "921": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "رایتل"},
    "922": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "رایتل"},
    "923": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "رایتل"},
    "930": {"province": [], "base": "کشوری", "type": ["permanent", "credit"], "operator": "ایرانسل"},
    "933": {"province": [], "base": "کشوری", "type": ["permanent", "credit"], "operator": "ایرانسل"},
    "935": {"province": [], "base": "کشوری", "type": ["permanent", "credit"], "operator": "ایرانسل"},
    "936": {"province": [], "base": "کشوری", "type": ["permanent", "credit"], "operator": "ایرانسل"},
    "937": {"province": [], "base": "کشوری", "type": ["permanent", "credit"], "operator": "ایرانسل"},
    "938": {"province": [], "base": "کشوری", "type": ["permanent", "credit"], "operator": "ایرانسل"},
    "939": {"province": [], "base": "کشوری", "type": ["permanent", "credit"], "operator": "ایرانسل"},
    "900": {"province": [], "base": "کشوری", "type": ["permanent", "credit"], "operator": "ایرانسل"},
    "901": {"province": [], "base": "کشوری", "type": ["permanent", "credit"], "operator": "ایرانسل"},
    "902": {"province": [], "base": "کشوری", "type": ["permanent", "credit"], "operator": "ایرانسل"},
    "903": {"province": [], "base": "کشوری", "type": ["permanent", "credit"], "operator": "ایرانسل"},
    "905": {"province": [], "base": "کشوری", "type": ["permanent", "credit"], "operator": "ایرانسل"},
    "904": {
        "province": [],
        "base": "کشوری",
        "model": "سیم\u200cکارت کودک",
        "type": ["credit"],
        "operator": "ایرانسل",
    },
    "941": {"province": [], "base": "کشوری", "model": "TD-LTE", "type": ["credit"], "operator": "ایرانسل"},
    "99810": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "شاتل موبایل"},
    "99811": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "شاتل موبایل"},
    "99812": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "شاتل موبایل"},
    "99813": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "شاتل موبایل"},
    "99814": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "شاتل موبایل"},
    "99815": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "شاتل موبایل"},
    "99816": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "شاتل موبایل"},
    "99817": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "شاتل موبایل"},
    "934": {"province": [], "base": "کشوری", "type": ["permanent"], "operator": "تله کیش"},
    "99910": {"province": [], "base": "کشوری", "type": ["permanent"], "operator": "آپتل"},
    "99911": {"province": [], "base": "کشوری", "type": ["permanent"], "operator": "آپتل"},
    "99913": {"province": [], "base": "کشوری", "type": ["permanent"], "operator": "آپتل"},
    "99914": {"province": [], "base": "کشوری", "type": ["credit"], "operator": "آپتل"},
    "9999": {"province": [], "base": "کشوری", "type": ["credit", "permanent"], "operator": "سامانتل"},
    "99999": {"province": [], "base": "کشوری", "type": ["permanent"], "operator": "سامانتل"},
}
