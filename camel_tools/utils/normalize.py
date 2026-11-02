# -*- coding: utf-8 -*-


import re
import unicodedata

from camel_tools.utils.charmap import CharMapper


_ALEF_NORMALIZE_BW_RE = re.compile("[<>{|]")
_ALEF_NORMALIZE_SAFEBW_RE = re.compile("[IOLM]")
_ALEF_NORMALIZE_XMLBW_RE = re.compile("[IO{|]")
_ALEF_NORMALIZE_HSB_RE = re.compile("[\u0102\u00c2\u00c4\u0100]")
_ALEF_NORMALIZE_AR_RE = re.compile("[\u0625\u0623\u0671\u0622]")

_UNICODE_CHAR_FIX = CharMapper({
    "\ufdfc": "ريال",
    "\ufdfd": "بسم الله الرحمن الرحيم",
})


def normalize_unicode(s, compatibility=True):

    if compatibility:
        fixed = _UNICODE_CHAR_FIX(s)
        return unicodedata.normalize("NFKC", fixed)

    return unicodedata.normalize("NFC", s)


def normalize_alef_maksura_bw(s):

    return s.replace("Y", "y")


def normalize_alef_maksura_safebw(s):

    return s.replace("Y", "y")


def normalize_alef_maksura_xmlbw(s):

    return s.replace("Y", "y")


def normalize_alef_maksura_hsb(s):

    return s.replace("\u00fd", "y")


def normalize_alef_maksura_ar(s):

    return s.replace("\u0649", "\u064a")


def normalize_teh_marbuta_bw(s):

    return s.replace("p", "h")


def normalize_teh_marbuta_safebw(s):

    return s.replace("p", "h")


def normalize_teh_marbuta_xmlbw(s):

    return s.replace("p", "h")


def normalize_teh_marbuta_hsb(s):

    return s.replace("\u0127", "h")


def normalize_teh_marbuta_ar(s):

    return s.replace("\u0629", "\u0647")


def normalize_alef_bw(s):

    return _ALEF_NORMALIZE_BW_RE.sub("A", s)


def normalize_alef_safebw(s):

    return _ALEF_NORMALIZE_SAFEBW_RE.sub("A", s)


def normalize_alef_xmlbw(s):

    return _ALEF_NORMALIZE_XMLBW_RE.sub("A", s)


def normalize_alef_hsb(s):

    return _ALEF_NORMALIZE_HSB_RE.sub("A", s)


def normalize_alef_ar(s):

    return _ALEF_NORMALIZE_AR_RE.sub("\u0627", s)
