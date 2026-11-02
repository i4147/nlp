import re


def _multiple_replace(mapping, text):
    pattern = "|".join(map(re.escape, mapping.keys()))
    return re.sub(pattern, lambda m: mapping[m.group()], str(text))


def convert_farsi_char_to_english(input_str):
    mapping = {
        "ا": "a",
        "ع": "a",
        "غ": "g",
        "ب": "b",
        "ص": "s",
        "س": "s",
        "د": "d",
        "ی": "e",
        "ئ": "e",
        "ف": "f",
        "گ": "gh",
        "ح": "h",
        "ه": "h",
        "خ": "kh",
        "ج": "j",
        "چ": "ch",
        "ژ": "j",
        "ک": "k",
        "ل": "l",
        "م": "m",
        "ن": "n",
        "و": "v",
        "پ": "p",
        "ق": "q",
        "ر": "r",
        "س": "s",
        "ص": "s",
        "ث": "s",
        "ش": "sh",
        "ت": "t",
        "ط": "t",
        "وا": "u",
        "ز": "z",
        "ظ": "z",
        "ذ": "z",
        "ض": "z",
        " ": "_",
        "آ": "a",
        '"': "_",
        "?": "_",
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
        ".": ".",
        "،": "",
    }
    return _multiple_replace(mapping, input_str)
