# -*- coding: utf-8 -*-


import regex as re

from camel_tools.utils.charsets import EMOJI_ALL_CHARSET, UNICODE_PUNCT_SYMBOL_CHARSET


__all__ = ["simple_word_tokenize"]


_SYMBOLS_ALL = frozenset() | UNICODE_PUNCT_SYMBOL_CHARSET | EMOJI_ALL_CHARSET

_SYMBOLS_RE = re.compile(
    r"(?:(?:\p{RI}\p{RI}|\p{Emoji}(?:\p{EMod}|\uFE0F\u20E3?|[\U000E0020-\U000E007E]+\U000E007F)?(?:\u200D(?:\p{RI}\p{RI}|\p{Emoji}(?:\p{EMod}|\uFE0F\u20E3?|[\U000E0020-\U000E007E]+\U000E007F)?))*)|(?:[0-9*#]\u20e3)|\p{S}|\p{P})"
)

_DIGITS_RE = re.compile(r"\d+")


def _simple_word_tokenize(s: str):
    tokens = []

    for word in s.split():
        word_len = len(word)
        curr_ndx = 0
        curr_start = 0

        while curr_ndx < word_len:
            symbol_match = _SYMBOLS_RE.match(word, pos=curr_ndx)
            if symbol_match is not None:
                match_str = symbol_match[0]
                if match_str in _SYMBOLS_ALL:
                    if curr_ndx != curr_start:
                        tokens.append(word[curr_start:curr_ndx])
                    tokens.append(match_str)
                    curr_ndx = curr_start = symbol_match.end()
                    continue

            curr_ndx += 1

        last = word[curr_start:curr_ndx]
        if len(last):
            tokens.append(word[curr_start:])

    return tokens


def _simple_word_tokenize_split_digit(s: str):
    tokens = []

    for word in s.split():
        word_len = len(word)
        curr_ndx = 0
        curr_start = 0

        while curr_ndx < word_len:
            symbol_match = _SYMBOLS_RE.match(word, pos=curr_ndx)
            if symbol_match is not None:
                match_str = symbol_match[0]
                if match_str in _SYMBOLS_ALL:
                    if curr_ndx != curr_start:
                        tokens.append(word[curr_start:curr_ndx])
                    tokens.append(match_str)
                    curr_ndx = curr_start = symbol_match.end()
                    continue

            digit_match = _DIGITS_RE.match(word, pos=curr_ndx)
            if digit_match is not None:
                if curr_ndx != curr_start:
                    tokens.append(word[curr_start:curr_ndx])
                tokens.append(digit_match[0])
                curr_ndx = curr_start = digit_match.end()
                continue

            curr_ndx += 1

        last = word[curr_start:curr_ndx]
        if len(last):
            tokens.append(word[curr_start:])

    return tokens


def simple_word_tokenize(s, split_digits=False):

    if split_digits:
        return _simple_word_tokenize_split_digit(s)

    return _simple_word_tokenize(s)
