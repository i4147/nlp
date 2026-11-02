# -*- coding: utf-8 -*-


from __future__ import absolute_import

from collections import deque
from collections.abc import Mapping
import os
import json

from .stringutils import isunicode


class InvalidCharMapKeyError(ValueError):
    def __init__(self, key, message):
        super(InvalidCharMapKeyError, self).__init__(message)
        self.key = key
        self.message = message

    def __repr__(self):
        return "InvalidCharMapKeyError({}, {})".format(repr(self.key), repr(self.message))

    def __str__(self):
        return self.message


class BuiltinCharMapNotFoundError(ValueError):
    def __init__(self, map_name, message):
        super(BuiltinCharMapNotFoundError, self).__init__(message)
        self.map_name = map_name
        self.message = message

    def __repr__(self):
        return "BuiltinCharMapNotFoundError({}, {})".format(repr(self.map_name), repr(self.message))

    def __str__(self):
        return self.message


class CharMapper(object):
    BUILTIN_CHARMAPS = frozenset((
        "ar2bw",
        "ar2safebw",
        "ar2xmlbw",
        "ar2hsb",
        "bw2ar",
        "bw2safebw",
        "bw2xmlbw",
        "bw2hsb",
        "safebw2ar",
        "safebw2bw",
        "safebw2xmlbw",
        "safebw2hsb",
        "xmlbw2ar",
        "xmlbw2bw",
        "xmlbw2safebw",
        "xmlbw2hsb",
        "hsb2ar",
        "hsb2bw",
        "hsb2safebw",
        "hsb2xmlbw",
        "arclean",
    ))

    @staticmethod
    def _expand_char_map(charmap):

        new_map = {}

        for key in charmap.keys():
            if not isunicode(key):
                raise TypeError("Expected string as key. Got {} instead.".format(type(key)))

            if len(key) == 1:
                if charmap[key] is not None and not isunicode(charmap[key]):
                    raise TypeError(
                        ("Expected a Unicode string or None value for key value, got {} instead.").format(
                            type(charmap[key])
                        )
                    )
                else:
                    new_map[key] = charmap[key]

            elif len(key) == 3 and key[1] == "-":
                if ord(key[0]) >= ord(key[2]):
                    raise InvalidCharMapKeyError(key, "")
                else:
                    if charmap[key] is not None and not isunicode(charmap[key]):
                        raise TypeError(
                            ("Expected a Unicode string or None value for key value, got {} instead.").format(
                                type(charmap[key])
                            )
                        )
                    for char in range(ord(key[0]), ord(key[2]) + 1):
                        new_map[chr(char)] = charmap[key]

            else:
                raise InvalidCharMapKeyError(key, "Invalid character or character range")

        return new_map

    def __init__(self, charmap, default=None):

        if isinstance(charmap, Mapping):
            self._charmap = self._expand_char_map(charmap)
        else:
            raise TypeError(("Expected a dictionary like object for charmap, got {} instead").format(type(charmap)))

        if default is None or isunicode(default):
            self._default = default
        else:
            raise TypeError(
                ("Expected a Unicode string or None value for default, got {} instead.").format(type(default))
            )

    def __call__(self, s):
        return self.map_string(s)

    @staticmethod
    def mapper_from_json(fpath):

        with open(fpath, "r", encoding="utf-8") as charmap_fp:
            jsonstr = charmap_fp.read()
            json_dict = json.loads(jsonstr)

        return CharMapper(json_dict.get("charMap", {}), default=json_dict.get("default", None))

    @staticmethod
    def builtin_mapper(map_name):

        if map_name not in CharMapper.BUILTIN_CHARMAPS:
            raise BuiltinCharMapNotFoundError(
                map_name, "No built in mapping with name '{}' was found.".format(map_name)
            )

        try:
            charmaps_dir = os.path.join(os.path.dirname(__file__), "charmaps")

        except Exception:
            raise BuiltinCharMapNotFoundError(map_name, "Could not create mapping with name '{}'.".format(map_name))

        map_path = os.path.join(charmaps_dir, "{}_map.json".format(map_name))

        return CharMapper.mapper_from_json(map_path)

    def map_string(self, s):

        if not isunicode(s):
            raise TypeError(("Expected Unicode string as input, got {} instead.").format(type(s)))

        buff = deque()

        for char in s:
            transliteration = self._charmap.get(char, self._default)
            if transliteration is None:
                buff.append(char)
            else:
                buff.append(transliteration)

        return "".join(buff)
