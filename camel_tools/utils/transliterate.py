# -*- coding: utf-8 -*-


from __future__ import absolute_import

from collections import deque
import re
import six

from camel_tools.utils.charmap import CharMapper


_WHITESPACE_RE = re.compile(r"\s")


class Transliterator(object):
    def __init__(self, mapper, marker="@@IGNORE@@"):
        self._mapper = mapper

        if not isinstance(mapper, CharMapper):
            raise TypeError("Mapper is not a CharMapper instance.")

        if not isinstance(marker, six.string_types):
            raise TypeError("Marker is not a string.")

        if not marker:
            raise ValueError("Marker is empty.")
        elif _WHITESPACE_RE.search(marker) is None:
            self._marker = marker
        else:
            raise ValueError("Marker contains whitespace.")

        self._markerre = re.compile(r"({}\S+)".format(re.escape(marker)), re.UNICODE | re.MULTILINE)

    def transliterate(self, s, strip_markers=False, ignore_markers=False):

        buff = deque()

        splits = self._markerre.split(s)
        for spl in splits:
            if spl.startswith(self._marker):
                if ignore_markers:
                    if not strip_markers:
                        buff.append(self._marker)
                    buff.append(self._mapper.map_string(spl[len(self._marker) :]))
                else:
                    if strip_markers:
                        buff.append(spl[len(self._marker) :])
                    else:
                        buff.append(spl)
            else:
                buff.append(self._mapper.map_string(spl))

        return "".join(buff)
