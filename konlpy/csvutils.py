# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
import csv
import codecs

if sys.version_info[0] >= 3:
    from io import StringIO as cStringIO
else:
    import cStringIO


def _stringify(s, encoding="utf-8"):
    if s is None:
        return ""
    elif isinstance(s, unicode):
        return s.encode(encoding)
    elif isinstance(s, (int, float)):
        return s
    elif not isinstance(s, str):
        return str(s)
    return s


def _stringify_list(l, encoding="utf-8"):
    try:
        return [_stringify(s, encoding) for s in iter(l)]
    except TypeError as e:
        raise csv.Error(str(e))


class UTF8Recoder:
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")


class UnicodeReader:
    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return [unicode(s, "utf-8") for s in row]

    def __iter__(self):
        return self


class UnicodeWriter:
    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row, encoding="utf-8"):
        self.writer.writerow(_stringify_list(row, encoding))
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        data = self.encoder.encode(data)
        self.stream.write(data)
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)
