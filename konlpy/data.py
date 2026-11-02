# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import io
import sys
import glob
import textwrap

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle

from konlpy import utils
from konlpy.constants import DATA_DIR, make_dir


FORMATS = {
    "pickle": "A serialized Python object, stored using the ``pickle`` module.",
    "raw": "The raw (byte string) contents of a file.",
}

path = []

path += [d for d in os.environ.get("KONLPY_DATA", "").split(os.pathsep) if d]
if os.path.expanduser("~/") != "~/":
    path += [os.path.expanduser("~/konlpy_data")]

if sys.platform.startswith("win"):
    path += [
        r"C:\konlpy_data",
        r"D:\konlpy_data",
        r"E:\konlpy_data",
        os.path.join(sys.prefix, "konlpy_data"),
        os.path.join(sys.prefix, "lib", "konlpy_data"),
        os.path.join(os.environ.get("APPDATA", "C:\\"), "konlpy_data"),
    ]

else:
    path += [
        "/usr/share/konlpy_data",
        "/usr/local/share/konlpy_data",
        "/usr/lib/konlpy_data",
        "/usr/local/lib/konlpy_data",
    ]

path += ["%s/data" % utils.installpath]


def find(resource_url):
    for p in path:
        f = os.path.join(p, resource_url)
        if os.path.exists(f):
            return FileSystemPathPointer(f)

    sep = "*" * 70
    msg = textwrap.fill(
        "Resource %s not found. Please use the KoNLPy data downloader to obtain the resource: >>> konlpy.download()"
        % resource_url,
        initial_indent="  ",
        subsequent_indent="  ",
        width=66,
    )
    msg += "\n  Searched in:" + "".join("\n  - %s" % p for p in path)
    raise LookupError("\n%s\n%s\n%s" % (sep, msg, sep))


def load(resource_url, format="auto"):

    if format == "auto":
        format = os.path.splitext(resource_url)[-1].strip(".")

    if format == "pickle":
        resource_val = pickle.load(find(resource_url))
    elif format == "raw":
        resource_val = find(resource_url).open()
    else:
        assert format not in FORMATS
        raise ValueError("Unknown format type: %s" % format)

    return resource_val


class PathPointer(object):
    def open(self, encoding="utf-8"):
        raise NotImplementedError("Abstract base class")

    def file_size(self):
        raise NotImplementedError("Abstract base class")


class FileSystemPathPointer(PathPointer, str):
    def __init__(self, path):
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise IOError("No such file or directory: %s" % path)
        self.path = path

    def open(self, encoding="utf-8"):
        return utils.load_txt(self.path)

    def file_size(self):
        return os.stat(self.path).st_size


def clear():

    items = os.listdir(DATA_DIR + "*")
    for item in items:
        os.remove(item)


def listdir():

    utils.pprint(os.listdir(DATA_DIR))


class CorpusReader(object):
    def __init__(self, extension=".txt"):

        self.items = glob.glob(DATA_DIR + "*" + extension)
        self.corpus = {}

    def read(self):

        for filename in self.items:
            reader = io.open(filename, mode="r+", encoding="utf-8")
            self.corpus[os.path.basename(filename)] = reader.read()


class StringWriter(object):
    def __init__(self, filename):
        make_dir()
        self.writer = io.open(DATA_DIR + filename, mode="a", encoding="utf-8")

    def write(self, string):
        self.writer.write(string)
        self.writer.write("\n")


__all__ = [
    "find",
    "load",
    "listdir",
    "clear",
    "path",
    "FileSystemPathPointer",
    "PathPointer",
    "CorpusReader",
    "StringWriter",
]
