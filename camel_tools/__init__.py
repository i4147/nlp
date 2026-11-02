from __future__ import print_function, absolute_import

import os


try:
    version_file = os.path.join(os.path.dirname(__file__), "VERSION")
    with open(version_file, "r", encoding="utf-8") as version_fp:
        __version__ = version_fp.read().strip()
except Exception:
    __version__ = "???"
