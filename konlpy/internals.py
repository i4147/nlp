# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import stat

from konlpy import data


def get_datadir():
    for konlpydir in data.path:
        if os.path.exists(konlpydir) and is_writable(konlpydir):
            return konlpydir


def is_writable(path):
    if not os.path.exists(path):
        return False

    if hasattr(os, "getuid"):
        statdata = os.stat(path)
        perm = stat.S_IMODE(statdata.st_mode)
        if perm & 0o002:
            return True
        elif statdata.st_uid == os.getuid() and (perm & 0o200):
            return True
        elif (statdata.st_gid in [os.getgid()] + os.getgroups()) and (perm & 0o020):
            return True
        else:
            return False
    return True


def chmod(path):
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
