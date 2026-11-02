# -*- coding: utf-8 -*-


import sys


def open_files(finpath, foutpath):
    if finpath is None:
        fin = sys.stdin
    else:
        try:
            fin = open(finpath, "r", encoding="utf-8")
        except OSError:
            sys.stderr.write("Error: Couldn't open input file {}.\n".format(repr(finpath)))
            sys.exit(1)

    if foutpath is None:
        fout = sys.stdout
    else:
        try:
            fout = open(foutpath, "w", encoding="utf-8")
        except OSError:
            sys.stderr.write("Error: Couldn't open output file {}.\n".format(repr(foutpath)))
            if finpath is not None:
                fin.close()
            sys.exit(1)

    return fin, fout
