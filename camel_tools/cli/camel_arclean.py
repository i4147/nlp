# -*- coding: utf-8 -*-


from __future__ import print_function, absolute_import

import sys

from docopt import docopt
import six

import camel_tools as camelt
from camel_tools.utils.stringutils import force_encoding, force_unicode
from camel_tools.utils.charmap import CharMapper


__version__ = camelt.__version__


def _open_files(finpath, foutpath):
    if finpath is None:
        fin = sys.stdin
    else:
        try:
            fin = open(finpath, "r", encoding="utf-8")
        except Exception:
            sys.stderr.write("Error: Couldn't open input file {}.\n".format(repr(finpath)))
            sys.exit(1)

    if foutpath is None:
        fout = sys.stdout
    else:
        try:
            fout = open(foutpath, "w", encoding="utf-8")
        except Exception:
            sys.stderr.write("Error: Couldn't open output file {}.\n".format(repr(foutpath)))
            if finpath is not None:
                fin.close()
            sys.exit(1)

    return fin, fout


def _arclean(mapper, fin, fout):
    for line in fin:
        line = force_unicode(line)

        if six.PY3:
            fout.write(mapper.map_string(line))
        else:
            fout.write(force_encoding(mapper.map_string(line)))
    fout.flush()


def main():
    try:
        version = "CAMeL Tools v{}".format(__version__)
        arguments = docopt(__doc__, version=version)

        fin, fout = _open_files(arguments["FILE"], arguments["--output"])

        try:
            mapper = CharMapper.builtin_mapper("arclean")
            _arclean(mapper, fin, fout)

        except Exception:
            sys.stderr.write("Error: An error occured during cleaning.\n")
            fin.close()
            fout.close()
            sys.exit(1)

            if arguments["FILE"] is not None:
                fin.close()
            if arguments["--output"] is not None:
                fout.close()

        sys.exit(0)
    except KeyboardInterrupt:
        sys.stderr.write("Exiting...\n")
        sys.exit(1)
    except Exception:
        sys.stderr.write("Error: An unknown error occurred.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
