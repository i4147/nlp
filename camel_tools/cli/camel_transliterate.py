# -*- coding: utf-8 -*-


from __future__ import print_function, absolute_import

import sys

from docopt import docopt
import six

import camel_tools as camelt
from camel_tools.utils.stringutils import force_encoding, force_unicode
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.transliterate import Transliterator


__version__ = camelt.__version__


_BUILTIN_SCHEMES = [
    ("ar2bw", "Arabic to Buckwalter"),
    ("ar2safebw", "Arabic to Safe Buckwalter"),
    ("ar2xmlbw", "Arabic to XML Buckwalter"),
    ("ar2hsb", "Arabic to Habash-Soudi-Buckwalter"),
    ("bw2ar", "Buckwalter to Arabic"),
    ("bw2safebw", "Buckwalter to Safe Buckwalter"),
    ("bw2xmlbw", "Buckwalter to XML Buckwalter"),
    ("bw2hsb", "Buckwalter to Habash-Soudi-Buckwalter"),
    ("safebw2ar", "Safe Buckwalter to Arabic"),
    ("safebw2bw", "Safe Buckwalter to Buckwalter"),
    ("safebw2xmlbw", "Safe Buckwalter to XML Buckwalter"),
    ("safebw2hsb", "Safe Buckwalter to Habash-Soudi-Buckwalter"),
    ("xmlbw2ar", "XML Buckwalter to Arabic"),
    ("xmlbw2bw", "XML Buckwalter to Buckwalter"),
    ("xmlbw2safebw", "XML Buckwalter to Safe Buckwalter"),
    ("xmlbw2hsb", "XML Buckwalter to Habash-Soudi-Buckwalter"),
    ("hsb2ar", "Habash-Soudi-Buckwalter to Arabic"),
    ("hsb2bw", "Habash-Soudi-Buckwalter to Buckwalter"),
    ("hsb2safebw", "Habash-Soudi-Buckwalter to Safe Buckwalter"),
    ("hsb2xmlbw", "Habash-Soudi-Buckwalter to Habash-Soudi-Buckwalter"),
]


def _open_files(finpath, foutpath):
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


def main():
    try:
        version = "CAMeL Tools v{}".format(__version__)
        arguments = docopt(__doc__, version=version)

        if arguments["--list"]:
            for scheme in _BUILTIN_SCHEMES:
                print("{}   {}".format(scheme[0].ljust(20), scheme[1]))
            sys.exit(0)

        if arguments["--scheme"] is not None:
            if arguments["--scheme"] not in [s[0] for s in _BUILTIN_SCHEMES]:
                sys.stderr.write(
                    "Error: {} is not a valid scheme.\n"
                    "Run `camel_transliterate -l` to see the list"
                    " of available schemes."
                    "\n".format(repr(arguments["--scheme"]))
                )
                sys.exit(1)

            if arguments["--marker"] is None:
                marker = "@@IGNORE@@"
            else:
                marker = arguments["--marker"]

            ignore_markers = arguments["--ignore-markers"]
            strip_markers = arguments["--strip-markers"]

            fin, fout = _open_files(arguments["FILE"], arguments["--output"])

            try:
                mapper = CharMapper.builtin_mapper(arguments["--scheme"])
                trans = Transliterator(mapper, marker)
            except Exception:
                sys.stderr.write("Error: Could not load builtin scheme {}.\n".format(repr(arguments["--scheme"])))
                sys.exit(1)

            try:
                for line in fin:
                    line = force_unicode(line)

                    if six.PY3:
                        fout.write(trans.transliterate(line, strip_markers, ignore_markers))
                    else:
                        fout.write(force_encoding(trans.transliterate(line, strip_markers, ignore_markers)))
                fout.flush()

            except Exception:
                sys.stderr.write("Error: An unkown error occured during transliteration.\n")
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
