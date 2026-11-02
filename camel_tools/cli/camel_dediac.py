# -*- coding: utf-8 -*-


import re
import sys

from docopt import docopt

import camel_tools as camelt
from camel_tools.utils.dediac import dediac_ar, dediac_bw, dediac_safebw
from camel_tools.utils.dediac import dediac_xmlbw, dediac_hsb
from camel_tools.cli.utils import open_files


__version__ = camelt.__version__

_BUILTIN_SCHEMES = [
    ("ar", "Arabic script", dediac_ar),
    ("bw", "Buckwalter encoding", dediac_bw),
    ("safebw", "Safe Buckwalter encoding", dediac_safebw),
    ("xmlbw", "XML Buckwalter encoding", dediac_xmlbw),
    ("hsb", "Habash-Soudi-Buckwalter encoding", dediac_hsb),
]

_WHITESPACE_RE = re.compile(r"\s+|\S+")


def _dediac_marked_tokens(tokens, dediac_fn, marker=None, strip_markers=False):
    result = []
    n = len(marker)

    if strip_markers:
        for token in tokens:
            if token.startswith(marker):
                result.append(token[n:])
            else:
                result.append(dediac_fn(token))
    else:
        for token in tokens:
            if token.startswith(marker):
                result.append(token)
            else:
                result.append(dediac_fn(token))

    return result


def main():
    try:
        version = "CAMeL Tools v{}".format(__version__)
        arguments = docopt(__doc__, version=version)

        if arguments["--list"]:
            for scheme in _BUILTIN_SCHEMES:
                print("{}   {}".format(scheme[0].ljust(8), scheme[1]))
            sys.exit(0)

        dediac_fn = None

        for scheme in _BUILTIN_SCHEMES:
            if scheme[0] == arguments["--scheme"]:
                dediac_fn = scheme[2]

        if dediac_fn is None:
            sys.stderr.write(
                "Error: {} is not a valid scheme.\n"
                "Run `camel_dediac -l` to see the list"
                " of available schemes."
                "\n".format(repr(arguments["--scheme"]))
            )
            sys.exit(1)

        strip_markers = arguments["--strip-markers"]
        marker = arguments["--marker"]
        ignore_markers = arguments["--ignore-markers"]

        fin, fout = open_files(arguments["FILE"], arguments["--output"])

        try:
            if ignore_markers:
                for line in fin:
                    toks = _WHITESPACE_RE.findall(line)
                    dediac_toks = _dediac_marked_tokens(toks, dediac_fn, marker, strip_markers)
                    fout.write("".join(dediac_toks))
            else:
                for line in fin:
                    fout.write(dediac_fn(line))

        except Exception:
            sys.stderr.write("Error: An unkown error occured during de-diacritization.\n")
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
