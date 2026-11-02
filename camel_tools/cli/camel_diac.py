# -*- coding: utf-8 -*-


import re
import sys

from docopt import docopt

import camel_tools
from camel_tools.morphology.database import MorphologyDB
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.cli.utils import open_files


__version__ = camel_tools.__version__

_BUILTIN_DBS = MorphologyDB.list_builtin_dbs()
_DEFAULT_DB = "calima-msa-r13"
_WHITESPACE_RE = re.compile(r"\s+|\S+")


def _diac_tokens(tokens, disambig, ignore_markers, marker, strip_markers, pretokenized):
    result = []

    for token in tokens:
        if len(token.strip()) == 0:
            result.append(token)
        elif ignore_markers and token.startswith(marker):
            if strip_markers:
                result.append(token[len(marker) :])
            else:
                result.append(token)
        else:
            if pretokenized:
                subtokens = [token]
            else:
                subtokens = simple_word_tokenize(token)
            disambig_tokens = disambig.disambiguate(subtokens)
            result.extend([d.analyses[0].analysis.get("diac", d.word) for d in disambig_tokens])

    return result


def main():
    try:
        version = "CAMeL Tools v{}".format(__version__)
        arguments = docopt(__doc__, version=version)

        if arguments["--list"]:
            for db_entry in _BUILTIN_DBS:
                print(db_entry.name)
            sys.exit(0)

        db_name = None

        for db_entry in _BUILTIN_DBS:
            if arguments["--db"] == db_entry.name:
                db_name = db_entry.name

        if db_name is None:
            sys.stderr.write(
                "Error: {} is not a valid database name.\n"
                "Run `camel_diac -l` to see the list of available"
                " databases.\n".format(repr(arguments["--db"]))
            )
            sys.exit(1)

        disambig = MLEDisambiguator.pretrained(db_name)

        marker = arguments["--marker"]
        ignore_markers = arguments["--ignore-markers"]
        strip_markers = arguments["--strip-markers"]
        pretokenized = arguments["--pretokenized"]

        fin, fout = open_files(arguments["FILE"], arguments["--output"])

        try:
            for line in fin:
                toks = _WHITESPACE_RE.findall(line)
                diac_toks = _diac_tokens(toks, disambig, ignore_markers, marker, strip_markers, pretokenized)
                fout.write("".join(diac_toks))

        except Exception:
            sys.stderr.write("Error: An unkown error occured during diacritization.\n")
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
