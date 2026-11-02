# -*- coding: utf-8 -*-


import sys

from docopt import docopt

import camel_tools
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.cli.utils import open_files

from camel_tools.cli.utils import open_files


__version__ = camel_tools.__version__


_TOKENIZERS = [("simple_word_tokenize", simple_word_tokenize)]


def main():
    try:
        version = "CAMeL Tools v{}".format(__version__)
        arguments = docopt(__doc__, version=version)

        tokenize_fn = simple_word_tokenize

        fin, fout = open_files(arguments["FILE"], arguments["--output"])

        try:
            for line in fin:
                fout.write(" ".join(tokenize_fn(line)))
                fout.write("\n")

        except Exception:
            sys.stderr.write("Error: An unkown error occured during tokenization.\n")
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
