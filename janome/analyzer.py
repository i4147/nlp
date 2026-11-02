from typing import List, Iterator, Any, Optional
from .tokenizer import Tokenizer
from .charfilter import CharFilter
from .tokenfilter import TokenFilter


class Analyzer(object):
    def __init__(
        self,
        *,
        char_filters: List[CharFilter] = [],
        tokenizer: Optional[Tokenizer] = None,
        token_filters: List[TokenFilter] = [],
    ):
        if not tokenizer:
            self.tokenizer = Tokenizer()
        elif tokenizer.wakati:
            raise Exception("Invalid argument: A Tokenizer with wakati=True option is not accepted.")
        else:
            self.tokenizer = tokenizer
        self.char_filters = char_filters
        self.token_filters = token_filters

    def analyze(self, text: str) -> Iterator[Any]:
        for cfilter in self.char_filters:
            text = cfilter(text)
        tokens = self.tokenizer.tokenize(text, wakati=False)
        for tfilter in self.token_filters:
            tokens = tfilter(tokens)
        return tokens
