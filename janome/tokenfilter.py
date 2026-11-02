from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterator, Tuple, Any, List, Dict
from janome.tokenizer import Token


class TokenFilter(ABC):
    @abstractmethod
    def apply(self, tokens: Iterator[Token]) -> Iterator[Any]:
        pass

    def __call__(self, tokens: Iterator[Token]) -> Iterator[Any]:
        return self.apply(tokens)


class LowerCaseFilter(TokenFilter):
    def apply(self, tokens: Iterator[Token]) -> Iterator[Token]:
        for token in tokens:
            token.surface = token.surface.lower()
            token.base_form = token.base_form.lower()
            yield token


class UpperCaseFilter(TokenFilter):
    def apply(self, tokens: Iterator[Token]) -> Iterator[Token]:
        for token in tokens:
            token.surface = token.surface.upper()
            token.base_form = token.base_form.upper()
            yield token


class POSStopFilter(TokenFilter):
    def __init__(self, pos_list: List[str]):
        self.pos_list = pos_list

    def apply(self, tokens: Iterator[Token]) -> Iterator[Token]:
        for token in tokens:
            if any(token.part_of_speech.startswith(pos) for pos in self.pos_list):
                continue
            yield token


class POSKeepFilter(TokenFilter):
    def __init__(self, pos_list: List[str]):
        self.pos_list = pos_list

    def apply(self, tokens: Iterator[Token]) -> Iterator[Token]:
        for token in tokens:
            if any(token.part_of_speech.startswith(pos) for pos in self.pos_list):
                yield token


class WordStopFilter(TokenFilter):
    def __init__(self, stop_words: List[str]):
        self.stop_words = stop_words

    def apply(self, tokens: Iterator[Token]) -> Iterator[Token]:
        for token in tokens:
            if token.surface in self.stop_words:
                continue
            yield token


class WordKeepFilter(TokenFilter):
    def __init__(self, keep_words: List[str]) -> None:
        self.keep_words = keep_words

    def apply(self, tokens: Iterator[Token]) -> Iterator[Token]:
        for token in tokens:
            if token.surface in self.keep_words:
                yield token


class CompoundNounFilter(TokenFilter):
    def apply(self, tokens: Iterator[Token]) -> Iterator[Token]:
        _ret = None
        for token in tokens:
            if _ret:
                if token.part_of_speech.startswith("名詞") and _ret.part_of_speech.startswith("名詞"):
                    _ret.surface += token.surface
                    _ret.part_of_speech = "名詞,複合,*,*"
                    _ret.base_form += token.base_form
                    _ret.reading += token.reading
                    _ret.phonetic += token.phonetic
                else:
                    ret = _ret
                    _ret = token
                    yield ret
            else:
                _ret = token
        if _ret:
            yield _ret


class ExtractAttributeFilter(TokenFilter):
    def __init__(self, att: str):
        if att not in ["surface", "part_of_speech", "infl_type", "infl_form", "base_form", "reading", "phonetic"]:
            raise Exception(f"Unknown attribute name: {att}")
        self.att = att

    def apply(self, tokens: Iterator[Token]) -> Iterator[str]:
        for token in tokens:
            yield getattr(token, self.att)


class TokenCountFilter(TokenFilter):
    def __init__(self, att: str = "surface", sorted: bool = False):
        if att not in ["surface", "part_of_speech", "infl_type", "infl_form", "base_form", "reading", "phonetic"]:
            raise Exception(f"Unknown attribute name: {att}")
        self.att = att
        self.sorted = sorted

    def apply(self, tokens: Iterator[Token]) -> Iterator[Tuple[str, int]]:
        token_counts: Dict[str, int] = defaultdict(int)
        for token in tokens:
            token_counts[getattr(token, self.att)] += 1
        if self.sorted:
            return ((k, v) for k, v in sorted(token_counts.items(), key=lambda x: x[1], reverse=True))
        else:
            return ((k, v) for k, v in token_counts.items())
