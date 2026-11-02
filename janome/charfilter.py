from abc import ABC, abstractmethod
import unicodedata
import re


class CharFilter(ABC):
    @abstractmethod
    def apply(self, text: str) -> str:
        pass

    def __call__(self, text: str) -> str:
        return self.apply(text)


class RegexReplaceCharFilter(CharFilter):
    def __init__(self, pat: str, repl: str):
        self.pattern = re.compile(pat)
        self.replacement = repl

    def apply(self, text: str) -> str:
        return re.sub(self.pattern, self.replacement, text)


class UnicodeNormalizeCharFilter(CharFilter):
    def __init__(self, form: str = "NFKC"):
        self.form = form

    def apply(self, text: str) -> str:
        return unicodedata.normalize(self.form, text)
