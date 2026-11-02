# -*- coding: utf-8 -*-


from abc import ABC, abstractmethod
from collections import namedtuple


class ScoredAnalysis(namedtuple("ScoredAnalysis", ["score", "analysis", "diac", "pos_lex_logprob", "lex_logprob"])):
    def __lt__(self, other):
        if self.score > other.score:
            return True
        elif self.score == other.score:
            if self.pos_lex_logprob > other.pos_lex_logprob:
                return True
            elif self.pos_lex_logprob == other.pos_lex_logprob:
                if self.lex_logprob > other.lex_logprob:
                    return True
                elif self.lex_logprob == other.lex_logprob:
                    return self.diac < other.diac

        return False


class DisambiguatedWord(namedtuple("DisambiguatedWord", ["word", "analyses"])):
    pass


class Disambiguator(ABC):
    @abstractmethod
    def disambiguate(self, sentence, top=1):

        raise NotImplementedError

    @abstractmethod
    def disambiguate_word(self, sentence, word_ndx, top=1):

        raise NotImplementedError

    @abstractmethod
    def all_feats(self):

        raise NotImplementedError

    @abstractmethod
    def tok_feats(self):
        raise NotImplementedError
