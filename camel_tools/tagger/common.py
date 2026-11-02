# -*- coding: utf-8 -*-


from abc import ABC, abstractmethod


class Tagger(ABC):
    @abstractmethod
    def tag(self, sentence):

        raise NotImplementedError
