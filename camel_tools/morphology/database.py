# -*- coding: utf-8 -*-


from __future__ import absolute_import

from collections import namedtuple
from pathlib import Path
import re

from camel_tools.utils.stringutils import force_unicode
from camel_tools.morphology.utils import strip_lex
from camel_tools.morphology.errors import InvalidDatabaseFlagError
from camel_tools.morphology.errors import DatabaseParseError
from camel_tools.data import CATALOGUE


MorphologyDBFlags = namedtuple("MorphologyDBFlags", ["analysis", "generation", "reinflection"])


class MorphologyDB:
    @staticmethod
    def list_builtin_dbs():

        return list(CATALOGUE.get_component("MorphologyDB").datasets.values())

    @staticmethod
    def builtin_db(db_name=None, flags="a"):

        if db_name is None:
            db_name = CATALOGUE.components["MorphologyDB"].default

        db_info = CATALOGUE.components["MorphologyDB"].datasets[db_name]

        return MorphologyDB(str(Path(db_info.path, "morphology.db")), flags)

    def __init__(self, fpath, flags="a"):

        self._withAnalysis = False
        self._withReinflection = False
        self._withGeneration = False
        self._defaultKey = "pos"

        for flag in flags:
            if flag == "a":
                self._withAnalysis = True
            elif flag == "g":
                self._withGeneration = True
            elif flag == "r":
                self._withReinflection = True
                self._withAnalysis = True
                self._withGeneration = True
            else:
                raise InvalidDatabaseFlagError(flag)

        if self._withAnalysis and self._withGeneration:
            self._withReinflection = True

        self.flags = MorphologyDBFlags(self._withAnalysis, self._withGeneration, self._withGeneration)
        self.defines = {}
        self.defaults = {}
        self.order = None
        self.tokenizations = set()
        self.compute_feats = frozenset()
        self.stem_backoffs = {}

        self.prefix_hash = {}
        self.suffix_hash = {}
        self.stem_hash = {}

        self.prefix_cat_hash = {}
        self.suffix_cat_hash = {}
        self.lemma_hash = {}

        self.prefix_stem_compat = {}
        self.stem_suffix_compat = {}
        self.prefix_suffix_compat = {}
        self.stem_prefix_compat = {}
        self.max_prefix_size = 0
        self.max_suffix_size = 0

        self._parse_dbfile(fpath)

    def _parse_analysis_line_toks(self, toks):
        res = {}

        for tok in toks:
            if len(tok) == 0:
                continue

            subtoks = tok.split(":")
            if len(subtoks) < 2:
                raise DatabaseParseError("invalid key value pair {}".format(repr(tok)))

            res[subtoks[0]] = ":".join(subtoks[1:])

        return res

    def _parse_defaults_line_toks(self, toks):
        res = {}

        for tok in toks:
            subtoks = tok.split(":")
            if len(subtoks) < 2:
                raise DatabaseParseError("invalid key value pair {} in DEFAULTS".format(repr(tok)))

            feat = subtoks[0]
            val = ":".join(subtoks[1:])

            if val == "*":
                res[feat] = None
            else:
                res[feat] = val

        return res

    def _parse_dbfile(self, fpath):
        with open(fpath, "r", encoding="utf-8") as dbfile:
            for line in dbfile:
                line = line = force_unicode(line).strip()

                if line == "###DEFINES###":
                    continue

                if line == "###DEFAULTS###":
                    break

                toks = line.split(" ")

                if len(toks) < 3 or toks[0] != "DEFINE":
                    raise DatabaseParseError("invalid DEFINES line {}".format(repr(line)))

                new_define = toks[1]
                val_set = set()

                for tok in toks[2:]:
                    subtoks = tok.split(":")

                    if len(subtoks) != 2 and subtoks[0] != toks[1]:
                        raise DatabaseParseError("invalid key value pair {} in DEFINES".format(repr(tok)))

                    if len(toks) == 3 and subtoks[1] == "*open*":
                        val_set = None
                        break

                    val_set.add(subtoks[1])

                self.defines[new_define] = list(val_set) if val_set is not None else None

            for line in dbfile:
                line = force_unicode(line).strip()

                if line == "###ORDER###":
                    break

                toks = line.split(" ")

                if len(toks) < 2 or toks[0] != "DEFAULT":
                    raise DatabaseParseError("invalid DEFAULTS line {}".format(repr(line)))

                parsed_default = self._parse_defaults_line_toks(toks[1:])

                if self._defaultKey not in parsed_default:
                    raise DatabaseParseError("DEFAULTS line {} missing {} value".format(repr(line), self._defaultKey))

                dkey = parsed_default[self._defaultKey]
                self.defaults[dkey] = parsed_default

            for line in dbfile:
                line = force_unicode(line).strip()

                if line == "###TOKENIZATIONS###":
                    self.compute_feats = frozenset(self.order)
                    break

                toks = line.split(" ")

                if self.order is not None and len(toks) < 2 and toks[0] != "ORDER":
                    raise DatabaseParseError("invalid ORDER line {}".format(repr(line)))

                if toks[1] not in self.defines:
                    raise DatabaseParseError("invalid feature {} in ORDER line.".format(repr(toks[1])))

                self.order = toks[1:]

            for line in dbfile:
                line = force_unicode(line).strip()

                if line == "###STEMBACKOFF###":
                    self.tokenizations = frozenset(self.tokenizations)
                    break

                toks = line.split(" ")

                if self.order is not None and len(toks) < 2 and toks[0] != "TOKENIZATION":
                    raise DatabaseParseError("invalid TOKENIZATION line {}".format(repr(line)))

                if toks[1] not in self.defines:
                    raise DatabaseParseError("invalid feature {} in TOKENIZATION line.".format(repr(toks[1])))

                self.tokenizations.update(toks[1:])

            for line in dbfile:
                line = force_unicode(line).strip()

                if line == "###PREFIXES###":
                    break

                toks = line.split(" ")

                if len(toks) < 3 or toks[0] != "STEMBACKOFF":
                    raise DatabaseParseError("invalid STEMBACKOFFS line {}".format(repr(line)))

                self.stem_backoffs[toks[1]] = toks[2:]

            for line in dbfile:
                line = force_unicode(line)
                parts = line.split("\t")

                if len(parts) != 3:
                    if line.strip() == "###SUFFIXES###":
                        break
                    raise DatabaseParseError("invalid PREFIXES line {}".format(repr(line)))

                prefix = parts[0].strip()
                category = parts[1]
                analysis = self._parse_analysis_line_toks(parts[2].strip().split(" "))

                if self._withAnalysis:
                    if prefix not in self.prefix_hash:
                        self.prefix_hash[prefix] = []
                    self.prefix_hash[prefix].append((category, analysis))

                if self._withGeneration:
                    if category not in self.prefix_cat_hash:
                        self.prefix_cat_hash[category] = []
                    self.prefix_cat_hash[category].append(analysis)

            for line in dbfile:
                line = force_unicode(line)
                parts = line.split("\t")

                if len(parts) != 3:
                    if line.strip() == "###STEMS###":
                        break
                    raise DatabaseParseError("invalid SUFFIXES line {}".format(repr(line)))

                suffix = parts[0].strip()
                category = parts[1]
                analysis = self._parse_analysis_line_toks(parts[2].strip().split(" "))

                if self._withAnalysis:
                    if suffix not in self.suffix_hash:
                        self.suffix_hash[suffix] = []
                    self.suffix_hash[suffix].append((category, analysis))

                if self._withGeneration:
                    if category not in self.suffix_cat_hash:
                        self.suffix_cat_hash[category] = []
                    self.suffix_cat_hash[category].append(analysis)

            for line in dbfile:
                line = force_unicode(line).strip()

                if line == "###TABLE AB###":
                    break

                parts = line.split("\t")

                if len(parts) != 3:
                    raise DatabaseParseError("invalid STEMS line {}".format(repr(line)))

                stem = parts[0]
                category = parts[1]
                analysis = self._parse_analysis_line_toks(parts[2].split(" "))
                analysis["lex"] = strip_lex(analysis["lex"])

                if self._withAnalysis:
                    if stem not in self.stem_hash:
                        self.stem_hash[stem] = []
                    self.stem_hash[stem].append((category, analysis))

                if self._withGeneration:
                    lemma_key = analysis["lex"]
                    analysis["stemcat"] = category
                    if lemma_key not in self.lemma_hash:
                        self.lemma_hash[lemma_key] = []
                    self.lemma_hash[lemma_key].append(analysis)

            for line in dbfile:
                line = force_unicode(line).strip()

                if line == "###TABLE BC###":
                    break

                toks = line.split()

                if len(toks) != 2:
                    raise DatabaseParseError("invalid TABLE AB line {}".format(repr(line)))

                prefix_cat = toks[0]
                stem_cat = toks[1]

                if self._withAnalysis:
                    if prefix_cat not in self.prefix_stem_compat:
                        self.prefix_stem_compat[prefix_cat] = set()
                    self.prefix_stem_compat[prefix_cat].add(stem_cat)

                if self._withGeneration:
                    if stem_cat not in self.stem_prefix_compat:
                        self.stem_prefix_compat[stem_cat] = set()
                    self.stem_prefix_compat[stem_cat].add(prefix_cat)

            for line in dbfile:
                line = force_unicode(line).strip()

                if line == "###TABLE AC###":
                    break

                toks = line.split()

                if len(toks) != 2:
                    raise DatabaseParseError("invalid TABLE BC line {}".format(repr(line)))

                stem_cat = toks[0]
                suffix_cat = toks[1]

                if stem_cat not in self.stem_suffix_compat:
                    self.stem_suffix_compat[stem_cat] = set()
                self.stem_suffix_compat[stem_cat].add(suffix_cat)

            for line in dbfile:
                line = force_unicode(line).strip()

                toks = line.split()

                if len(toks) != 2:
                    raise DatabaseParseError("invalid TABLE AC line {}".format(repr(line)))

                prefix_cat = toks[0]
                suffix_cat = toks[1]

                if prefix_cat not in self.prefix_suffix_compat:
                    self.prefix_suffix_compat[prefix_cat] = set()
                self.prefix_suffix_compat[prefix_cat].add(suffix_cat)

            if self._withAnalysis:
                for prefix in self.prefix_hash.keys():
                    self.max_prefix_size = max(self.max_prefix_size, len(prefix))
                for suffix in self.suffix_hash.keys():
                    self.max_suffix_size = max(self.max_suffix_size, len(suffix))

    def all_feats(self):

        return frozenset(self.defines.keys())

    def tok_feats(self):

        return self.tokenizations
