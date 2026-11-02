# -*- coding: utf-8 -*-


from __future__ import absolute_import

import copy
import collections

from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.errors import GeneratorError
from camel_tools.morphology.errors import InvalidGeneratorFeature
from camel_tools.morphology.errors import InvalidGeneratorFeatureValue
from camel_tools.morphology.utils import merge_features, strip_lex


class Generator(object):
    def __init__(self, db):
        if not isinstance(db, MorphologyDB):
            raise GeneratorError("DB is not an instance of MorphologyDB")
        if not db.flags.generation:
            raise GeneratorError("DB does not support generation")

        self._db = db

    def generate(self, lemma, feats):

        lemma = strip_lex(lemma)
        if lemma not in self._db.lemma_hash:
            return []

        for feat in feats:
            if feat not in self._db.defines:
                raise InvalidGeneratorFeature(feat)
            elif self._db.defines[feat] is not None and feats[feat] not in self._db.defines[feat]:
                raise InvalidGeneratorFeatureValue(feat, feats[feat])

        if "pos" not in feats or feats["pos"] not in self._db.defines["pos"]:
            raise InvalidGeneratorFeatureValue("pos", feats.get("pos", None))

        feats = copy.copy(feats)

        default = self._db.defaults[feats["pos"]]
        default_feat_set = frozenset(default.keys())
        feat_set = frozenset(feats.keys())

        if not feat_set.issubset(default_feat_set):
            return []

        for feat in ["prc0", "prc1", "prc2", "prc3", "enc0", "enc1", "enc2"]:
            if feat not in feats and feat in default:
                feats[feat] = default[feat]

        stem_feats_list = self._db.lemma_hash[lemma]
        analyses = collections.deque()

        for stem_feats in stem_feats_list:
            if "vox" in feats and stem_feats["vox"] != feats["vox"]:
                continue
            if "rat" in feats and stem_feats["rat"] != feats["rat"]:
                continue
            if "pos" in feats and stem_feats["pos"] != feats["pos"]:
                continue

            ignore_stem = False
            for feat in ["prc0", "prc1", "prc2", "prc3", "enc0", "enc1", "enc2"]:
                if feat not in feats:
                    continue
                if feat in stem_feats and stem_feats[feat] != "0" and feats[feat] != stem_feats[feat]:
                    ignore_stem = True
                    break

            if ignore_stem:
                continue

            prefix_cats = self._db.stem_prefix_compat[stem_feats["stemcat"]]
            suffix_cats = self._db.stem_suffix_compat[stem_feats["stemcat"]]

            for prefix_cat in prefix_cats:
                if prefix_cat not in self._db.prefix_cat_hash:
                    continue

                prefix_feats_list = self._db.prefix_cat_hash[prefix_cat]
                for prefix_feats in prefix_feats_list:
                    ignore_prefix = False

                    for feat in ["prc0", "prc1", "prc2", "prc3"]:
                        if feat not in feats:
                            continue
                        if (
                            feats[feat] != "0" and feat not in prefix_feats and stem_feats.get(feat, "0") != feats[feat]
                        ) or (feat in prefix_feats and feats[feat] != prefix_feats[feat]):
                            ignore_prefix = True
                            break

                    if ignore_prefix:
                        continue

                    for suffix_cat in suffix_cats:
                        if suffix_cat not in self._db.suffix_cat_hash:
                            continue
                        suffix_feats_list = self._db.suffix_cat_hash[suffix_cat]
                        for suffix_feats in suffix_feats_list:
                            if (prefix_cat not in self._db.prefix_suffix_compat) or (
                                suffix_cat not in self._db.prefix_suffix_compat[prefix_cat]
                            ):
                                continue

                            ignore_suffix = False

                            for feat in ["enc0", "enc1", "enc2"]:
                                if feat not in feats:
                                    continue
                                if (
                                    feats[feat] != "0"
                                    and feat not in suffix_feats
                                    and stem_feats.get(feat, "0") != feats[feat]
                                ) or (feat in suffix_feats and feats[feat] != suffix_feats[feat]):
                                    ignore_suffix = True
                                    break

                            if ignore_suffix:
                                continue

                            merged = merge_features(self._db, prefix_feats, stem_feats, suffix_feats)

                            ignore_analysis = False
                            for feat in feats.keys():
                                if feat in merged and merged[feat] != feats[feat]:
                                    ignore_analysis = True
                                    break

                            if not ignore_analysis:
                                analyses.append(merged)

        return list(analyses)

    def all_feats(self):

        return self._db.all_feats()

    def tok_feats(self):
        return self._db.tok_feats()
