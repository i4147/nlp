# -*- coding: utf-8 -*-


from functools import reduce
import sys


__all__ = ["FEATS_10", "FEATS_14", "FEATS_16", "FEATURE_SET_MAP", "score_analysis_uniform"]

_BACKOFF_PENALTY = float(f"1e-{sys.float_info.dig}")

FEATS_10 = ["pos", "per", "form_gen", "form_num", "asp", "prc0", "prc1", "prc2", "prc3", "enc0"]

FEATS_14 = [
    "pos",
    "per",
    "form_gen",
    "form_num",
    "asp",
    "mod",
    "vox",
    "stt",
    "cas",
    "prc0",
    "prc1",
    "prc2",
    "prc3",
    "enc0",
]

FEATS_16 = [
    "pos",
    "per",
    "form_gen",
    "form_num",
    "asp",
    "mod",
    "vox",
    "stt",
    "cas",
    "prc0",
    "prc1",
    "prc2",
    "prc3",
    "enc0",
    "enc1",
    "enc2",
]

FEATURE_SET_MAP = {
    "feats_10": FEATS_10,
    "feats_14": FEATS_14,
    "feats_16": FEATS_16,
}


def score_analysis_uniform(analysis, reference, mle_model=None, tie_breaker=None, features=FEATS_16):

    if "form_num" not in analysis.keys():
        analysis["form_num"] = analysis["num"]
    if "form_gen" not in analysis.keys():
        analysis["form_gen"] = analysis["gen"]

    score = sum(analysis.get(feat, "") == reference.get(feat, "") for feat in features)

    if tie_breaker == "tag":
        score += _tie_breaker_tag(analysis, reference, mle_model)

    if analysis["source"] == "backoff":
        score -= _BACKOFF_PENALTY

    return score


def _tie_breaker_tag(analysis, reference, mle_model):

    ordered_feats = mle_model["info"]["features"]
    score_factored_tag = reduce(lambda x, y: x * y, [mle_model[x].get(analysis.get(x, ""), 0) for x in ordered_feats])
    unfactored_tag = " ".join(f"{x}:{analysis.get(x, '')}" for x in ordered_feats)

    score_unfactored_tag = mle_model["unfactored"].get(unfactored_tag, 0)

    return (score_factored_tag + score_unfactored_tag) / 2
