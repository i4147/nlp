# -*- coding: utf-8 -*-


import json
from pathlib import Path
import pickle
import warnings

from cachetools import LFUCache
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, BertTokenizer
from transformers import utils as tf_utils

from camel_tools.data import CATALOGUE
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.disambig.common import Disambiguator, DisambiguatedWord
from camel_tools.disambig.common import ScoredAnalysis
from camel_tools.disambig.bert._bert_morph_dataset import MorphDataset
from camel_tools.disambig.score_function import score_analysis_uniform
from camel_tools.disambig.score_function import FEATURE_SET_MAP
from camel_tools.utils.dediac import dediac_ar


tf_utils.logging.set_verbosity_error()


_SCORING_FUNCTION_MAP = {"uniform": score_analysis_uniform}


def _read_json(f_path):
    with open(f_path) as f:
        return json.load(f)


def _dediac_sentence(sentence):
    dediaced_sentence = []
    for word in sentence:
        dediaced = dediac_ar(word)
        if len(dediaced) > 0:
            dediaced_sentence.append(dediaced)
        else:
            dediaced_sentence.append(word)

    return dediaced_sentence


class _BERTFeatureTagger:
    def __init__(self, model_path, use_gpu=True):
        self._model = BertForTokenClassification.from_pretrained(model_path)
        self._tokenizer = BertTokenizer.from_pretrained(model_path)
        self._labels_map = self._model.config.id2label
        self._use_gpu = use_gpu

    def labels(self):
        num_labels = len(self._labels_map)
        labels = [self._labels_map[i] for i in range(num_labels)]

        return labels

    def _align_predictions(self, predictions, label_ids, sent_ids):

        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    preds_list[i].append(self._labels_map[preds[i][j]])

        final_preds_list = [[] for _ in range(len(set(sent_ids)))]
        for i, id in enumerate(sent_ids):
            id = id - sent_ids[0]
            final_preds_list[id].extend(preds_list[i])

        return final_preds_list

    def predict(self, sentences, batch_size=32, max_seq_length=512):

        if len(sentences) == 0:
            return []

        sorted_sentences = list(enumerate(sentences))
        sorted_sentences = sorted(sorted_sentences, key=lambda x: len(x[1]))
        sorted_sentences_idx = [i[0] for i in sorted_sentences]
        sorted_sentences_text = [i[1] for i in sorted_sentences]

        test_dataset = MorphDataset(
            sentences=sorted_sentences_text,
            tokenizer=self._tokenizer,
            labels=self.labels(),
            max_seq_length=max_seq_length,
        )

        data_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=self._collate_fn
        )

        label_ids = None
        preds = None
        sent_ids = None

        device = "cuda" if self._use_gpu and torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._model.eval()

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                inputs = {
                    "input_ids": batch["input_ids"],
                    "token_type_ids": batch["token_type_ids"],
                    "attention_mask": batch["attention_mask"],
                }

                label_ids = batch["label_ids"] if label_ids is None else torch.cat((label_ids, batch["label_ids"]))
                sent_ids = batch["sent_id"] if sent_ids is None else torch.cat((sent_ids, batch["sent_id"]))

                logits = self._model(**inputs)[0]

                preds = logits if preds is None else torch.cat((preds, logits), dim=0)
        predictions = self._align_predictions(preds.cpu().numpy(), label_ids.cpu().numpy(), sent_ids.cpu().numpy())

        sorted_predictions_pair = zip(sorted_sentences_idx, predictions)
        sorted_predictions = sorted(sorted_predictions_pair, key=lambda x: x[0])
        predictions = [i[1] for i in sorted_predictions]

        return predictions

    def _collate_fn(self, batch):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        label_ids = []
        sent_id = []

        max_seq_length = 0
        for sent in batch:
            l = len(sent["input_ids"][sent["input_ids"].nonzero()].squeeze())
            max_seq_length = max(max_seq_length, l)

        for sent in batch:
            for _, t in sent.items():
                if _ != "sent_id":
                    sent[_] = t[:max_seq_length]

        for sent in batch:
            input_ids.append(sent["input_ids"])
            token_type_ids.append(sent["token_type_ids"])
            attention_mask.append(sent["attention_mask"])
            label_ids.append(sent["label_ids"])
            sent_id.append(sent["sent_id"])

        return {
            "input_ids": torch.stack(input_ids),
            "token_type_ids": torch.stack(token_type_ids),
            "attention_mask": torch.stack(attention_mask),
            "label_ids": torch.stack(label_ids),
            "sent_id": torch.tensor(sent_id, dtype=torch.int32),
        }


class BERTUnfactoredDisambiguator(Disambiguator):
    def __init__(
        self,
        model_path,
        analyzer,
        features=FEATURE_SET_MAP["feats_14"],
        top=1,
        scorer="uniform",
        tie_breaker="tag",
        use_gpu=True,
        batch_size=32,
        ranking_cache=None,
        ranking_cache_size=100000,
    ):
        self._model = {"unfactored": _BERTFeatureTagger(model_path, use_gpu=use_gpu)}
        self._analyzer = analyzer
        self._features = features
        self._top = max(top, 1)
        self._scorer = _SCORING_FUNCTION_MAP.get(scorer, None)
        self._tie_breaker = tie_breaker
        self._use_gpu = use_gpu
        self._batch_size = batch_size
        self._mle = _read_json(f"{model_path}/mle_model.json")

        if ranking_cache is None:
            if ranking_cache_size <= 0:
                self._ranking_cache = None
                self._disambiguate_word_fn = self._disambiguate_word
            else:
                self._ranking_cache = LFUCache(ranking_cache_size)
                self._disambiguate_word_fn = self._disambiguate_word_cached
        else:
            self._ranking_cache = ranking_cache
            self._disambiguate_word_fn = self._disambiguate_word_cached

    @staticmethod
    def pretrained(
        model_name="msa",
        top=1,
        use_gpu=True,
        batch_size=32,
        cache_size=10000,
        pretrained_cache=False,
        ranking_cache_size=100000,
    ):

        model_info = CATALOGUE.get_dataset("DisambigBertUnfactored", model_name)
        model_config = _read_json(Path(model_info.path, "default_config.json"))
        model_path = str(model_info.path)
        features = FEATURE_SET_MAP[model_config["feature"]]
        db = MorphologyDB.builtin_db(model_config["db_name"], "a")
        analyzer = Analyzer(db, backoff=model_config["backoff"], cache_size=cache_size)
        scorer = model_config["scorer"]
        tie_breaker = model_config["tie_breaker"]
        ranking_cache = None
        if pretrained_cache:
            warnings.warn(
                "The `pretrained_cache` argument is deprecated and will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )

        return BERTUnfactoredDisambiguator(
            model_path,
            analyzer,
            top=top,
            features=features,
            scorer=scorer,
            tie_breaker=tie_breaker,
            use_gpu=use_gpu,
            batch_size=batch_size,
            ranking_cache=ranking_cache,
            ranking_cache_size=ranking_cache_size,
        )

    @staticmethod
    def _pretrained_from_config(
        config, top=1, use_gpu=True, batch_size=32, cache_size=10000, pretrained_cache=False, ranking_cache_size=100000
    ):

        model_config = _read_json(config)
        model_path = model_config["model_path"]
        features = FEATURE_SET_MAP[model_config["feature"]]
        db = MorphologyDB(model_config["db_path"], "a")
        analyzer = Analyzer(db, backoff=model_config["backoff"], cache_size=cache_size)
        scorer = model_config["scorer"]
        tie_breaker = model_config["tie_breaker"]
        ranking_cache = None
        if pretrained_cache:
            warnings.warn(
                "The `pretrained_cache` argument is deprecated and will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )

        return BERTUnfactoredDisambiguator(
            model_path,
            analyzer,
            top=top,
            features=features,
            scorer=scorer,
            tie_breaker=tie_breaker,
            use_gpu=use_gpu,
            batch_size=batch_size,
            ranking_cache=ranking_cache,
            ranking_cache_size=ranking_cache_size,
        )

    def _predict_sentences(self, sentences):

        preds = self._model["unfactored"].predict(sentences, batch_size=self._batch_size)
        parsed_predictions = []

        for sent, pred in zip(sentences, preds):
            parsed_prediction = []

            for word, pred in zip(sent, pred):
                d = {}
                for feat in pred.split("__"):
                    f, v = feat.split(":")
                    d[f] = v

                d["lex"] = word
                d["diac"] = word

                parsed_prediction.append(d)

            parsed_predictions.append(parsed_prediction)

        return parsed_predictions

    def _predict_sentence(self, sentence):

        parsed_predictions = []
        model = self._model["unfactored"]
        preds = model.predict([sentence], batch_size=self._batch_size)[0]

        for word, pred in zip(sentence, preds):
            d = {}
            for feat in pred.split("__"):
                f, v = feat.split(":")
                d[f] = v

            d["lex"] = word
            d["diac"] = word

            parsed_predictions.append(d)

        return parsed_predictions

    def _scored_analyses(self, word_dd, prediction):
        bert_analysis = prediction
        analyses = self._analyzer.analyze(word_dd)

        if len(analyses) == 0:
            return [
                ScoredAnalysis(
                    0,
                    bert_analysis,
                    word_dd,
                    -99,
                    -99,
                )
            ]

        scored = [
            (self._scorer(a, bert_analysis, self._mle, tie_breaker=self._tie_breaker, features=self._features), a)
            for a in analyses
        ]

        max_score = max(s[0] for s in scored)

        if max_score == 0:
            max_score = 1

        scored_analyses = [
            ScoredAnalysis(
                s / max_score,
                a,
                a["diac"],
                a.get("pos_lex_logprob", -99),
                a.get("lex_logprob", -99),
            )
            for s, a in scored
        ]

        scored_analyses.sort()

        return scored_analyses

    def _disambiguate_word(self, word, pred):
        scored_analyses = self._scored_analyses(word, pred)

        return DisambiguatedWord(word, scored_analyses[: self._top])

    def _disambiguate_word_cached(self, word, pred):
        key = (word, tuple(pred[feat] for feat in self._features))

        if key in self._ranking_cache:
            scored_analyses = self._ranking_cache[key]
        else:
            scored_analyses = self._scored_analyses(word, pred)
            self._ranking_cache[key] = scored_analyses

        return DisambiguatedWord(word, scored_analyses[: self._top])

    def disambiguate_word(self, sentence, word_ndx):

        return self.disambiguate(sentence)[word_ndx]

    def disambiguate(self, sentence):

        dediaced_sentence = _dediac_sentence(sentence)
        predictions = self._predict_sentence(dediaced_sentence)

        return [self._disambiguate_word_fn(w, p) for (w, p) in zip(sentence, predictions)]

    def disambiguate_sentences(self, sentences):

        dediaced_sentences = [_dediac_sentence(s) for s in sentences]
        predictions = self._predict_sentences(dediaced_sentences)
        disambiguated_sentences = []

        for sentence, prediction in zip(sentences, predictions):
            disambiguated_sentence = [self._disambiguate_word_fn(w, p) for (w, p) in zip(sentence, prediction)]
            disambiguated_sentences.append(disambiguated_sentence)

        return disambiguated_sentences

    def tag_sentences(self, sentences, use_analyzer=True):

        if use_analyzer:
            tagged_sentences = []
            for prediction in self.disambiguate_sentences(sentences):
                tagged_sentence = [a.analyses[0].analysis for a in prediction]
                tagged_sentences.append(tagged_sentence)

            return tagged_sentences

        return self._predict_sentences(sentences)

    def tag_sentence(self, sentence, use_analyzer=True):

        if use_analyzer:
            return [a.analyses[0].analysis for a in self.disambiguate(sentence)]

        return self._predict_sentence(sentence)

    def all_feats(self):

        return self._analyzer.all_feats()

    def tok_feats(self):

        return self._analyzer.tok_feats()

    def set_analyzer(self, analyzer):

        if analyzer is not None and isinstance(analyzer, Analyzer):
            self._analyzer = analyzer
