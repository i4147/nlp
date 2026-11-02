# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import Dataset


def _prepare_sentences(sentences, placeholder=""):

    guid_index = 1
    prepared_sentences = []

    for words in sentences:
        labels = [placeholder] * len(words)
        prepared_sentences.append(_PrepSentence(guid=f"{guid_index}", words=words, labels=labels))
        guid_index += 1

    return prepared_sentences


class _PrepSentence:
    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels


class MorphDataset(Dataset):
    def __init__(self, sentences, tokenizer, labels, max_seq_length):
        prepared_sentences = _prepare_sentences(sentences, placeholder=labels[0])
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index
        self.features = self._featurize_input(
            prepared_sentences,
            labels,
            max_seq_length,
            tokenizer,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id,
        )

    def _featurize_input(
        self,
        prepared_sentences,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token="[CLS]",
        cls_token_segment_id=0,
        sep_token="[SEP]",
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ):

        label_map = {label: i for i, label in enumerate(label_list)}
        features = []

        for sent_id, sentence in enumerate(prepared_sentences):
            tokens = []
            label_ids = []

            for word, label in zip(sentence.words, sentence.labels):
                if word is None or word.strip() == "":
                    continue
                word_tokens = tokenizer.tokenize(word)
                if not word_tokens or all(token == tokenizer.unk_token for token in word_tokens):
                    word_tokens = [tokenizer.unk_token]

                tokens.append(word_tokens)
                label_ids.append([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

            token_segments = []
            token_segment = []
            label_ids_segments = []
            label_ids_segment = []
            num_word_pieces = 0
            seg_seq_length = max_seq_length - 2

            if len(tokens) == 0:
                data = self._add_special_tokens(
                    token_segment,
                    label_ids_segment,
                    tokenizer,
                    max_seq_length,
                    cls_token,
                    sep_token,
                    pad_token,
                    cls_token_segment_id,
                    pad_token_segment_id,
                    pad_token_label_id,
                    sequence_a_segment_id,
                    mask_padding_with_zero,
                )
                data["sent_id"] = sent_id
                features.append(data)
            else:
                for idx, word_pieces in enumerate(tokens):
                    if num_word_pieces + len(word_pieces) > seg_seq_length:
                        data = self._add_special_tokens(
                            token_segment,
                            label_ids_segment,
                            tokenizer,
                            max_seq_length,
                            cls_token,
                            sep_token,
                            pad_token,
                            cls_token_segment_id,
                            pad_token_segment_id,
                            pad_token_label_id,
                            sequence_a_segment_id,
                            mask_padding_with_zero,
                        )
                        data["sent_id"] = sent_id
                        features.append(data)

                        token_segments.append(token_segment)
                        label_ids_segments.append(label_ids_segment)
                        token_segment = list(word_pieces)
                        label_ids_segment = list(label_ids[idx])
                        num_word_pieces = len(word_pieces)
                    else:
                        token_segment.extend(word_pieces)
                        label_ids_segment.extend(label_ids[idx])
                        num_word_pieces += len(word_pieces)

                if len(token_segment) > 0:
                    data = self._add_special_tokens(
                        token_segment,
                        label_ids_segment,
                        tokenizer,
                        max_seq_length,
                        cls_token,
                        sep_token,
                        pad_token,
                        cls_token_segment_id,
                        pad_token_segment_id,
                        pad_token_label_id,
                        sequence_a_segment_id,
                        mask_padding_with_zero,
                    )
                    data["sent_id"] = sent_id
                    features.append(data)

                    token_segments.append(token_segment)
                    label_ids_segments.append(label_ids_segment)

        return features

    def _add_special_tokens(
        self,
        tokens,
        label_ids,
        tokenizer,
        max_seq_length,
        cls_token,
        sep_token,
        pad_token,
        cls_token_segment_id,
        pad_token_segment_id,
        pad_token_label_id,
        sequence_a_segment_id,
        mask_padding_with_zero,
    ):

        _tokens = list(tokens)
        _label_ids = list(label_ids)

        _tokens += [sep_token]
        _label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(_tokens)

        _tokens = [cls_token] + _tokens
        _label_ids = [pad_token_label_id] + _label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(_tokens)

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        _label_ids += [pad_token_label_id] * padding_length

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(input_mask),
            "token_type_ids": torch.tensor(segment_ids),
            "label_ids": torch.tensor(_label_ids),
        }

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]
