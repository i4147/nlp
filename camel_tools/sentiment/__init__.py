# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as torch_fun
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import utils as tf_utils

from camel_tools.data import CATALOGUE


tf_utils.logging.set_verbosity_error()


_LABELS = ("positive", "negative", "neutral")


class SentimentDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_seq_length):
        self.encoded_sents = tokenizer(
            sentences,
            add_special_tokens=True,
            padding=True,
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        return {
            "input_ids": self.encoded_sents.input_ids[idx],
            "token_type_ids": self.encoded_sents.token_type_ids[idx],
            "attention_mask": self.encoded_sents.attention_mask[idx],
        }

    def __len__(self):
        return self.encoded_sents.input_ids.shape[0]


class SentimentAnalyzer:
    def __init__(self, model_path, use_gpu=True):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.labels_map = self.model.config.id2label
        self.use_gpu = use_gpu

    @staticmethod
    def pretrained(model_name=None, use_gpu=True):

        if model_name is None:
            model_name = CATALOGUE.components["SentimentAnalysis"].default

        model_info = CATALOGUE.components["SentimentAnalysis"].datasets[model_name]
        model_path = str(model_info.path)

        return SentimentAnalyzer(model_path, use_gpu)

    @staticmethod
    def labels():

        return list(_LABELS)

    def predict_sentence(self, sentence):

        return self.predict([sentence])[0]

    def predict(self, sentences, batch_size=32):

        sentiment_dataset = SentimentDataset(sentences, self.tokenizer, max_seq_length=512)

        data_loader = DataLoader(sentiment_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"

        self.model.to(device)
        self.model.eval()

        predicted_labels = []

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                inputs = {
                    "input_ids": batch["input_ids"],
                    "token_type_ids": batch["token_type_ids"],
                    "attention_mask": batch["attention_mask"],
                }
                logits = self.model(**inputs)[0]

                predictions = torch_fun.softmax(logits, dim=-1)
                max_predictions = torch.argmax(predictions, dim=-1)
                batch_preds = [self.labels_map[p.item()] for p in max_predictions]
                predicted_labels.extend(batch_preds)

        return predicted_labels
