# -*- coding: utf-8 -*-
import os

DATA_DIR = "data/"
ALPHABET = ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파", "하"]


def make_dir(directory=DATA_DIR):

    if not os.path.exists(directory):
        os.mkdir(directory)
