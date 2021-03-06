#/usr/bin/python2
# -*- coding: utf-8 -*-

"""
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/tacotron
"""

import codecs
import csv
import os
import re

from hyperparams import Hyperparams as hp
import numpy as np

def store_vocab(idx2char):
    writer = csv.writer(codecs.open(hp.vocab_file, "w", "utf-8"))
    for idx in idx2char:
        writer.writerow([idx, idx2char[idx].encode("unicode-escape")])

def learn_vocab():
    reader = csv.reader(codecs.open(hp.text_file, "r", "utf-8"))
    vocab = set()
    for sound_fname, text in reader:
        vocab = vocab.union(set(list(text.decode("unicode-escape"))))
    char2idx = {char:idx for idx, char in enumerate(vocab)}
    idx2char = {idx:char for idx, char in enumerate(vocab)}
    return char2idx, idx2char

def load_vocab():
    reader = csv.reader(codecs.open(hp.vocab_file, "r", "utf-8"))
    char2idx = {}
    idx2char = {}
    for idx, char in reader:
        char2idx[char.decode("unicode-escape")] = int(idx)
        idx2char[int(idx)] = char.decode("unicode-escape")
    return char2idx, idx2char

def create_train_data():
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    texts, sound_files = [], []
    reader = csv.reader(codecs.open(hp.text_file, "rb", "utf-8"))
    for sound_fname, text in reader:
        sound_file = hp.sound_fpath + "/" + sound_fname + ".wav"
        text = text.decode("unicode-escape")

        if len(text) <= hp.max_len:
            texts.append(np.array([char2idx[char] for char in text if char in char2idx], np.int32).tostring())
            sound_files.append(sound_file)

    return texts, sound_files

def load_train_data():
    """We train on the whole data but the last num_samples."""
    return create_train_data()

def load_eval_data():
    """We evaluate on the last num_samples."""
    texts, _ = create_train_data()

    X = np.zeros(shape=[len(texts), hp.max_len], dtype=np.int32)
    for i, text in enumerate(texts):
        _text = np.fromstring(text, np.int32) # byte to int
        X[i, :len(_text)] = _text

    return X
