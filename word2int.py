import torch
import Model_Builder
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from torch import nn

voc_data = np.load("vocabulary.npy")
vocab = {word:int(idx) for [word, idx] in voc_data}


class WordToInteger2(nn.Module):
    def __init__(self, vocab=vocab, max_seq_length=80, PAD=0, UNK=1):
        super().__init__()
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.PAD = PAD
        self.UNK = UNK
        
    def forward(self, x):
        tokenized_sentence = word_tokenize(x)
        emb = []
        if len(tokenized_sentence) > self.max_seq_length:
            tokenized_sentence = tokenized_sentence[:self.max_seq_length]
        for i in range(len(tokenized_sentence)):
            if tokenized_sentence[i] in self.vocab:
                emb.append(vocab[tokenized_sentence[i]])
            else:
                emb.append(self.UNK)
        if len(emb)<self.max_seq_length:
                emb = emb + [self.PAD] * (self.max_seq_length-len(emb))
        return emb