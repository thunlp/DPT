import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, Parameter
from transformers import BertModel, BertPreTrainedModel, ElectraModel, ElectraForPreTraining, ElectraPreTrainedModel
from textcls.datasets.processors import txtcls_processors


model_base = {
    'bert': 'bert-large-uncased',
    'fs_bert_with_label_marker': 'bert-large-uncased',
    'fs_bert_with_label_init': 'bert-large-uncased',
    'electra': 'google/electra-large-discriminator',
    'fs_electra_with_label_marker': 'google/electra-large-discriminator',
    'fs_electra_with_label_itself': 'google/electra-large-discriminator',
    'fs_electra_add_label_itself': 'google/electra-large-discriminator',
    'fs_electra_with_label_init': 'google/electra-large-discriminator',
    'fs_electra_add_label_init': 'google/electra-large-discriminator',
    'begin_with_labels_and_downline': 'google/electra-large-discriminator',
    'begin_with_labels': 'google/electra-large-discriminator',
    'begin_with_sentiment_and_downline': 'google/electra-large-discriminator',
    'begin_with_sentiment': 'google/electra-large-discriminator',
}


def tokenize_all_data_with_labels(data, tokenizer, label_tokens, max_len, label_positions):
    # tokenize data
    dic = {"input_ids": [], "attention_mask": [], "positions": []}
    #from tqdm import tqdm
    #for sentence_pair in tqdm(data):
    for sentence_pair in data:
        sent_tokens = tokenize_sentence_pair(tokenizer, sentence_pair, max_len, len(label_tokens))
        begin_of_label_tokens = len(sent_tokens)
        sent_tokens = sent_tokens + label_tokens
        sent_token_ids = tokenizer.convert_tokens_to_ids(sent_tokens)
        len_valid_tokens = len(sent_token_ids)
        sent_token_ids = sent_token_ids + [0] * (max_len - len_valid_tokens)
        dic["input_ids"].append(sent_token_ids)

        # construct att_mask
        att_mask = [1] * len_valid_tokens + [0] * (max_len - len_valid_tokens)
        dic["attention_mask"].append(att_mask)

        # construct positions of each label
        positions = [t + begin_of_label_tokens for t in label_positions]
        dic["positions"].append(positions)
    return dic


def tokenize_sentence_pair(tokenizer, sentence_pair, max_len, occupied_len=0):
    sentence0 = sentence_pair[0]
    sentence0_tokens = tokenizer.tokenize(sentence0)
    sentence1 = sentence_pair[1]
    sentence1_tokens = [] if sentence1 is None else tokenizer.tokenize(sentence1)
    current_len = len(sentence0_tokens) + len(sentence1_tokens)
    current_len = current_len + 2 if sentence1 is None else current_len + 3

    # truncation
    overflowed_len = current_len + occupied_len - max_len
    if overflowed_len > 0:
        if sentence1 is None:
            sentence0_tokens = sentence0_tokens[: len(sentence0_tokens) - overflowed_len]
        else:
            sentence1_tokens = sentence1_tokens[: len(sentence1_tokens) - overflowed_len]

    # add special tokens
    sentence1_tokens = [] if sentence1 is None else sentence1_tokens + ["[SEP]"]
    sent_tokens = ["[CLS]"] + sentence0_tokens + ["[SEP]"] + sentence1_tokens
    return sent_tokens

