import torch
import numpy as np
from torch.utils.data import Dataset
import json
import random
from tqdm import tqdm
from transformers import BertTokenizer
import json
import collections

import time
import random
import os
import numpy as np
import math

import argparse
import logging

import torch
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    PreTrainedTokenizer,
    BertConfig,
    BertPreTrainedModel,
    BertTokenizer,
    BertModel,
    RobertaConfig,
    RobertaTokenizer,
    RobertaModel,
    get_linear_schedule_with_warmup,
)
from xpinyin import Pinyin
import networkx as nx
from ddparser import DDParser


ddp = DDParser(prob = True, use_pos = True)
p_wrong = Pinyin()


class MCExample():
    def __init__(self,
                 set_type,
                 text1,
                 text2,
                 labels=None):
        self.set_type=set_type,
        self.text1=text1,
        self.text2=text2,
        self.labels=labels

class MCFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 labels=None):
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.labels = labels

class DataProcessor:
    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            examples = json.load(f)
        return examples

    @staticmethod
    def _example_generator(raw_examples, set_type):
        examples = []
        for _ex in raw_examples:
            text1 = _ex['text1']
            text2 = _ex['text2']
            labels = _ex['label']
            examples.append(MCExample(set_type=set_type,
                                       text1=text1,
                                       text2 = text2,
                                       labels=labels))
        return examples
    def get_train_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'train')
    def get_dev_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'dev')

def fine_grade_tokenize(raw_text, tokenizer):
    tokens = []
    for _ch in raw_text:
        if not len(tokenizer.tokenize(_ch)):
            tokens.append('[UNK]')
        else:
            tokens.append(_ch)
    return tokens


class BaseDataset(Dataset):
    def __init__(self, features, mode):
        self.nums = len(features)

        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).long() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]

        self.labels = None
        if mode == 'train':
            self.labels = [torch.tensor(example.labels) for example in features]
    def __len__(self):
        return self.nums

class MCDataset(BaseDataset):
    def __init__(self,
                 features,
                 mode):
        super(MCDataset, self).__init__(features, mode)

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index]}
        if self.labels is not None:
            data['labels'] = self.labels[index]
        return data


def build_dataset(features, mode):
    dataset = MCDataset(features, mode)
    return dataset




def convert_examples_to_features( examples,tokenizer ,max_seq_len):

    features = []
    count_ = 0
    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features"):
        text1 = example.text1
        text2 = example.text2
        labels = int(example.labels)
        text1 = fine_grade_tokenize(text1, tokenizer)
        text2 = fine_grade_tokenize(text2, tokenizer)
        encode_dict = tokenizer.encode_plus(text=text1,
                                            text_pair=text2,
                                            max_length=max_seq_len,
                                            truncation=True,
                                            padding='max_length',
                                            is_pretokenized=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True)

        token_ids = encode_dict['input_ids']
        attention_masks = encode_dict['attention_mask']
        token_type_ids = encode_dict['token_type_ids']

        assert len(token_ids) == max_seq_len
        assert len(attention_masks) == max_seq_len
        assert len(token_type_ids) == max_seq_len

        features.append(MCFeature(token_ids=token_ids, attention_masks=attention_masks, token_type_ids=token_type_ids,
                          labels=labels))
        count_ += 1
    print(f'total sample is : ',count_)
    return features






