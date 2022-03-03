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
                 label=None):
        self.set_type=set_type,
        self.text1=text1,
        self.text2=text2,
        self.label=label

class MCFeature:
    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

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
            label = _ex['label']
            examples.append(MCExample(set_type=set_type,
                                       text1=text1,
                                       text2 = text2,
                                       label=label))
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

def convert_example(examples, max_seq_len, tokenizer: BertTokenizer):
    features = convert_examples_to_features(examples,max_seq_len,tokenizer)
    preds = None
    for (ex_index, example) in tqdm(enumerate(examples), desc="detect misspelling"):

        raw_text1 = example.text1
        raw_text2 = example.text2

        pinyin_a = p_wrong.get_pinyin(raw_text1)
        pinyin_b = p_wrong.get_pinyin(raw_text2)

        if pinyin_a == pinyin_b:
            yin_same = True  # 判断是否是同音字
        else:
            yin_same = False

        if preds is None:
            if yin_same:
                preds = np.array([1])
            else:
                preds = np.array([0])
        else:
            if yin_same:
                preds = np.append(preds, [1])
            else:
                preds = np.append(preds, [0])
    # all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    # all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    # all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    # all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    # dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
    #
    # return dataset, preds


    if raw_label is not None:
        raw_label = np.array(raw_label, dtype="int64")
    raw_text1 = fine_grade_tokenize(raw_text1, tokenizer)
    raw_text2 = fine_grade_tokenize(raw_text2, tokenizer)
    encode_dict = tokenizer.encode_plus(text=raw_text1,
                                        text_pair = raw_text2,
                                        max_length=max_seq_len,
                                        truncation = True,
                                        padding='max_length',
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True,return_tensors='pt')

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    feature = MCFeature(token_ids=token_ids,
                             attention_masks=attention_masks,
                             token_type_ids=token_type_ids,
                             labels=raw_label)

    return feature

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




def convert_examples_to_features( examples, max_seq_len,tokenizer:PreTrainedTokenizer):
    features = []
    count_ = 0
    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples to features"):
        text1 = example.text1
        text2 = example.text2
        label = int(example.label)
        if '被' in text1:
            out = ddp.parse(text1)

            length = len(out[0]['word'])

            entity = ['n', 'f', 's', 'nz', 'nw', 'r', 'PER', 'LOC', 'ORG', 'TIME']
            subject = []
            object = []

            edges = []
            for idx in range(length):
                num_head = out[0]['head'][idx]
                if num_head != 0:
                    edges.append((idx, num_head - 1))

                if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'SBV':
                    subject.append(idx)
                if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'VOB':
                    object.append(idx)
                if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'POB':
                    object.append(idx)

            graph = nx.Graph(edges)

            for sub in subject:
                for obj in object:

                    set_sub = []
                    set_sub_ = []
                    set_obj = []
                    set_obj_ = []

                    for idx in range(len(out[0]['head'])):
                        if out[0]['head'][idx] - 1 == sub:
                            set_sub.append(idx)
                        if idx == sub:
                            set_sub.append(idx)
                    for i in range(set_sub[0], set_sub[-1] + 1):
                        set_sub_.append(i)

                    for idx in range(len(out[0]['head'])):
                        if out[0]['head'][idx] - 1 == obj:
                            set_obj.append(idx)
                        if idx == obj:
                            set_obj.append(idx)
                    for i in range(set_obj[0], set_obj[-1] + 1):
                        set_obj_.append(i)
                    sub_ = set_sub_[0]
                    obj_ = set_obj_[0]
                    number_path = nx.shortest_path(graph, source=sub, target=obj)
                    token_path = [out[0]['word'][idx] for idx in number_path]
                    token_output = out[0]['word']
                    token_output[number_path[-2]] = token_path[1]
                    token_output[number_path[1]] = ''

                    a = ''.join([out[0]['word'][idx] for idx in set_sub_])
                    b = ''.join([out[0]['word'][idx] for idx in set_obj_])

                    for i in set_sub_:
                        token_output[i] = ''
                    for i in set_obj_:
                        token_output[i] = ''

                    token_output[obj_] = a
                    token_output[sub_] = b

                    if len(number_path) == 4 and '被' in token_path:
                        print(text1)
                        text1 = ''.join(token_output)
                        print(text1 + '\n')

        if '被' in text2:
            out = ddp.parse(text2)

            length = len(out[0]['word'])

            entity = ['n', 'f', 's', 'nz', 'nw', 'r', 'PER', 'LOC', 'ORG', 'TIME']
            subject = []
            object = []

            edges = []
            for idx in range(length):
                num_head = out[0]['head'][idx]
                if num_head != 0:
                    edges.append((idx, num_head - 1))

                if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'SBV':
                    subject.append(idx)
                if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'VOB':
                    object.append(idx)
                if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'POB':
                    object.append(idx)

            graph = nx.Graph(edges)

            for sub in subject:
                for obj in object:

                    set_sub = []
                    set_sub_ = []
                    set_obj = []
                    set_obj_ = []

                    for idx in range(len(out[0]['head'])):
                        if out[0]['head'][idx] - 1 == sub:
                            set_sub.append(idx)
                        if idx == sub:
                            set_sub.append(idx)
                    for i in range(set_sub[0], set_sub[-1] + 1):
                        set_sub_.append(i)

                    for idx in range(len(out[0]['head'])):
                        if out[0]['head'][idx] - 1 == obj:
                            set_obj.append(idx)
                        if idx == obj:
                            set_obj.append(idx)
                    for i in range(set_obj[0], set_obj[-1] + 1):
                        set_obj_.append(i)
                    sub_ = set_sub_[0]
                    obj_ = set_obj_[0]
                    number_path = nx.shortest_path(graph, source=sub, target=obj)
                    token_path = [out[0]['word'][idx] for idx in number_path]
                    token_output = out[0]['word']
                    token_output[number_path[-2]] = token_path[1]
                    token_output[number_path[1]] = ''

                    a = ''.join([out[0]['word'][idx] for idx in set_sub_])
                    b = ''.join([out[0]['word'][idx] for idx in set_obj_])

                    for i in set_sub_:
                        token_output[i] = ''
                    for i in set_obj_:
                        token_output[i] = ''

                    token_output[obj_] = a
                    token_output[sub_] = b
                    if len(number_path) == 4 and '被' in token_path:
                        print(text2)
                        text2 = ''.join(token_output)
                        print(text2 + '\n')

        bpe_tokens_a = tokenizer.tokenize(text1)
        bpe_tokens_b = tokenizer.tokenize(text2)

        bpe_tokens = [tokenizer.cls_token] + bpe_tokens_a + [tokenizer.sep_token] + bpe_tokens_b + [
            tokenizer.sep_token]
        assert isinstance(bpe_tokens, list)
        input_ids = tokenizer.convert_tokens_to_ids(bpe_tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1)

        padding = [0] * (max_seq_len - len(input_ids))
        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        input_ids = input_ids[:max_seq_len]
        attention_mask = attention_mask[:max_seq_len]
        token_type_ids = token_type_ids[:max_seq_len]

        assert len(input_ids) == max_seq_len
        assert len(attention_mask) == max_seq_len
        assert len(token_type_ids) == max_seq_len

        features.append(MCFeature(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                          label=label))
        count_ += 1
    print(f'total sample is : ',count_)
    return features






