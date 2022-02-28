import torch
import numpy as np
from torch.utils.data import Dataset
import json
import random
from tqdm import tqdm
from transformers import BertTokenizer

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

def convert_trigger_example(example: MCExample, max_seq_len, tokenizer: BertTokenizer):
    """
    convert trigger examples to trigger features
    """
    raw_text1 = example.text1
    raw_text2 = example.text2
    raw_label = example.label
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
                                        return_attention_mask=True)

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
        self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
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






















def read_text_pair(data_path, is_test=False):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if is_test == False:
                if len(data) != 3:
                    continue
                yield {'query1': data[0], 'query2': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    continue
                yield {'query1': data[0], 'query2': data[1]}



def convert_example(example, tokenizer, max_seq_length=512, is_test=False):

    query, title = example["query1"], example["query2"]

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids