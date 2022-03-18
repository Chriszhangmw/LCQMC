
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
from transformers import BertModel
from torch import nn
from torch.nn import functional as F



class RDropLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(RDropLoss, self).__init__()
        if reduction not in ['sum', 'mean', 'none', 'batchmean']:
            raise ValueError(
                "'reduction' in 'RDropLoss' should be 'sum', 'mean' 'batchmean', or 'none', "
                "but received {}.".format(reduction))
        self.reduction = reduction

    def forward(self, p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1)).sum(-1)
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1)).sum(-1)

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss = torch.masked_select(p_loss, pad_mask)
            q_loss = torch.masked_select(q_loss, pad_mask)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        loss = (p_loss + q_loss) / 2
        return loss

class QuestionMatching(nn.Module):
    def __init__(self, bert_dir, dropout=None, rdrop_coef=0.1):
        super().__init__()
        config_path = os.path.join(bert_dir, 'config.json')
        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
            'pretrained bert file does not exist'
        self.ptm = BertModel.from_pretrained(bert_dir)
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.bert_config = self.ptm.config
        hidden_size = self.bert_config.hidden_size
        self.classifier = nn.Linear(hidden_size, 2)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = RDropLoss()

        self.activation = nn.Softmax()
        self.criterion = torch.nn.CrossEntropyLoss()

        # self.metric = torchmetrics.Accuracy()
        init_blocks = [self.classifier]
        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def _init_weights(self,blocks, **kwargs):
        """
                参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
                """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.zeros_(module.bias)
                    nn.init.ones_(module.weight)

    def forward(self,
                token_ids,
                token_type_ids=None,
                position_ids=None,
                attention_masks=None,
                labels=None,
                do_evaluate=False):
        token_ids = torch.squeeze(token_ids)
        token_type_ids = torch.squeeze(token_type_ids)
        attention_masks = torch.squeeze(attention_masks)
        bert_outputs = self.ptm(token_ids, token_type_ids, position_ids,
                                     attention_masks)
        cls_embedding1 = bert_outputs[1]
        cls_embedding1 = self.dropout(cls_embedding1)
        logits1 = self.classifier(cls_embedding1)
        if self.rdrop_coef > 0 and not do_evaluate:
            bert_outputs = self.ptm(token_ids, token_type_ids, position_ids,
                                         attention_masks)
            cls_embedding2 = bert_outputs[1]
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2)
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.0
        celoss  = self.criterion(logits1, labels.squeeze())
        loss = celoss + self.rdrop_coef * kl_loss

        logits1 = self.activation(logits1)
        logits1 = (logits1,)
        out = (loss,) + logits1

        return out




import os
import sys
import numpy as np
import json

import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


from transformers import BertPreTrainedModel, BertModel, PretrainedConfig, PreTrainedModel

class QuestionMatching2(BertPreTrainedModel):
    def __init__(self, config, rdrop_coef = 0.1):
        super().__init__(config)
        self.rdrop_coef = rdrop_coef
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.hidden_size_last3 = config.hidden_size * 3
        self.dropout_prob = config.hidden_dropout_prob
        config.output_hidden_states = True
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.last_classifier = nn.Linear(self.hidden_size_last3, self.num_labels)

        self.rdrop_loss = RDropLoss()

        self.init_weights()

    def forward(
            self,
            input_ids = None,       #g:输入维度 (batch_size, max_seq_length)
            attention_mask = None,
            token_type_ids = None,
            labels = None,          #g:输入维度 (batch_size, label)
    ):

        outputs = self.bert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        hidden_output = outputs[2]

        last_cat = torch.cat((pooled_output, hidden_output[-1][:, 0], hidden_output[-2][:, 0]), 1)
        last_output_linear = self.dropout(last_cat)
        logits1 = self.last_classifier(last_output_linear)
        if self.rdrop_coef > 0:
            outputs2 = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            pooled_output2 = outputs2[1]
            hidden_output2 = outputs2[2]
            last_cat2 = torch.cat((pooled_output2, hidden_output2[-1][:, 0], hidden_output2[-2][:, 0]), 1)
            last_output_linear2 = self.dropout(last_cat2)
            logits2 = self.last_classifier(last_output_linear2)
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.
        nll_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            nll_loss = loss_fct(logits1, labels) #g: logits的维度是（[batch_size, 2]）, labels的维度是([batch_size, 1])
        loss = nll_loss + self.rdrop_coef * kl_loss
        output = (logits1, ) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

class QuestionMatchingLast3EmbeddingCls(nn.Module):
    def __init__(self, bert_dir, rdrop_coef = 0):
        super().__init__()

        self.rdrop_coef = rdrop_coef
        self.num_labels = 2
        self.bert = BertModel.from_pretrained(bert_dir)
        self.bert_config = self.bert.config
        hidden_size = self.bert.config.hidden_size
        self.hidden_size = hidden_size
        self.hidden_size_last4 = self.hidden_size * 4
        self.dropout_prob =0.2
        self.bert_config.output_hidden_states = True

        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.last_classifier = nn.Linear(self.hidden_size_last4, self.num_labels)
        self.activation = nn.Softmax()
        self.rdrop_loss = RDropLoss()

    def forward(
            self,
            token_ids = None,       #g:输入维度(batch_size, max_seq_length)
            attention_masks = None,
            token_type_ids = None,
            labels = None,          #g:输出维度 (batch_size, label)
    ):
        outputs = self.bert(
            token_ids,
            attention_mask = attention_masks,
            token_type_ids = token_type_ids,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        hidden_output = outputs[2]

        last_cat = torch.cat((pooled_output, hidden_output[-1][:, 0], hidden_output[-2][:, 0], hidden_output[-3][:, 0]), 1)
        last_output_linear = self.dropout(last_cat)
        logits1 = self.last_classifier(last_output_linear)
        if self.rdrop_coef > 0:
            outputs2 = self.bert(
                token_ids,
                attention_mask=attention_masks,
                token_type_ids=token_type_ids,
            )
            pooled_output2 = outputs2[1]
            hidden_output2 = outputs2[2]
            last_cat2 = torch.cat((pooled_output2, hidden_output2[-1][:, 0], hidden_output2[-2][:, 0],hidden_output[-3][:, 0]), 1)
            last_output_linear2 = self.dropout(last_cat2)
            logits2 = self.last_classifier(last_output_linear2)
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.
        nll_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            nll_loss = loss_fct(logits1, labels.squeeze())
        loss = nll_loss + self.rdrop_coef * kl_loss

        logits1 = self.activation(logits1)
        logits1 = (logits1,)
        out = (loss,) + logits1
        return out



