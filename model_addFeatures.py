
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
from transformers import BertModel
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss


class RDropLoss(nn.Module):
    """
    R-Drop Loss implementation
        reduction(str, optional):
            Indicate how to average the loss, the candicates are ``'none'``,``'batchmean'``,``'mean'``,``'sum'``.
            If `reduction` is ``'mean'``, the reduced mean loss is returned;
            If `reduction` is ``'batchmean'``, the sum loss divided by batch size is returned;
            If `reduction` is ``'sum'``, the reduced sum loss is returned;
            If `reduction` is ``'none'``, no reduction will be applied.
            Defaults to ``'none'``.
    """
    def __init__(self, reduction='none'):
        super(RDropLoss, self).__init__()
        if reduction not in ['sum', 'mean', 'none', 'batchmean']:
            raise ValueError(
                "'reduction' in 'RDropLoss' should be 'sum', 'mean' 'batchmean', or 'none', "
                "but received {}.".format(reduction))
        self.reduction = reduction

    def forward(self, p, q, pad_mask=None):
        """
        Args:
            p(Tensor): the first forward logits of training examples.
            q(Tensor): the second forward logits of training examples.
            pad_mask(Tensor, optional): The Tensor containing the binary mask to index with, it's data type is bool.
        Returns:
            Tensor: Returns tensor `loss`, the rdrop loss of p and q.
        """
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

import os
import sys
import numpy as np
import json

import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


from transformers import BertPreTrainedModel, BertModel, PretrainedConfig, PreTrainedModel



class LacLayer(nn.Module):
    def __init__(self,
                 vocab_size=49,
                 emb_dim=4,
                 padding_idx=0,
                 gru_hidden_size=198,
                 gru_layers=1,
                 dropout_rate=0.1,
                 ):
        super(LacLayer, self).__init__()
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx
        )

        self.gru_layer = nn.GRU(input_size=emb_dim,
                                hidden_size=gru_hidden_size,
                                num_layers=gru_layers,
                                bidirectional=True,
                                dropout=dropout_rate
                                )

    def forward(self, lac_ids):
        embedded_text = self.embedder(lac_ids)
        encoded_text, last_hidden = self.gru_layer(embedded_text)
        return encoded_text


class DepLayer(nn.Module):
    def __init__(self,
                 vocab_size=29,
                 emb_dim=4,
                 padding_idx=0,
                 gru_hidden_size=198,
                 gru_layers=1,
                 dropout_rate=0.1,
                 ):
        super(DepLayer, self).__init__()
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx
        )
        self.gru_layer = nn.GRU(input_size=emb_dim,
                                hidden_size=gru_hidden_size,
                                num_layers=gru_layers,
                                bidirectional=True,
                                dropout=dropout_rate
                                )

    def forward(self, lac_ids):
        embedded_text = self.embedder(lac_ids)
        encoded_text, last_hidden = self.gru_layer(embedded_text)
        return encoded_text


class QuestionMatchingOtherTeatures(nn.Module):
    def __init__(self, config, rdrop_coef = 0.1):
        super(QuestionMatchingOtherTeatures, self).__init__()

        self.rdrop_coef = rdrop_coef
        self.num_labels = 2
        self.dropout_prob = config.hidden_dropout_prob
        config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(config.bert_dir)

        self.lac_layer = LacLayer(
            vocab_size=config.lac_vocab_size,
            emb_dim=config.gru_emb_dim,
            padding_idx=0,
            gru_hidden_size=config.gru_hidden_size,
            gru_layers=config.gru_layers,
            dropout_rate=config.gru_dropout_rate)

        self.dep_layer = DepLayer(vocab_size=config.dep_vocab_size,
                                  emb_dim=config.gru_emb_dim,
                                  padding_idx=0,
                                  gru_hidden_size=config.gru_hidden_size,
                                  gru_layers=config.gru_layers,
                                  dropout_rate=config.gru_dropout_rate)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.rdrop_loss = RDropLoss()
        self.criterion = CrossEntropyLoss()
        self.activation = nn.Softmax()
        self.classifier = nn.Linear(self.bert.config.hidden_size + config.gru_hidden_size * 2 * 2, 2)

        # self.init_weights()

    def forward(
            self,
            input_ids = None,       #g:输入维度(batch_size, max_seq_length)
            attention_mask = None,
            token_type_ids = None,
            select_tokens = None,
            lac_ids= None,
            dep_ids= None,
            sequence_length = None,
            labels = None,          #g:输出维度 (batch_size, label)
            do_evaluate = False
    ):
        batch_shape = len(select_tokens)
        outputs = self.bert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        hidden_state,cls_embedding1 = outputs[0],outputs[1]
        lac_hidden = self.lac_layer(lac_ids)
        dep_hidden = self.dep_layer(dep_ids)

        select_length = select_tokens.sum(axis=1)[:, 0].detach()  # 每个样本的mask后的长度
        hidden_state = torch.cat([hidden_state, lac_hidden, dep_hidden], dim=-1)

        x = hidden_state * select_tokens.detach()  #
        tmp_sum = torch.sum(x, dim=1)
        x = tmp_sum / (select_length.reshape((batch_shape, 1)))
        logits1 = self.classifier(x)
        if self.rdrop_coef > 0 and not do_evaluate:
            outputs2 = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            hidden_state2 = outputs2[0]
            hidden_state2 = torch.cat([hidden_state2, lac_hidden, dep_hidden], dim=-1)

            x2 = hidden_state2 * select_tokens.detach()  #
            tmp_sum2 = torch.sum(x2, dim=1)
            x2 = tmp_sum2 / (select_length.reshape((batch_shape, 1)))
            logits2 = self.classifier(x2)
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.0
        celoss = self.criterion(logits1, labels.squeeze())
        loss = celoss + self.rdrop_coef * kl_loss
        logits1 = self.activation(logits1)
        logits1 = (logits1,)
        out = (loss,) + logits1

        return out




















