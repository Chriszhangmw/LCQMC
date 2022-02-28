
import torch.nn.functional as F
import os
import torch
import torch.nn as nn

from torch import nn
from torch.nn import functional as F

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
        p_loss = self.kld(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1)).sum(-1)
        q_loss = self.kld(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1)).sum(-1)

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
    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = RDropLoss()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                do_evaluate=False):

        _, cls_embedding1 = self.ptm(input_ids, token_type_ids, position_ids,
                                     attention_mask)
        cls_embedding1 = self.dropout(cls_embedding1)
        logits1 = self.classifier(cls_embedding1)

        # For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        # Original implementation please refer to this code: https://github.com/dropreg/R-Drop
        if self.rdrop_coef > 0 and not do_evaluate:
            _, cls_embedding2 = self.ptm(input_ids, token_type_ids, position_ids,
                                         attention_mask)
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2)
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.0

        return logits1, kl_loss