
import torch
import torchmetrics
from tqdm import tqdm
import numpy as np


def get_base_out(model, loader, device):
    model.eval()
    with torch.no_grad():
        for idx, _batch in enumerate(tqdm(loader)):
            for key in _batch.keys():
                _batch[key] = _batch[key].to(device)
            labels = _batch[-1]
            loss,pred = model(**_batch)
            yield loss,pred,labels

def mc_evaluation(model, dev_info, device):
    dev_loader = dev_info
    accuracy = torchmetrics.Accuracy()
    pred_logits = None
    target = []
    for loss,pred,labels in get_base_out(model, dev_loader, device):
        tmp_pred = pred.cpu().numpy()
        target.append(labels)
        if pred_logits is None:
            pred_logits = tmp_pred
        else:
            pred_logits = np.append(pred_logits, tmp_pred, axis=0)
    acc = accuracy(pred_logits,target).item()
    accuracy.reset()
    return acc












