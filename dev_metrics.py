
import torch
# import torchmetrics
from tqdm import tqdm
import numpy as np


def get_base_out(model, loader, device):
    model.eval()
    with torch.no_grad():
        for idx, _batch in enumerate(tqdm(loader)):
            for key in _batch.keys():
                _batch[key] = _batch[key].to(device)
            labels = _batch["labels"]
            loss,pred = model(**_batch)
            yield loss,pred,labels

def get_base_out_test(model, loader, device):
    model.eval()
    with torch.no_grad():
        for idx, _batch in enumerate(tqdm(loader)):
            for key in _batch.keys():
                _batch[key] = _batch[key].to(device)
            loss,pred = model(**_batch)
            yield loss,pred











