from torch.utils.data import DataLoader,RandomSampler
import torch
import numpy as np
from data import MCExample
from data import convert_examples_to_features,build_dataset

def predict(model, data_loader):
    batch_logits = []
    model.eval()
    with torch.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data

            input_ids = torch.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)

            batch_logit, _ = model(
                input_ids=input_ids, token_type_ids=token_type_ids)

            batch_logits.append(batch_logit.numpy())

        batch_logits = np.concatenate(batch_logits, axis=0)

        return batch_logits


def process_test(testdata,opt):
    examples = []
    for _ex in testdata:
        _ex = _ex.strip().split('\t')
        assert len(_ex) == 2
        text1 = _ex[0]
        text2 = _ex[1]
        examples.append(MCExample(set_type=None,
                                  text1=text1,
                                  text2=text2,
                                  label=None))
    feature = convert_examples_to_features(examples, opt.bert_dir, opt.max_seq_len)
    testdata = build_dataset(feature,"test")
    test_sampler = RandomSampler(testdata)
    testloader = DataLoader(dataset=testdata,
                              batch_size=opt.train_batch_size,
                              sampler=test_sampler,
                              num_workers=0)
    return examples

if __name__ == '__main__':
    testdata = open('./sourceData/test.tsv','r',encoding='utf-8').readlines()









