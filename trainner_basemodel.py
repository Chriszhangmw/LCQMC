import numpy as np
from sourceData.data_engineering.data import RawData

from model import QuestionMatchingLast3EmbeddingCls,QuestionMatching
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset,IterableDataset
from tqdm import tqdm
from functions_utils import set_seed, get_model_path_list, load_model_and_parallel
import copy
import os
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
from functions_utils import load_model_and_parallel,swa,PGD,FGM
from config import TrainArgs
import pandas as pd
from dev_metrics import get_base_out,get_base_out_test
from pytorch_transformers import WarmupLinearSchedule


class QMFeature:
    def __init__(self,
                 token_ids,
                 token_type_ids,
                 attention_masks,
                 labels=None):
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.labels = labels
opt = TrainArgs().get_parser()

def convert_example(example, tokenizer,max_seq_length,is_test=False):
    query, title = example["text_a"], example["text_b"]
    len_query,len_title = len(query),len(title)
    if max_seq_length - 3 < len_query + len_title: #超过长度
        over_size = len_query + len_title - max_seq_length + 3 #超了多少长度
        l = (over_size + 1) // 2
        query = query[:l]
        title = title[:l]
        example['text_a'] = query
        example['text_b'] = title
        print("data was cutted!")
    input_tokens = ['[CLS]'] + [c for c in query] + ['[SEP]'] + [c for c in title] + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    token_type_ids = [0] * (len(query) + 2) + [1] * (len(title) + 1)

    attention_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    attention_mask += padding
    token_type_ids += padding

    input_ids = input_ids[:max_seq_length]
    attention_mask = attention_mask[:max_seq_length]
    token_type_ids = token_type_ids[:max_seq_length]
    label = np.array([int(float(example["label"]))], dtype="int64")
    assert len(input_ids) == max_seq_length
    assert len(token_type_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length
    qm = QMFeature(token_ids = input_ids,
                   token_type_ids = token_type_ids,
                   attention_masks = attention_mask,
                   labels = label)
    return qm


class BaseDataset(Dataset):
    def __init__(self, features, mode):
        self.nums = len(features)
        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).long() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        self.labels = None
        if mode == 'train' or mode == "dev":
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


def build_optimizer_and_scheduler(opt, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.other_lr}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(opt.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler

def package_optimizer(model,opt,num_train_optimization_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': opt.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(opt.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
    return optimizer,scheduler




def simple_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()

def mc_evaluation(model, dev_info, device):
    dev_loader = dev_info
    pred_logits = None
    target = []
    for loss,pred,labels in get_base_out(model, dev_loader, device):
        tmp_pred = pred.cpu().numpy()
        labels = [l.item() for l in labels]
        target.extend(labels)
        tmp_pred = [np.argmax(x) for x in tmp_pred]
        if pred_logits is None:
            pred_logits = tmp_pred
        else:
            pred_logits = np.append(pred_logits, tmp_pred)
    acc = simple_accuracy(pred_logits,target)
    return acc
def predict(model, dev_info, device):
    dev_loader = dev_info
    pred_logits = None
    for loss,pred in get_base_out_test(model, dev_loader, device):
        tmp_pred = pred.cpu().numpy()
        tmp_pred = [np.argmax(x) for x in tmp_pred]
        if pred_logits is None:
            pred_logits = tmp_pred
        else:
            pred_logits = np.append(pred_logits, tmp_pred)
    return pred_logits

def save_model(opt, model, global_step):
    output_dir = os.path.join(opt.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    print(f'Saving model & optimizer & scheduler checkpoint to {output_dir}')
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'model.pt'))

def build_dataset(features, mode):
    dataset = MCDataset(features, mode)
    return dataset

def train(opt,model,train_dataset):

    swa_raw_model = copy.deepcopy(model)

    train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=opt.train_batch_size,
                              sampler=train_sampler,
                              num_workers=8)
    model, device = load_model_and_parallel(model, opt.gpu_ids)

    use_n_gpus = False
    if hasattr(model, "module"):
        use_n_gpus = True

    t_total = len(train_loader) * opt.train_epochs

    # optimizer, scheduler = build_optimizer_and_scheduler(opt, model, t_total)
    optimizer,scheduler = package_optimizer(model,opt,t_total/opt.train_batch_size)
    global_step = 0
    model.zero_grad()
    fgm, pgd = None, None
    attack_train_mode = opt.attack_train.lower()
    if attack_train_mode == 'fgm':
        fgm = FGM(model=model)
    elif attack_train_mode == 'pgd':
        pgd = PGD(model=model)

    pgd_k = 3
    save_steps = t_total // opt.train_epochs
    eval_steps = save_steps
    print(f'Save model in {save_steps} steps; Eval model in {eval_steps} steps')

    log_loss_steps = 20
    avg_loss = 0.
    for epoch in range(opt.train_epochs):
        for step, batch_data in enumerate(train_loader):
            model.train()
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            loss,logits1 = model(**batch_data)
            if use_n_gpus:
                loss = loss.mean()
            loss.backward()
            if fgm is not None:
                fgm.attack()
                loss_adv = model(**batch_data)[0]
                if use_n_gpus:
                    loss_adv = loss_adv.mean()
                loss_adv.backward()
                fgm.restore()
            elif pgd is not None:
                pgd.backup_grad()
                for _t in range(pgd_k):
                    pgd.attack(is_first_attack=(_t == 0))
                    if _t != pgd_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss_adv = model(**batch_data)[0]
                    if use_n_gpus:
                        loss_adv = loss_adv.mean()
                    loss_adv.backward()
                pgd.restore()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            if global_step % log_loss_steps == 0:
                avg_loss /= log_loss_steps
                print('Step: %d / %d ----> total loss : %.5f' % (global_step, t_total, avg_loss))
                avg_loss = 0.
            else:
                avg_loss += loss.item()
            if global_step % save_steps == 0:
                save_model(opt, model, global_step)
    swa(swa_raw_model, opt.output_dir, swa_start=opt.swa_start)
    print('Train done')


def main(opt):
    print(opt.file_path)
    data = pd.read_csv(opt.file_path,sep='\t')
    dev_df = data.iloc[-28802:, :]
    train_df = data.iloc[:-28802, :]
    print(f"training samples number :",len(train_df))
    train_features = []
    tokenizer = BertTokenizer.from_pretrained(opt.bert_dir)
    max_seq_length = opt.max_seq_len
    for (ex_index, example) in tqdm(enumerate(train_df.iterrows()), desc="convert train examples to features"):
        example = example[1]
        example = dict(example)
        train_data = convert_example(example, tokenizer,max_seq_length)
        train_features.append(train_data)
    train_dataset = build_dataset(train_features, 'train')
    # model = QuestionMatchingLast3EmbeddingCls(opt.bert_dir) #dev效果并不好，loss也部怎么下降
    model = QuestionMatching(opt.bert_dir)
    train(opt,model,train_dataset)


    print(f"dev samples number :", len(dev_df))
    dev_features = []
    for (ex_index, example) in tqdm(enumerate(dev_df.iterrows()), desc="convert dev_df examples to features"):
        example = example[1]
        example = dict(example)
        dev_data = convert_example(example, tokenizer,max_seq_length)
        dev_features.append(dev_data)

    dev_dataset = build_dataset(dev_features, 'dev')
    dev_loader = DataLoader(dev_dataset, batch_size=opt.eval_batch_size,shuffle=False, num_workers=8)
    model_path_list = get_model_path_list(opt.output_dir)
    max_acc = 0.
    max_acc_step = 0
    performance = {}
    for idx, model_path in enumerate(model_path_list):
        tmp_step = model_path.split('/')[-2].split('-')[-1]
        model, device = load_model_and_parallel(model, opt.gpu_ids[0],
                                                ckpt_path=model_path)
        acc = mc_evaluation(model, dev_loader, device)
        performance[tmp_step] = acc
        if acc > max_acc:
            max_acc = acc
            max_acc_step = tmp_step
    print(f"max_acc_step is :", max_acc_step)
    print(performance)




def test(opt):
    test_dataset = MQDatasetIter_test(opt)
    model = QuestionMatchingLast3EmbeddingCls(opt)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=opt.train_batch_size,
                            shuffle=False,
                            num_workers=0, collate_fn=_collate_fn_test)

    best_model_path = "/home/zmw/big_space/zhangmeiwei_space/nlp_out/question_matching/roberta_wwm/checkpoint-76900/model.pt"
    model, device = load_model_and_parallel(model, opt.gpu_ids[0],
                                            ckpt_path=best_model_path)
    res = predict(model, test_loader, device)
    files = open('./predict.csv','w',encoding='utf-8')
    for r in res:
        files.write(str(r) + '\n')


if __name__ == "__main__":
    args = TrainArgs().get_parser()
    #training

    args.output_dir = os.path.join(args.output_dir, args.bert_type)
    set_seed(seed=2022)
    if args.weight_decay:
        args.output_dir += '_wd'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)



    # test(args)



