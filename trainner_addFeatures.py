import numpy as np
from sourceData.data_engineering.data import RawData
from config import lac2id, dep2id
from model_addFeatures import QuestionMatchingOtherTeatures
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from functions_utils import set_seed, get_model_path_list, load_model_and_parallel
import copy
import os
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler
from functions_utils import load_model_and_parallel,swa,PGD,FGM
from config import TrainArgs
import pandas as pd
from dev_metrics import get_base_out


def getMaskIndexWithLac(example ,all_select):
    x1 = example['text_a']
    x2 = example['text_b']
    lac1 = example['postag_a']
    lac2 = example['postag_b']

    s1 = example['word_a']
    s2 = example['word_b']

    deprel1 = example['deprel_a']
    deprel2 = example['deprel_b']
    lac1 = eval(lac1)
    lac2 = eval(lac2)
    s1 = eval(s1)
    s2 = eval(s2)
    deprel1 = eval(deprel1)
    deprel2 = eval(deprel2)
    if all_select: #全部为1
        index1 = np.ones((len(x1),800))
        index2 = np.ones((len(x2),800))
    else: #
        index1 = np.zeros((len(x1), 800))
        index2 = np.zeros((len(x2), 800))

    lac12id = np.zeros(len(x1))
    lac22id = np.zeros(len(x2))

    dep2id1 = np.zeros(len(x1))
    dep2id2 = np.zeros(len(x2))
    if len(s1) != len(lac1) or len(s1) != len(deprel1):
        print(s1,type(s1))

    assert len(s1) == len(lac1) and len(s1) == len(deprel1),'长度不一致！%s _ %s _ %s' % (len(s1),len(lac1),len(deprel1))
    assert len(s2) == len(lac2) and len(s2) == len(deprel2), '长度不一致！！%s _ %s _ %s'% (len(s2),len(lac2),len(deprel2))
    i = 0
    for w,l,d in  zip(s1,lac1,deprel1):
        if w not in s2 and not all_select:
            for j in range(i,i+len(w)):
                if j < len(x1):
                    index1[j,:] = 1
        for j in range(i,i+len(w)):
            if j==i and j < len(x1):
                dep2id1[j] = dep2id.get('B-' + d, 0)
                lac12id[j] = lac2id.get('B-' + l, 0)
            elif j < len(x1):
                dep2id1[j] = dep2id.get('I-' + d, 0)
                lac12id[j] = lac2id.get('I-' + l, 0)
            else:
                break
        i += len(w)

    i = 0
    for w,l,d in zip(s2,lac2,deprel2):
        if w not in s1 and not  all_select:
            for j in range(i,i+len(w)):
                if j<len(x2):
                    index2[j,:] = 1
        for j in range(i,i+len(w)):
            if j==i and j < len(x2):
                dep2id2[j] = dep2id.get('B-' + d, 0)
                lac22id[j] = lac2id.get('B-' + l, 0)
            elif j < len(x2):
                dep2id2[j] = dep2id.get('I-' + d, 0)
                lac22id[j] = lac2id.get('I-' + l, 0)
            else:
                break
        i += len(w)

    return index1,index2,lac12id,lac22id,dep2id1,dep2id2

class QMFeature:
    def __init__(self,
                 input_ids,
                 token_type_ids,
                 select_tokens,
                 lac_ids,
                 dep_ids,
                 sequence_length,
                 attention_mask,
                 labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.select_tokens = select_tokens
        self.lac_ids = lac_ids
        self.dep_ids = dep_ids
        self.sequence_length = sequence_length
        self.labels = labels

def convert_example_with_lac(example,opt, is_test=False):
    ratio = opt.ratio
    tokenizer = BertTokenizer.from_pretrained(opt.bert_dir)
    max_seq_length = opt.max_seq_len
    TOKEN_MASK_SHAPE = (1,800)
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


    ind1, ind2, lac12id, lac22id, dep2id1, dep2id2 = getMaskIndexWithLac(example,int(example['ratio']) < ratio)

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

    dep_feat = np.concatenate([np.array([0]) , dep2id1 , np.array([0]) , dep2id2 , np.array([0])])
    lac_feat = np.concatenate([np.array([0]) , lac12id , np.array([0]) , lac22id , np.array([0])]) #
    pad_dep_len = max_seq_length - len(dep_feat)
    pad_lac_len = max_seq_length - len(lac_feat)
    if pad_dep_len > 0:
        pad_1 = np.array([0]*pad_dep_len)
        dep_feat = np.concatenate([dep_feat,pad_1])
    if pad_lac_len > 0:
        pad_2 =  np.array([0]*pad_lac_len)
        lac_feat = np.concatenate([lac_feat, pad_2])


    sequence_length = len(input_ids)
    assert  len(input_ids) == len(token_type_ids)

    select_index = np.concatenate([np.ones(TOKEN_MASK_SHAPE),ind1,np.ones(TOKEN_MASK_SHAPE),ind2,
                                   np.ones(TOKEN_MASK_SHAPE)])
    pad_select_index = max_seq_length - 3 - len(ind1) - len(ind2)
    if pad_select_index > 0:
        pad_metrics = np.zeros((pad_select_index,800))
        select_index = np.concatenate([select_index,pad_metrics])

    label = np.array([example["label"]], dtype="int64")
    assert len(input_ids) == max_seq_length
    assert len(token_type_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length
    assert len(select_index) == max_seq_length
    assert len(lac_feat) == max_seq_length
    assert len(dep_feat) == max_seq_length
    qm = QMFeature(input_ids = input_ids,
                   token_type_ids = token_type_ids,
                   attention_mask = attention_mask,
                   select_tokens = select_index,
                   lac_ids = lac_feat,
                   dep_ids = dep_feat,
                   sequence_length = sequence_length,
                   labels = label)
    return qm



class BaseDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)
        self.input_ids = [torch.tensor(example.input_ids).long() for example in features]
        self.attention_mask = [torch.tensor(example.attention_mask).long() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        self.select_tokens = [torch.tensor(example.select_tokens).long() for example in features]
        self.lac_ids = [torch.tensor(example.lac_ids).long() for example in features]
        self.dep_ids = [torch.tensor(example.dep_ids).long() for example in features]
        self.labels = [torch.tensor(example.labels) for example in features]
    def __len__(self):
        return self.nums

class QMDataset(BaseDataset):
    def __init__(self,
                 features
                 ):
        super(QMDataset, self).__init__(features)
    def __getitem__(self, index):
        data = {'input_ids': self.input_ids[index],'attention_mask': self.attention_mask[index],
                'token_type_ids': self.token_type_ids[index], 'select_tokens': self.select_tokens[index],
                'lac_ids': self.lac_ids[index], 'dep_ids': self.dep_ids[index], 'labels': self.labels[index]}
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

def train(opt,model,train_dataset):
    swa_raw_model = copy.deepcopy(model)

    train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=opt.train_batch_size,
                              sampler=train_sampler,
                              num_workers=0)
    model, device = load_model_and_parallel(model, opt.gpu_ids)

    use_n_gpus = False
    if hasattr(model, "module"):
        use_n_gpus = True

    t_total = len(train_loader) * opt.train_epochs

    optimizer, scheduler = build_optimizer_and_scheduler(opt, model, t_total)
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

    log_loss_steps = 2000
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
                print('Step: %d / %d ----> total loss and acc: %.5f' % (global_step, t_total, avg_loss))
                avg_loss = 0.
            else:
                avg_loss += loss.item()
            if global_step % save_steps == 0:
                save_model(opt, model, global_step)
    swa(swa_raw_model, opt.output_dir, swa_start=opt.swa_start)
    print('Train done')




def main(opt):
    data = pd.read_csv('./sourceData/data_engineering/train_eda_ratio.csv')
    # dev_df = data.iloc[-28802:, :]
    # train_df = data.iloc[:-28802, :]
    train_df = data.iloc[-200:, :]
    dev_df = data.iloc[-200:, :]
    # train_features = []
    # for (ex_index, example) in tqdm(enumerate(train_df.iterrows()), desc="convert examples to features"):
    #     example = example[1]
    #     example = dict(example)
    #     train_feature = convert_example_with_lac(example,  opt)
    #     train_features.append(train_feature)
    # train_dataset = QMDataset(train_features)
    model = QuestionMatchingOtherTeatures(opt)
    # train(opt,model,train_dataset)

    #start dev
    dev_features = []
    for (ex_index, example) in tqdm(enumerate(dev_df.iterrows()), desc="convert dev_df examples to features"):
        example = example[1]
        example = dict(example)
        dev_feature = convert_example_with_lac(example, opt)
        dev_features.append(dev_feature)
    dev_dataset = QMDataset(dev_features)
    dev_loader = DataLoader(dev_dataset, batch_size=opt.eval_batch_size,
                            shuffle=False, num_workers=8)
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

    max_metric_str = f'Max Accuracy is: {max_acc}, in step {max_acc_step}'
    print(max_metric_str)
    print('*'*10 + " performance summary "+"*"*10)
    print(performance)



if __name__ == "__main__":
    args = TrainArgs().get_parser()
    args.output_dir = os.path.join(args.output_dir, args.bert_type)
    set_seed(seed=2022)
    if args.weight_decay:
        args.output_dir += '_wd'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)





