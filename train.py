
import os
from dev_metrics import mc_evaluation
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from data import DataProcessor,convert_examples_to_features,build_dataset
from trainer import train
from config import TrainArgs
from model import QuestionMatching
from functions_utils import set_seed, get_model_path_list, load_model_and_parallel


def train_base(opt, train_examples, dev_info=None):
    train_features = convert_examples_to_features(train_examples, opt.bert_dir,opt.max_seq_len)

    train_dataset = build_dataset(train_features, 'train')

    model = QuestionMatching(opt.bert_dir,opt.rdrop_coef)

    train(opt, model, train_dataset)

    if dev_info is not None:
        dev_examples = dev_info

        dev_features = convert_examples_to_features(dev_examples, opt.bert_dir, opt.max_seq_len)
        dev_dataset = build_dataset( dev_features, 'dev')
        dev_loader = DataLoader(dev_dataset, batch_size=opt.eval_batch_size,
                                shuffle=False, num_workers=8)

        model_path_list = get_model_path_list(opt.output_dir)

        max_acc = 0.
        max_acc_step = 0

        for idx, model_path in enumerate(model_path_list):

            tmp_step = model_path.split('/')[-2].split('-')[-1]

            model, device = load_model_and_parallel(model, opt.gpu_ids[0],
                                                    ckpt_path=model_path)
            acc = mc_evaluation(model,dev_loader,device)
            if acc > max_acc:
                max_acc = acc
                max_acc_step = tmp_step

        max_metric_str = f'Max Accuracy is: {max_acc}, in step {max_acc_step}'
        print(max_metric_str)


def training(opt):
    processor = DataProcessor()

    train_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'dev.json'))
    train_examples = processor.get_train_examples(train_raw_examples)

    dev_info = None
    if opt.eval_model:
        dev_raw_examples = processor.read_json(os.path.join(opt.raw_data_dir, 'dev.json'))
        dev_info = processor.get_dev_examples(dev_raw_examples)

    train_base(opt, train_examples, dev_info)




if __name__ == '__main__':
    args = TrainArgs().get_parser()
    args.output_dir = os.path.join(args.output_dir, args.bert_type)
    set_seed(seed=2022)
    if args.weight_decay:
        args.output_dir += '_wd'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    training(args)
