
import argparse


class BaseArgs:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        # args for path
        parser.add_argument('--raw_data_dir', default='/home/zmw/projects/question_matching/sourceData',
                            help='the data dir of raw data')

        parser.add_argument('--output_dir', default='/home/zmw/big_space/zhangmeiwei_space/nlp_out/question_matching/',
                            help='the output dir for model checkpoints')

        parser.add_argument('--bert_dir', default='/home/zmw/big_space/zhangmeiwei_space/pre_models/pytorch/chinese_roberta_wwm_ext_pytorch',
                            help='bert dir for ernie / roberta-wwm / uer / semi-bert')

        parser.add_argument('--bert_type', default='roberta_wwm',
                            help='roberta_wwm / ernie_1 / uer_large for bert')

        # other args
        parser.add_argument('--gpu_ids', type=str, default='3,4',
                            help='gpu ids to use, -1 for cpu, "1, 3" for multi gpu')

        # args used for train / dev
        parser.add_argument('--max_seq_len', default=128, type=int)

        parser.add_argument('--eval_batch_size', default=16, type=int)

        parser.add_argument('--swa_start', default=1, type=int,
                            help='the epoch when swa start')

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()


class TrainArgs(BaseArgs):
    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        parser = BaseArgs.initialize(parser)

        parser.add_argument('--train_epochs', default=10, type=int,
                            help='Max training epoch')

        parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')
        parser.add_argument('--rdrop_coef', default=0.1, type=float,
                            help='rdrop_coef')

        parser.add_argument('--lr', default=5e-5, type=float,
                            help='learning rate for the bert module')

        parser.add_argument('--other_lr', default=2e-4, type=float,
                            help='learning rate for the module except bert')

        parser.add_argument('--max_grad_norm', default=1.0, type=float,
                            help='max grad clip')

        parser.add_argument('--warmup_proportion', default=0.1, type=float)

        parser.add_argument('--weight_decay', default=0., type=float)

        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--train_batch_size', default=32, type=int)
        parser.add_argument('--eval_model', default=True, action='store_true',
                            help='whether to eval model after training')
        parser.add_argument('--attack_train', default='fgm', type=str,
                            help='fgm / pgd attack train when training')
        return parser




