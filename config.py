
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

        parser.add_argument('--file_path',
                            default="/home/zmw/projects/question_matching/sourceData/data_engineering/train_eda_t_ratio_bak.csv",
                            help='the data dir of raw data')

        parser.add_argument('--train_file_path', default="/home/zmw/projects/question_matching/sourceData/data_engineering/train_eda_t_ratio.csv",
                            help='the data dir of raw data')
        parser.add_argument('--dev_file_path',
                            default="/home/zmw/projects/question_matching/sourceData/data_engineering/dev_eda_t_ratio.csv",
                            help='the data dir of raw data')
        parser.add_argument('--test_file_path',
                            default="/home/zmw/projects/question_matching/sourceData/data_engineering/test_t_ratio.csv",
                            help='the data dir of raw data')

        parser.add_argument('--output_dir', default='/home/zmw/big_space/zhangmeiwei_space/nlp_out/question_matching/basemodel/',
                            help='the output dir for model checkpoints')


        parser.add_argument('--bert_dir', default='/home/zmw/big_space/zhangmeiwei_space/pre_models/pytorch/chinese_roberta_wwm_ext_pytorch',
                            help='bert dir for ernie / roberta-wwm / uer / semi-bert')

        parser.add_argument('--bert_type', default='roberta_wwm',
                            help='roberta_wwm / ernie_1 / uer_large for bert')

        parser.add_argument('--overwrite_cache', default=False,
                            help='overwrite_cache')


        # other args
        parser.add_argument('--gpu_ids', type=str, default='3,4',#3,4
                            help='gpu ids to use, -1 for cpu, "1, 3" for multi gpu')

        parser.add_argument('--ratio', type=float, default=70,
                            help='ratio for sentence distence')
        parser.add_argument('--gru_hidden_size', type=int, default=8,
                            help='gru_hidden_size')
        parser.add_argument('--hidden_dropout_prob', type=float, default=0.2,
                            help='hidden_dropout_prob')
        parser.add_argument('--gru_layers', type=int, default=1,
                            help='gru_layers')
        parser.add_argument('--lac_vocab_size', type=int, default=49,
                            help='lac_vocab_size')
        parser.add_argument('--dep_vocab_size', type=int, default=29,
                            help='dep_vocab_size')
        parser.add_argument('--gru_emb_dim', type=int, default=4,
                            help='gru_emb_dim')
        parser.add_argument('--gru_dropout_rate', type=float, default=0.1,
                            help='gru_dropout_rate')



        # args used for train / dev
        parser.add_argument('--max_seq_len', default=128, type=int)

        parser.add_argument('--eval_batch_size', default=32, type=int)

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
                            help='learning rate for the bert module')#1e-5

        parser.add_argument('--other_lr', default=0.01, type=float,
                            help='learning rate for the module except bert')#2e-4

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



LAC_TABLE = {
    'n': '普通名词',
    'f': '方位名词',
    's': '处所名词',
    'nw': '作品名',
    'nz': '其他专名',
    'v': '普通动词',
    'vd': '动副词',
    'vn': '名动词',
    'a': '形容词',
    'ad': '副形词',
    'an': '名形词',
    'd': '副词',
    'm': '数量词',
    'q': '量词',
    'r': '代词',
    'p': '介词',
    'c': '连词',
    'u': '助词',
    'xc': '其他虚词',
    'w': '标点符号',
    'PER': '人名',
    'LOC': '地名',
    'ORG': '机构名',
    'TIME': '时间'
}
DEP_TABLE = {
'SBV':	'主谓关系',
'VOB':	'动宾关系'	,
'POB':	'介宾关系',
'ADV':	'状中关系',
'CMP':	'动补关系',
'ATT':	'定中关系',
'F':	'方位关系',
'COO':	'并列关系',
'DBL':	'兼语结构',
'DOB':	'双宾语结构',
'VV':	'连谓结构',
'IC':	'子句结构',
'MT':	'虚词成分',
'HED':	'核心关系',
}

def getLabels2Id(LABELS):
    '''BIO标记法，获得tag:id 映射字典'''
    labels = ['O']
    for label in LABELS:
        labels.append('B-' + label)
        labels.append('I-' + label)
    labels2id = {label: id_ for id_, label in enumerate(labels)}
    id2labels = {id_: label for id_, label in enumerate(labels)}
    return labels2id, id2labels
lac2id,id2lac = getLabels2Id(LAC_TABLE.keys())
dep2id,id2dep = getLabels2Id(DEP_TABLE.keys())


