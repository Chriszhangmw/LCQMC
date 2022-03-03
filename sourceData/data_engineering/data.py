
import os
import re
import warnings

import jieba
import numpy as np
import paddle
import pandas as pd
from LAC import LAC
from fuzzywuzzy import fuzz
from paddle.io import Dataset


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

# 装载分词模型
lac = LAC(mode='lac')

class RawData(object):
    def __init__(self, ):
        BQ_PATH = '../bq_corpus/bq_corpus/'
        self.bq_dev = pd.read_csv(os.path.join(BQ_PATH, 'dev.tsv'), sep='\t', error_bad_lines=False, header=None,
                             names=['text_a', 'text_b', 'label'])
        print('self.bq_dev number :',len(self.bq_dev))
        self.bq_test = pd.read_csv(os.path.join(BQ_PATH, 'test.tsv'), sep='\t', error_bad_lines=False, header=None,
                              names=['text_a', 'text_b', 'label'])
        print('self.bq_test number :', len(self.bq_test))
        self.bq_train = pd.read_csv(os.path.join(BQ_PATH, 'train.tsv'), sep='\t', error_bad_lines=False, header=None,)
        print('self.bq_train number :', len(self.bq_train))
        self.bq_train.columns =  ['text_a','text_b','label']

        self.bq_dev['domain'] = 0
        self.bq_train['domain'] = 0
        self.bq_test['domain'] = 0

        LCQMC_PATH = '../lcqmc/lcqmc/'
        self.lcqmc_dev = pd.read_csv(os.path.join(LCQMC_PATH, 'dev.tsv'), sep='\t', error_bad_lines=False, header=None,
                                names=['text_a', 'text_b', 'label'])
        print('self.lcqmc_dev number :', len(self.lcqmc_dev))
        self.lcqmc_test = pd.read_csv(os.path.join(LCQMC_PATH, 'test.tsv'), sep='\t', error_bad_lines=False, header=None,
                                 names=['text_a', 'text_b', 'label'])
        print('self.lcqmc_test number :', len(self.lcqmc_test))
        self.lcqmc_train = pd.read_csv(os.path.join(LCQMC_PATH, 'train.tsv'), sep='\t', error_bad_lines=False, header=None,
                                  names=['text_a', 'text_b', 'label'])
        print('self.lcqmc_train number :', len(self.lcqmc_train))
        self.lcqmc_dev['domain'] = 1
        self.lcqmc_test['domain'] = 1
        self.lcqmc_train['domain'] = 1

        OPPO_PATH = '../oppo/'
        self.oppo_dev = pd.read_csv(os.path.join(OPPO_PATH, 'dev.tsv'), sep='\t', error_bad_lines=False, header=None,
                               names=['text_a', 'text_b', 'label'])
        print('self.oppo_dev number :', len(self.oppo_dev))
        self.oppo_train = pd.read_csv(os.path.join(OPPO_PATH, 'train.tsv'), sep='\t', error_bad_lines=False, header=None,
                                 names=['text_a', 'text_b', 'label'])
        print('self.oppo_train number :', len(self.oppo_train))
        self.oppo_test = pd.read_csv(os.path.join(OPPO_PATH, 'test.tsv'), sep='\t', error_bad_lines=False,
                                      header=None,
                                      names=['text_a', 'text_b'])
        print('self.oppo_test number :', len(self.oppo_test))
        self.oppo_dev['domain'] = 2
        self.oppo_train['domain'] = 2
        self.oppo_test['domain'] = 2

        self.test_data = pd.read_csv('../test.tsv', sep= '\t', header=None)
        self.test_data.columns = ['text_a', 'text_b']


        self.train_data = pd.concat(
            [self.bq_train, self.bq_test,
             self.lcqmc_test, self.lcqmc_train, self.oppo_train,self.oppo_test]).dropna().reset_index(drop=True)

        self.dev_data = pd.concat([self.bq_dev, self.lcqmc_dev, self.oppo_dev]).dropna().reset_index(drop=True)

    def getTrain(self,debug = False):
        return self.train_data if not debug else self.bq_train

    def getDev(self,debug = False):

        return self.dev_data if not debug else self.bq_dev

    def getTest(self):

        return self.test_data

    def __getitem__(self, item):
        pass


class QMSet(Dataset):
    '''
    提取一些特征、以及将每行数据转换为json形式
    '''
    def __init__(self,df,choice=1):

        if choice==1:
            print('init data set ,lac feat!')
            for col in ['deprel_a', 'deprel_b', 'postag_a', 'postag_b', 'word_a', 'word_b']:
                if isinstance(df[col].iloc[0],str):
                    df[col] = df[col].apply(lambda x:eval(x))
            df['ratio'] = df.apply(lambda x: fuzz.ratio(x.text_a, x.text_b), axis=1) #编辑距离
            self.data = df
        elif choice==2:
            print('init data set origin feat!')
            df['text_a'] = df['text_a'].apply(removeOral)
            df['text_b'] = df['text_b'].apply(removeOral)
            df['ratio'] = df.apply(lambda x: fuzz.ratio(x.text_a, x.text_b), axis=1)  # 编辑距离
            self.data = df
        elif choice ==3:
            print('init data set origin feat! , select ratio > 70')
            df['text_a'] = df['text_a'].apply(removeOral)
            df['text_b'] = df['text_b'].apply(removeOral)
            df['ratio'] = df.apply(lambda x: fuzz.ratio(x.text_a, x.text_b), axis=1)  # 编辑距离
            df = df[df.ratio > 70]
            self.data = df
        elif choice ==4:
            print('init data set origin feat! , select ratio <= 70')
            df['text_a'] = df['text_a'].apply(removeOral)
            df['text_b'] = df['text_b'].apply(removeOral)
            df['ratio'] = df.apply(lambda x: fuzz.ratio(x.text_a, x.text_b), axis=1)  # 编辑距离
            df = df[df.ratio <= 70]
            self.data = df
    def __getitem__(self, item):
        '''

        :param item:返回第item个元素的json 字典形式
        :return:
        '''
        sample = self.data.iloc[item].to_dict()
        return sample
    def __len__(self,):
        return self.data.shape[0]

def getMaskIndexWithLac(example ,all_select):
    '''
    使用百度的lac分词器
    :param x1:
    :param x2:
    :return:
    '''
    #分词
    x1 = example['text_a']
    x2 = example['text_b']

    lac1 = example['postag_a']
    lac2 = example['postag_b']

    s1 = example['word_a']
    s2 = example['word_b']

    deprel1 = example['deprel_a']
    deprel2 = example['deprel_b']


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

def getMaskIndex(x1,x2):
    '''
    jieba 分词器
    :param x1:
    :param x2:
    :return:
    '''
    index1 = np.zeros((len(x1),768))
    index2 = np.zeros((len(x2),768))
    s1,s2  = list(jieba.cut(x1)),list(jieba.cut(x2))

    s1 = [(i,v) for i, v in enumerate(s1)]
    s2 = [(i,v) for i, v in enumerate(s2)]

    i = 0
    for n,w in enumerate(s1):
        w = w[1]
        if (n,w) not in s2:
            for j in range(i,i+len(w)):
                index1[j,:] = 1
        i += len(w)
    i = 0
    for n,w in enumerate(s2):
        w = w[1]
        if (n,w) not in s1:
            for j in range(i,i+len(w)):
                index2[j,:] = 1
        i += len(w)
    return index1,index2

def removeOral(string):
    startslist = ['有谁知道','大家知道','你知道','谁知道','谁了解','有谁了解','大家了解','你了解']
    endslist = ['吗']
    s,e = False,False
    for st in startslist:
        if string.startswith(st):
            s = True
            break
    for end in endslist:
        if string.endswith(end):
            e = True
            break
    if s and e:
        tmp = re.findall(st + '(.*)' + end,string)[0]
        if len(tmp)==0:
            return 'None'
        else:
            return tmp
    return string

def read_text_pair_base(data_path, is_test=False):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if is_test == False:
                if len(data) != 3:
                    continue
                yield {'query1': data[0], 'query2': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    continue
                yield {'query1': data[0], 'query2': data[1]}


def markSentenceWithDiff(s1, s2,ratio ):
    if fuzz.ratio(s1, s2) < ratio:
        return s1, s2
    s1, s2 = list(jieba.cut(s1)), list(jieba.cut(s2))
    s1 = [(i,v) for i ,v in enumerate(s1)]
    s2 = [(i,v) for i,v in enumerate(s2)]

    new_s1, new_s2 = '', ''
    for i,w in enumerate(s1):
        if (i,w) in s2:
            new_s1 += w
        else:
            tmp = '@' + w + '@'
            new_s1 += tmp

    for i,w in enumerate(s2):
        if (i,w) in s1:
            new_s2 += w
        else:
            tmp = '#' + w + '#'
            new_s2 += tmp
    return new_s1, new_s2

from tqdm import tqdm
if __name__ == "__main__":
    #训前的配置
    rd = RawData()
    train_df = rd.getTrain()
    dev_df = rd.getDev()
    for (ex_index, example) in tqdm(enumerate(train_df.iterrows()), desc="convert examples to features"):
        if ex_index < 5:
            print(example)
            print(example["text_a"])
