import pandas as pd

from tqdm import tqdm
from xpinyin import Pinyin
p = Pinyin()
import argparse
def process(train_data):
    train_data['text_a_pinyin'] = train_data.apply(lambda x: p.get_pinyin(x.text_a), axis=1)
    train_data['text_b_pinyin'] = train_data.apply(lambda x: p.get_pinyin(x.text_b), axis=1)
    return train_data
def rule(test_data):
    res = []
    for i, line in tqdm(test_data.iterrows(), total=test_data.shape[0], desc='规则处理ing'):
        if line['text_a_pinyin'] == line['text_b_pinyin']:
            test_data.loc[i, 'label'] = 1
            res.append(1)
        else:
            if line["ratio"] > 80:
                test_data.loc[i, 'label'] = 1
                res.append(1)
            else:
                test_data.loc[i, 'label'] = 0
                res.append(0)
    return res

def add_pinyin(path):
    test_data = pd.read_csv(path, sep='\t')
    test_data = process(test_data)
    res = rule(test_data)
    return res


def main():
    path1 = './predict.csv'
    path2 = '/home/zmw/projects/question_matching/sourceData/data_engineering/test_t_ratio.csv'
    rule_res = add_pinyin(path2)
    model_res = [line.strip() for line in open(path1,'r',encoding='utf-8').readlines()]
    assert len(rule_res) == len(model_res)
    final_res1 = open('./final1.csv','w',encoding='utf-8')
    final_res2 = open('./final2.csv', 'w', encoding='utf-8')
    # method 1,尊重rules的1
    for a,b in zip(rule_res,model_res):
        if a == 1:
            final_res1.write('1'+'\n')
        else:
            final_res1.write(str(b)+'\n')
    # method 1,尊重rules的2
    for a, b in zip(rule_res, model_res):
        if a == 0:
            final_res2.write('0' + '\n')
        else:
            final_res2.write(str(b) + '\n')





if __name__ == "__main__":
    main()












