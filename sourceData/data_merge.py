

import json

json_file_path = './result.json'
json_file = open(json_file_path, mode='w')
train_data = []
with open('bq_corpus/bq_corpus/train.tsv','r',encoding='utf-8') as f1:
    data1 = f1.readlines()
    f1.close()
for line in data1:
    line = line.strip().split('\t')
    assert len(line) == 3
    text1 = line[0]
    text2 = line[1]
    label = line[2]
    train_data.append([{"text1":text1,"text2":text2,"label":label}])

with open('lcqmc/lcqmc/train.tsv','r',encoding='utf-8') as f2:
    data2 = f2.readlines()
    f2.close()
for line in data2:
    line = line.strip().split('\t')
    assert len(line) == 3
    text1 = line[0]
    text2 = line[1]
    label = line[2]
    train_data.append([{"text1":text1,"text2":text2,"label":label}])

with open('paws-x-zh/paws-x-zh/train.tsv','r',encoding='utf-8') as f3:
    data3 = f3.readlines()
    f3.close()
for line in data3:
    line = line.strip().split('\t')
    assert len(line) == 3
    text1 = line[0]
    text2 = line[1]
    label = line[2]
    train_data.append([{"text1":text1,"text2":text2,"label":label}])














