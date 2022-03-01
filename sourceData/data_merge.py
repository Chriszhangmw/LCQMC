

import json


def list_dic_json(data_list,path):
    data_keys = data_list[0]
    keys = []
    for k,_ in data_keys.items():
        keys.append(k)
    with open(path,'w',encoding='utf-8') as w:
        for data in data_list:
            json.dump({k:data[k] for k in keys},w,ensure_ascii=False)
            w.write('\n')

def list_dic_json2(data_list,path):
    with open(path, 'w', encoding='utf-8') as w:
        json.dump(data_list,w,ensure_ascii=False)


def process_train():
    train_data = []
    with open('bq_corpus/bq_corpus/train.tsv','r',encoding='utf-8') as f1:
        data1 = f1.readlines()
        f1.close()
    for line in data1:
        line = line.strip().split('\t')
        if len(line) != 3:continue
        text1 = line[0]
        text2 = line[1]
        label = line[2]
        temp = {"text1":text1,"text2":text2,"label":label}
        train_data.append(temp)

    with open('lcqmc/lcqmc/train.tsv','r',encoding='utf-8') as f2:
        data2 = f2.readlines()
        f2.close()
    for line in data2:
        line = line.strip().split('\t')
        if len(line) != 3:continue
        text1 = line[0]
        text2 = line[1]
        label = line[2]
        temp = {"text1": text1, "text2": text2, "label": label}
        train_data.append(temp)

    with open('paws-x-zh/paws-x-zh/train.tsv','r',encoding='utf-8') as f3:
        data3 = f3.readlines()
        f3.close()
    for line in data3:
        line = line.strip().split('\t')
        if len(line) != 3:continue
        text1 = line[0]
        text2 = line[1]
        label = line[2]
        temp = {"text1": text1, "text2": text2, "label": label}
        train_data.append(temp)


    print(f'total training number is :',len(train_data))
    list_dic_json2(train_data, 'train.json')


def process_dev():
    dev_data = []
    with open('bq_corpus/bq_corpus/dev.tsv', 'r', encoding='utf-8') as f1:
        data1 = f1.readlines()
        f1.close()
    for line in data1:
        line = line.strip().split('\t')
        if len(line) != 3: continue
        text1 = line[0]
        text2 = line[1]
        label = line[2]
        temp = {"text1": text1, "text2": text2, "label": label}
        dev_data.append(temp)

    with open('lcqmc/lcqmc/dev.tsv', 'r', encoding='utf-8') as f2:
        data2 = f2.readlines()
        f2.close()
    for line in data2:
        line = line.strip().split('\t')
        if len(line) != 3: continue
        text1 = line[0]
        text2 = line[1]
        label = line[2]
        temp = {"text1": text1, "text2": text2, "label": label}
        dev_data.append(temp)

    with open('paws-x-zh/paws-x-zh/dev.tsv', 'r', encoding='utf-8') as f3:
        data3 = f3.readlines()
        f3.close()
    for line in data3:
        line = line.strip().split('\t')
        if len(line) != 3: continue
        text1 = line[0]
        text2 = line[1]
        label = line[2]
        temp = {"text1": text1, "text2": text2, "label": label}
        dev_data.append(temp)

    print(f'total dev number is :', len(dev_data))
    list_dic_json2(dev_data, './dev.json')

if __name__ =="__main__":
    process_train()
    # process_dev()



