
import json

test = open('./test.tsv','w',encoding='utf-8')
dev = open('./dev.tsv','w',encoding='utf-8')
train = open('./train.tsv','w',encoding='utf-8')

with open('./oppp.json','r',encoding='utf-8') as f:
    data = json.load(f)
print(data.keys())
for k,v in data.items():
    if k == 'train':
        for traindata in v:
            text1 = traindata['q1']
            text2 = traindata['q2']
            label = traindata['label']
            train.write(text1 + '\t' + text2 + '\t' + str(label) + '\n')
    elif k == 'dev':
        for devdata in v:
            text1 = devdata['q1']
            text2 = devdata['q2']
            label = devdata['label']
            dev.write(text1 + '\t' + text2 + '\t' + str(label) + '\n')
    else:
        for testdata in v:
            text1 = testdata['q1']
            text2 = testdata['q2']
            test.write(text1 + '\t' + text2 +  '\n')













