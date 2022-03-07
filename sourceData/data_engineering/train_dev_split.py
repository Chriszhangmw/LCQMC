

'''

这里重新划分验证集，测试集

'''
import pandas as pd


data = pd.read_csv('./all_eda_t_ratio.csv',sep='\t')
dev_df = data.iloc[-28802:, :]
train_df = data.iloc[:-28802, :]

train_df.to_csv('./train_eda_t_ratio.csv',index = None,sep='\t')
dev_df.to_csv('./dev_eda_t_ratio.csv',index = None,sep='\t')



