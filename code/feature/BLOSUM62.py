# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 14:26:40 2022

@author: hp
"""

import pandas as pd
import re

def read_data(file_path):
    
    dataset = pd.read_csv(file_path)
    '''
    #查看并显示前三条记录
    print(dataset.iloc[0:3,:])
    print('-' * 30)
    
    #查看样本数和特征数
    print(dataset.shape)
    print('-' * 30)
    '''
    return(dataset)

def look_up_dict(seq, dictionary):
    m = len(seq)
    n = len(seq[0])
    code = pd.DataFrame()

    for i in range(m):
        # 对所有序列遍历
        scores_list = []
        for j in range(n):
            # 对所有位置遍历
            scores = dictionary[seq[i][j]]
            scores_list += scores
        code = code.append([scores_list], ignore_index=True)
    print(code.iloc[0:3,:])
    print(code.shape)
    return (code)

def BLOSUM62_dictionary():
    f=open('../../source_data/BLOSUM62.txt')
    BLOSUM62_matrix=pd.DataFrame()
    
    for line in f:
        if line.startswith('#'):
            #获取序列顺序
            aa_seq=list(line.replace('#', '').replace('\n', ''))
            #print(aa_seq)

        else:
            BlOSUM62_line=line.strip()
            BlOSUM62_line=re.split(r'[ ]+',BlOSUM62_line)
            #print(len(BlOSUM62_line))

            BLOSUM62_matrix=BLOSUM62_matrix.append([BlOSUM62_line],ignore_index=True)
    f.close()
    BLOSUM62_matrix['key']=aa_seq
    #print(BLOSUM62_matrix)
    
    #创建BLOSUM62字典
    BLOSUM62_dict=BLOSUM62_matrix.set_index('key').T.to_dict('list')
    #print(BLOSUM62_dict)
    
    return(BLOSUM62_dict)

def main():
    #获取BLOSUM62字典
    BLOSUM62_dict=BLOSUM62_dictionary()

    # 读取要编码的数据
    dataset = read_data('../../source_data/CD_All_Seq.csv')
    # 获取肽段序列
    seq = dataset['sequence']
    # 肽段长度K
    K = len(seq[0])
    # L=(K-1)/2
    # print(K)

    BLOSUM62_code=look_up_dict(seq,BLOSUM62_dict)
    print(BLOSUM62_code.shape)
    
    data_index=dataset.loc[:,['accession','position','type']]
    dataset=pd.concat([data_index,BLOSUM62_code],axis=1)
    print(dataset.shape)
    dataset.to_excel(excel_writer=r'../../result/feature/BLOSUM62_code.xlsx',index=False)
    
  
    
if __name__ == '__main__':

    main()