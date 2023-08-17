# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:11:03 2022

@author: hp
"""

import pandas as pd
from math import log

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

def PSSM_dictionary(seq,K):
    PSSM_dict={}
    aa_seq='ARNDCQEGHILKMFPSTWVYX'
    m=len(seq)
    n=len(aa_seq)
    #print(m)
    for k in aa_seq:
        aa_frequency=[0]*K
        for i in range(K):
            #对每个位置遍历
            for j in seq:
                if j[i]==k:
                    aa_frequency[i]+=1
        #得到第k个氨基酸的位置频度
        #print(aa_frequency)
        #得到第k个氨基酸的位置概率
        aa_probability=list(map(lambda x:x/m,aa_frequency))
        #print(aa_probability)
        #得到第k个氨基酸的位置比重,b=1/n
        aa_weight=list(map(lambda x:log(x*n+1),aa_probability))
        #print(aa_weight)
        #字典赋键
        PSSM_dict[k]=aa_weight
    #print(PSSM_dict)
    return(PSSM_dict)
        
def look_up_dict(seq,dictionary):
    m=len(seq)
    n=len(seq[0])
    code=pd.DataFrame()
    
    for i in range(m):
        #对所有序列遍历
        scores_list=[]
        for j in range(n):
            #对所有位置遍历
            scores=dictionary[seq[i][j]][j]
            scores_list+=[scores]
        code=code.append([scores_list],ignore_index=True)
    #print(code.iloc[0:3,:])
    #print(code.shape)
    return(code)

def main():
    #读取要编码的数据
    dataset=read_data('../../source_data/Drop_Seq.csv')
    #获取肽段序列
    seq=dataset['sequence']
    #肽段长度K
    K=len(seq[0])
    #L=(K-1)/2
    #print(K)
    
    PSSM_dict=PSSM_dictionary(seq,K)
    PSSM_code=look_up_dict(seq,PSSM_dict)
    
    data_index=dataset.loc[:,['accession','position']]
    dataset=pd.concat([data_index,PSSM_code],axis=1)
    print(dataset.shape)
    dataset.to_csv('../../source_data/feature/PSSM_code.csv',index=False)

if __name__ == '__main__':

    main()