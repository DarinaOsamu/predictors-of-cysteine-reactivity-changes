# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:54:35 2022

@author: hp
"""

import pandas as pd

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

def main():
    
    #读取Atchley因子矩阵
    Atchley_factor=read_data('../../source_data/Atchley_factor.csv')
    #创建Atchley_factor的字典
    Atchley_dict=Atchley_factor.set_index('Amino acid').T.to_dict('list')
    #print(Atchley_dict)

    # 读取要编码的数据
    dataset = read_data('../../source_data/Drop_Seq.csv')
    # 获取肽段序列
    seq = dataset['sequence']
    # 肽段长度K
    K = len(seq[0])
    # L=(K-1)/2
    # print(K)

    Atchley_code=look_up_dict(seq,Atchley_dict)

    data_index=dataset.loc[:,['accession','position']]
    dataset=pd.concat([data_index,Atchley_code],axis=1)
    print(dataset.iloc[0:3, :])
    print(dataset.shape)
    dataset.to_csv('../../source_data/feature/Atchley_code.csv',index=False)

    
if __name__ == '__main__':

    main()