# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 07:12:13 2022

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

def CKSAAP(seq,K,K_space,Dipeptide):
    # 计算氨基酸对数目
    NN = K - K_space - 1
    print(NN)

    CKSAAP_code = pd.DataFrame()
    for i in seq:
        N_list = [0] * len(Dipeptide)

        for j in range(NN):
            j = i[j] + i[j + K_space + 1]
            for k in Dipeptide:
                if j == k:
                    index = Dipeptide.index(k)
                    # print(N_list[index])
                    N_list[index] += 1
        # print(N_list)
        Naa_list = list(map(lambda x: round(x / NN, 2), N_list))
        # print(len(Naa_list))
        CKSAAP_code = CKSAAP_code.append([Naa_list], ignore_index=True)
    # print(CKSAAP_code)
    return(CKSAAP_code)

def data_connect(dataset,CKSAAP_code):
    data_index = dataset.loc[:, ['accession', 'position']]
    dataset = pd.concat([data_index, CKSAAP_code], axis=1)
    print(dataset.shape)
    #print(dataset.iloc[0:3])
    return dataset

def main():
    aa_seq='ARNDCQEGHILKMFPSTWVYX'
    #aaseq=list(aa_seq)
    #获取氨基酸对
    Dipeptide=[]
    for i in aa_seq:
        for j in aa_seq:
            Dipeptide+=[i+j]
    #print(len(Dipeptide))
    
    #读取要编码的数据
    dataset=read_data('../../source_data/Drop_Seq.csv')
    #获取肽段序列
    seq=dataset['sequence']
    #肽段长度K
    K=len(seq[0])
    #print(K)

    CKSAAP_code0=CKSAAP(seq, K, K_space=0, Dipeptide=Dipeptide)
    CKSAAP_code1 = CKSAAP(seq, K, K_space=1, Dipeptide=Dipeptide)

    dataset0=data_connect(dataset,CKSAAP_code0)
    dataset1 = data_connect(dataset, CKSAAP_code1)

    dataset0.to_csv('../../source_data/feature/CKSAAP_0_code.csv',index=False)
    dataset1.to_csv('../../source_data/feature/CKSAAP_1_code.csv',index=False)
        
    
if __name__ == '__main__':

    main()