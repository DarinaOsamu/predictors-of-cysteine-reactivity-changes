# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 08:28:12 2022

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

def H_num(seq,Hn):
    H_list=[]
    K=len(seq[0])
    for i in seq:
        in_list=[]
        for j in range(K):
            if i[j] in Hn:
                in_list+=[1]
            else:
                in_list+=[0]
        #print(in_list)
        H_score=sum(in_list)/K
        H_list+=[H_score]
    #print(len(H_list))
    #print(H_list)
    return(H_list)
                
    

def main():
    #氨基酸的理化性质
    C1=['A','F','G','I','L','M','P','V','W']
    C2=['C','N','Q','S','T','Y']
    C3=['K','H','R']
    C4=['D','E']
    
    #按理化性质分组,记录会记为1的组别,空缺符X始终记0
    H1=C1+C2
    H2=C1+C3
    H3=C1+C4
    
    #读取要编码的数据
    dataset=read_data('../../source_data/CD_All_Seq.csv')
    #获取肽段序列
    seq=dataset['sequence']
    #肽段长度K
    K=len(seq[0])
    #print(K)
    L=round((K-1)/2)
    
    #设定子序列个数J
    J=5
    #子序列划分规则
    #在原规则上进行了改进，取中心位点两侧的肽段
    Lj_list=[]
    for j in range(J):
        jth_L=round(L*(j+1)/J)
        Lj_list+=[jth_L]
    #print(Lj_list)
    
    #获取子序列
    seq_list=pd.DataFrame()
    for i in seq:
        seq_i_list=[]
        for j in range(J):
            Lj=Lj_list[j]
            #print(Lj)
            #print(L)
            seq_i_list+=[i[L-Lj:L+Lj+1]]
        #print(seq_i_list)
        seq_list=seq_list.append([seq_i_list],ignore_index=True)
    #print(seq_list.shape)
    
    EBGW_code=pd.DataFrame()
    for j in range(J):
        seq=seq_list.iloc[:,j]
        #获取序列的得分
        H1_list=pd.Series(H_num(seq,H1))
        H2_list=pd.Series(H_num(seq,H2))
        H3_list=pd.Series(H_num(seq,H3))
        #print(H1_list[0:3],H2_list[0:3],H3_list[0:3])
        H_list=pd.concat([H1_list,H2_list,H3_list],axis=1)
        EBGW_code=pd.concat([EBGW_code,H_list],axis=1)
    print(EBGW_code.shape)
    
    data_index=dataset.loc[:,['accession','position','type']]
    dataset=pd.concat([data_index,EBGW_code],axis=1)
    print(dataset.shape)
    dataset.to_excel(excel_writer=r'../../result/feature/EBGW_code.xlsx',index=False)


if __name__ == '__main__':

    main()