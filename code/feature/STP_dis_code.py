# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 06:24:52 2022

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

def STP_dis_code(P_min_dis):
    
    P_dis_score=[]
    #转化为倒数
    for i in P_min_dis:
        if i=='No_Site':
            P_dis_score+=[0]
        else:
            i=float(i)
            P_dis_score+=[round(1/i,3)]
    #print(P_dis_score)
    return P_dis_score

def main():
    dataset=read_data('../../source_data/All_Seq_STPdis.csv')
    SP_min_dis=dataset['SP_min_dis']
    TP_min_dis=dataset['TP_min_dis']
    
    SP_dis_score=STP_dis_code(SP_min_dis)
    TP_dis_score=STP_dis_code(TP_min_dis)
    STP_dis_score=list(map(lambda x,y:max(x,y),SP_dis_score,SP_dis_score))
    
    #写入数据
    dataset=dataset.loc[:,['accession','position','type']]
    dataset['SP_dis_score']=SP_dis_score
    dataset['TP_dis_score']=TP_dis_score
    dataset['STP_dis_score']=STP_dis_score
    #print(dataset.iloc[0:3,-3:])
    #print(dataset.shape)
    dataset.to_excel(excel_writer=r'../../result/feature/STP_dis_code.xlsx',index=False)
    
    
    
if __name__ == '__main__':

    main()