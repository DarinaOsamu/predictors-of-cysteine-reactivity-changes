# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 19:36:40 2022

@author: hp
"""

import pandas as pd
from sklearn.model_selection import train_test_split

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

def main():
    
    ##############################
    
    #读取特征编码后的数据
    PSSM_code=read_data('../source_data/feature/PSSM_code.csv')
    CKSAAP_0_code=read_data('../source_data/feature/CKSAAP_0_code.csv')
    CKSAAP_1_code=read_data('../source_data/feature/CKSAAP_1_code.csv')
    Atchley_code=read_data('../source_data/feature/Atchley_code.csv')
    EBGW_code=read_data('../source_data/feature/EBGW_code.csv')
    BLOSUM62_code=read_data('../source_data/feature/BLOSUM62_code.csv')
    STP_dis_code=read_data('../source_data/feature/STP_dis_code.csv')
    In_Disorder_code=read_data('../source_data/feature/In_Disorder_code.csv')
    
    Feature_list=[PSSM_code,CKSAAP_0_code,CKSAAP_1_code,Atchley_code,EBGW_code,BLOSUM62_code,STP_dis_code,In_Disorder_code]
    data_index=In_Disorder_code.loc[:,['accession','position','type']]
    #记录各特征维度数
    dimen=[]
    n=0
    for i in Feature_list:
        #print(i.shape[1]-3)
        Feature_list[n]=Feature_list[n].iloc[:,3:]
        dimen+=[i.shape[1]-3]
        n+=1
    print(dimen)
    #总特征数
    dimen_sum=sum(dimen)
    print(dimen_sum)
    '''
    for i in Feature_list:
        print(i.shape[1])
    '''
    
    #特征合并
    dataset=pd.concat(Feature_list,axis=1)
    dataset=pd.concat([data_index,dataset],axis=1)
    print(dataset.shape)
    print(dataset.iloc[0:3,:])
    
    ##############################
    '''
    #划分正负集
    dataset_incre=dataset.loc[dataset['type']=='Increased']
    print(dataset_incre.shape)
    dataset_decre=dataset.loc[dataset['type']=='Decreased']
    print(dataset_decre.shape)
    
    #有改变集，不需要
    #dataset_pos=pd.concat([dataset_incre,dataset_decre],axis=0,ignore_index=True)
    #print(dataset_pos.shape)
    
    dataset_neg=dataset.loc[dataset['type']=='Unchanged']
    print(dataset_neg.shape)
    
    dataset_list=[dataset_incre,dataset_decre,dataset_neg]
    
    #划分X,y
    X_list=[]
    y_list=[]
    n=0
    for i in dataset_list:
        X_list+=[i.iloc[:,3:]]
        y_list+=[i.loc[:,'type']]
        #print(X_list[n],y_list[n])
        n+=1
        
    #20%的独立验证集
    X_train_test_list=[]
    X_validation_list=[]
    y_train_test_list=[]
    y_validation_list=[]
    for n in range(3):
        X_train_test,X_validation,y_train_test,y_validation=train_test_split(X_list[n],y_list[n],test_size=0.2,random_state=0)
        
        X_train_test_list+=[X_train_test]
        X_validation_list+=[X_validation]
        
        y_train_test_list+=[y_train_test]
        y_validation_list+=[y_validation]
        
        #查看各验证集的大小
        print(X_train_test.shape,X_validation.shape,len(y_train_test),len(y_validation))
    
    #合并得到总的独立验证集
    X_validation_all=pd.concat(X_validation_list,ignore_index=True)
    y_validation_all=pd.concat(y_validation_list,ignore_index=True)
    print(X_validation_all.shape,len(y_validation_all))
    
    #保存独立验证集数据
    Validation_all=pd.concat([X_validation_all,y_validation_all],axis=1)
    print(Validation_all.shape)
    #Validation_all.to_excel(excel_writer=r'../result/dataset/validation.xlsx',index=False)
    
    #保存上升训练集的X
    #X_train_test_list[0].to_excel(excel_writer=r'../result/dataset/train_test_Incre.xlsx',index=False)
    #保存下降训练集的X
    #X_train_test_list[1].to_excel(excel_writer=r'../result/dataset/train_test_Decre.xlsx',index=False)
    #保存不变训练集的X
    #X_train_test_list[2].to_excel(excel_writer=r'../result/dataset/train_test_Unchange.xlsx',index=False)
    '''
    ##############################

if __name__ == '__main__':

    main()