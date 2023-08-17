# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 07:41:32 2022

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

def look_up_dict(dataset,dictionary):
    accession=dataset['accession']
    position=dataset['position']
    new_col=[]
    for i in range(dataset.shape[0]):
        key=accession[i]+'_C'+str(position[i])
        #print(key)
        if key in dictionary.keys():
            new_col.append(dictionary[key])
        else:
            new_col.append(0)
    #print(new_col)
    return(new_col)

def main():
    #创建In_disorder的字典
    f=open('../../source_data/IUPred2A.result')
    #建立字典
    disorder_dict={}
    for line in f:
        if line.startswith('>'):
            #取蛋白质编号
            name=line.replace('>', '').replace('\n', '').split('|')[1]
            #print(name)
        elif line[0].isdigit():
            site=line.replace('\n', '').split('\t')[0]
            residue=line.replace('\n', '').split('\t')[1]
            score=line.replace('\n', '').split('\t')[2]

            #只提取半胱胺酸残基
            if residue=='C':
                #蛋白质编号+_C+位点作为键名
                key=name+'_'+residue+str(site)
                disorder_dict[key]=score
    f.close()
    
    print(len(disorder_dict))
    #print(disorder_dict)

    #读取要查找的数据
    dataset=read_data('../../source_data/Drop_Seq.csv')
    In_disorder_col=look_up_dict(dataset,disorder_dict)

    #写入数据
    dataset['In_disorder_score']=In_disorder_col
    dataset=dataset.drop(columns=['sequence'])
    print(dataset.iloc[0:3,:])
    print(dataset.shape)
    dataset.to_csv('../../source_data/feature/In_Disorder_code.csv',index=False)

if __name__ == '__main__':

    main()