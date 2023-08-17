# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 19:03:16 2022

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

def get_dict(dataset,mod_type):
    dataset=dataset.loc[:,['Entry','Modified residue']]
    
    #创建字典
    dictionary=dataset.set_index('Entry').T.to_dict('list')
    
    for index,row in dataset.iterrows():
        site_str=row['Modified residue']
        site_str=str(site_str)
        #分割出需要的修饰位点信息
        items=site_str.replace(' ', '').split(';')
        #print(items)
        
        mod_list=[]
        for i in items:
            if pd.isnull(i)==False:
                #有位点
                if i.startswith('MOD_RES'):
                    position=i.replace('MOD_RES', '')
                    if position.isdigit():
                        position=int(position)

                elif i.startswith('/note='):
                    mod=i.replace('/note=', '').replace('"','')
                    #取标准位点
                    if type(position)==int:
                        #取指定形式
                        if mod==mod_type:
                            mod_list.append(position)
        #print(mod_list)
        #得到位点数组，设为键的值
        dictionary[row['Entry']]=mod_list
        
    '''
    #查看并显示第一个键值对
    for key, value in dictionary.items():
        print(key,value)
        break
    
    #查看前三个键值对
    print(list(dictionary.items())[:5])
    
    #查看字典长度
    print(len(dictionary))
    print('-' * 30)
    '''
    #返回字典
    return(dictionary)

def look_up_dict(dataset,dictionary):
    accession=dataset['accession']
    position=dataset['position']
    new_col=[]
    for i in range(dataset.shape[0]):
        key=accession[i]
        #print(key)
        if key in dictionary.keys():
            if dictionary[key]==[]:
                new_col.append('No_Site')
            else:
                #找到字典值数列中最小差值
                P_site=dictionary[key]
                distance=list(map(lambda x:abs(x-position[i]),P_site))
                min_dis=min(distance)
                new_col.append(min_dis)
        else:
            new_col.append('No_Data')
    #print(new_col)
    return(new_col)

    
def main():
    #读取STP位点数据
    STP_site=read_data('../../source_data/uniprot-pho.csv')
    #分别创建SP_site/TP_site的字典
    SP_dict=get_dict(STP_site,'Phosphoserine')
    TP_dict=get_dict(STP_site,'Phosphothreonine')

    #读取要查找的数据
    dataset=read_data('../../source_data/Drop_seq.csv')
    SP_col=look_up_dict(dataset,SP_dict)
    TP_col=look_up_dict(dataset,TP_dict)
    
    #print(SP_col)
    
    #写入数据
    dataset['SP_min_dis']=SP_col
    dataset['TP_min_dis']=TP_col
    #print(dataset.iloc[0:3,:])
    #print(dataset.shape)
    dataset.to_csv('../../source_data/All_Seq_STPdis.csv',index=False)

if __name__ == '__main__':

    main()