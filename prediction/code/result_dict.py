# -*- coding: utf-8 -*-

import pandas as pd



def read_data(file_path):
    dataset = pd.read_csv(file_path)
    '''
    #查看并显示前三条记录
    print(dataset.iloc[0:3,:])
    print('-' * 30)
    '''
    # 查看样本数和特征数
    print(dataset.shape)
    print('-' * 30)

    return (dataset)


def get_dict(dataset):

    dataset.columns=['accession','position','result']

    # 创建字典
    dictionary={}
    for row in dataset.index:
        accession=dataset.loc[row,'accession']
        position=dataset.loc[row,'position']
        result=dataset.loc[row,'result']
        if accession in dictionary:
            dictionary[accession].append([position,result])
        else:
            dictionary[accession] = [[position,result]]  #用列表方式接收

    #print(dictionary)

    '''
    #查看并显示第一个键值对
    for key, value in dictionary.items():
        print(key,value)
        break
    
    #查看前三个键值对
    #print(list(dictionary.items())[:5])
    '''
    #查看字典长度
    print(len(dictionary))
    print('-' * 30)

    # 返回字典
    return (dictionary)


def main():
    result_data = read_data('../result/Result_all.csv')
    #创建结果字典
    result_dict=get_dict(result_data)


    #创建各类型位点数量字典
    number_dict={}
    for key in result_dict:
        n_de=0
        n_un=0
        n_in=0
        for i in range(len(result_dict[key])):
            if result_dict[key][i][1]==-1:
                n_de+=1
            elif result_dict[key][i][1]==1:
                n_in+=1
            else:
                n_un+=1
        number_dict[key]=[n_de,n_un,n_in]

    #查看前三个键值对
    #print(list(number_dict.items())[:5])

    # 查看字典长度
    print(len(number_dict))
    print('-' * 30)

    #输出list：不同类型位点的蛋白
    de_list=[]
    in_list=[]
    for key in number_dict:
        if number_dict[key][0]>0:
            de_list+=[key]
        if number_dict[key][-1]>0:
            in_list+=[key]
    print(len(de_list))
    print(len(in_list))

    #仅上升/仅下降/都包含/有改变
    both_list=list(set(de_list) & set(in_list))
    de_only=list(set(de_list) - set(in_list))
    in_only = list(set(in_list) - set(de_list))
    change_list=list(set(de_list) | set(in_list))

    print(len(both_list))
    print(len(de_only))
    print(len(in_only))
    print(len(change_list))

    #保存结果
    f = open('../enrichment_analysis/both.txt', 'w')
    for item in both_list:
        f.write(item + '\n')
    f.close()

    f = open('../enrichment_analysis/de_only.txt', 'w')
    for item in de_only:
        f.write(item + '\n')
    f.close()

    f = open('../enrichment_analysis/in_only.txt', 'w')
    for item in in_only:
        f.write(item + '\n')
    f.close()

    f = open('../enrichment_analysis/change.txt', 'w')
    for item in change_list:
        f.write(item + '\n')
    f.close()

if __name__ == '__main__':
    main()