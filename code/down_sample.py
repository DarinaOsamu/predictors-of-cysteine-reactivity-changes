# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


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
    return (dataset)

def main():

    # 加载位点数据
    Neg_site = read_data('../source_data/CD_Neg_Site_All.csv')
    Incre_site = read_data('../source_data/CD_Incre_Site.csv')
    Decre_site = read_data('../source_data/CD_Decre_Site.csv')

    Neg_N=Incre_site.shape[0]+Decre_site.shape[0]
    print(Neg_N)

    Neg_Index = Neg_site.index
    np.random.seed(0)

    Neg_Index_choice = np.random.choice(Neg_Index, size=Neg_N, replace=False)
    print(Neg_Index_choice)
    Neg_choice = Neg_site.iloc[Neg_Index_choice, :]

    print(Neg_choice.iloc[0:3,:])
    print(Neg_choice.shape)

    Neg_choice.to_excel(excel_writer=r'../result/dataset/CD_Neg_Site.xlsx', index=False)

if __name__ == '__main__':
    main()