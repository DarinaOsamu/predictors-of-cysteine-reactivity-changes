# -*- coding: utf-8 -*-

import pandas as pd


def read_data(file_path):
    dataset = pd.read_csv(file_path)

    #查看并显示前三条记录
    print(dataset.iloc[0:3,:])
    print('-' * 30)

    #查看样本数和特征数
    print(dataset.shape)
    print('-' * 30)

    return (dataset)

def main():
    PSSM_code = read_data('../source_data/Result_all.csv')
    print(PSSM_code['accession'].nunique())
    print(PSSM_code.groupby('type')['accession'].nunique())
    print(PSSM_code.groupby('0')['accession'].nunique())


if __name__ == '__main__':

    main()