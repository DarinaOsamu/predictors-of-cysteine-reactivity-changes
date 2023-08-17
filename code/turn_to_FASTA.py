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
    dataset=read_data('../source_data/CD_All_Seq.csv')
    with open("../source_data/seq_21.txt", "w") as f:
        for i in range(dataset.shape[0]):
            ex='>'+dataset.iloc[i,0]+'|'+str(dataset.iloc[i,1])+'|'+dataset.iloc[i,3]
            sq=dataset.iloc[i,2]
            #print(ex,sq)

            f.write(ex)
            f.write('\n')
            f.write(sq)
            f.write('\n')

if __name__ == '__main__':
    main()