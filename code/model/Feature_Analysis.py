# -*- coding: utf-8 -*-

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import heapq
import seaborn as sns


def read_data(file_path):
    dataset = pd.read_csv(file_path)
    '''
    #查看并显示前三条记录
    print(dataset.iloc[0:3,:])
    print('-' * 30)
    
    # 查看样本数和特征数
    print(dataset.shape)
    print('-' * 30)
    '''
    return (dataset)

def count_feature_selected(selector,K):

    select_list=selector.get_support()
    feature_number=[K,441,441,5*K,15,21*K,3,1]
    feature_pos_list=[]
    feature_start_pos=0
    for i in range(8):
        feature_start_pos+=feature_number[i]
        feature_pos_list+=[feature_start_pos]
    #print(feature_pos_list)

    N0 = pd.value_counts(select_list[0:feature_pos_list[0]])
    selected_number=[N0[True]]
    for i in range(7):
        N=pd.value_counts(select_list[feature_pos_list[i]:feature_pos_list[i+1]])
        if True in N.index:
            selected_number += [N[True]]
        else:
            selected_number += [0]
    #print(selected_number)
    count_feature_plot(selected_number)
    return(selected_number)

def count_feature_plot(selected_number):

    features_name = ['PSSM', 'CKSAAP0', 'CKSAAP1', 'Archley', 'EBGW', 'Blosum62', 'STPdis', 'IUPRED']
    plt.tick_params(labelsize=8)
    rect=plt.bar(features_name,selected_number)
    plt.bar_label(rect, label_type='edge')
    plt.show()

def read_coef(enet,feature_number):
    coef = enet.coef_
    coef=list(coef)
    coef_abs=list(map(abs,coef))
    index = map(coef_abs.index, heapq.nlargest(10, coef_abs))
    index=list(index)

    coef=np.array(coef)
    value = coef[index]
    print(index)
    print(value)
    #print(heapq.nlargest(10, coef_abs))
    return(index)

def trans_index(index_list,K):
    feature_number = [K, 441, 441, 5 * K, 15, 21 * K, 3, 1]
    STP_list=['S-P_dis','T-P_dis','ST-P_dis']
    aa_seq = 'ARNDCQEGHILKMFPSTWVYX'

    # 获取氨基酸对
    Dipeptide = []
    for i in aa_seq:
        for j in aa_seq:
            Dipeptide += [i + j]

    trans_list=[]
    for index in index_list:
        if index==sum(feature_number)-1:
            trans_list+=['IUPRED']
        elif index>sum(feature_number)-5:
            trans_list+=[STP_list[index-1464]]
        elif K<=index<K+441:
            trans_list += [Dipeptide[index-K+1]]
        elif K+441<=index<K+882:
            trans_list += [Dipeptide[index - K -441 + 1][0]+'_'+Dipeptide[index - K -441 + 1][1]]
        else:
            trans_list += ['']
    return(trans_list)

def plot_box(index_list,trans_list,content):
    # 读取训练集
    X1 = read_data('../../source_data/train_test_Incre.csv')
    X2 = read_data('../../source_data/train_test_Decre.csv')
    X3 = read_data('../../source_data/train_test_Unchange.csv')

    if content == 1:
        X = pd.concat([X1, X2, X3], ignore_index=True)
        X_max = X.iloc[:, index_list]
        X_max.columns = trans_list
        X_max['label']=['Change']*(X1.shape[0]+X2.shape[0])+['Unchange']*(X3.shape[0])
        #X_max['label'] = ['Increase'] * (X1.shape[0]) + ['Decrease'] * (X2.shape[0]) + ['Unchange'] * (X3.shape[0])
    elif content == 2:
        X = pd.concat([X1, X2], ignore_index=True)
        X_max = X.iloc[:, index_list]
        X_max.columns = trans_list
        X_max['label'] = ['Increase'] * (X1.shape[0]) + ['Decrease'] * (X2.shape[0])


    # 绘图
    num=len(index_list)
    #sns.set(color_codes=True)
    for i in range(num):
        if index_list[i]>=1464:
            sns.boxplot(x='label',y=trans_list[i],data=X_max)
        else:
            plot_with_hue(data=X_max,hue='label',feature=trans_list[i])
            #sns.countplot(x=trans_list[i], hue='label',data=X_max)
        '''
        #保存图片
        if content == 1:
            plt.savefig('../../result/figure/max_feature/PosNeg_'+trans_list[i])
        elif content == 2:
            plt.savefig('../../result/figure/max_feature/IncreDecre_'+trans_list[i])
        '''
        plt.show()


def plot_with_hue(data,hue,feature):
    #按百分比绘制条形图
    df=data
    x,y=hue,feature
    (df
     .groupby(x)[y]
     .value_counts(normalize=True)
     .mul(100)
     .rename('percent')
     .reset_index()
     .pipe((sns.catplot, 'data'), x=x, y='percent', hue=y, kind='bar'))

def plot_difference(index_list, trans_list, content):
    # 读取训练集
    X1 = read_data('../../source_data/train_test_Incre.csv')
    X2 = read_data('../../source_data/train_test_Decre.csv')
    X3 = read_data('../../source_data/train_test_Unchange.csv')

    if content == 1:
        X_Pos = pd.concat([X1, X2], ignore_index=True)
        X_Neg = X3
        X_Pos_max = X_Pos.iloc[:, index_list]
        X_Neg_max = X_Neg.iloc[:, index_list]
        #X_Pos_max.columns = trans_list
        #X_Neg_max.columns = trans_list
        plt.title('average difference (Change - Unchange)')

    elif content == 2:
        X_Pos = X1
        X_Neg = X2
        X_Pos_max = X_Pos.iloc[:, index_list]
        X_Neg_max = X_Neg.iloc[:, index_list]
        plt.title('average difference (Increase - Decrease)')

    num = len(index_list)
    difference_list=[]
    trans_list_CKSAAP=[]
    for i in range(num):
        if index_list[i] < 1464:
            average_Pos = X_Pos_max.iloc[:,i].mean()
            average_Neg = X_Neg_max.iloc[:,i].mean()
            average_difference=average_Pos-average_Neg
            difference_list+=[average_difference]
            trans_list_CKSAAP+=[trans_list[i]]
    #print(difference_list)
    plt.tick_params(labelsize=8)
    rect=plt.bar(trans_list_CKSAAP,difference_list,color='None',edgecolor='black')
    plt.bar_label(rect, label_type='edge',fmt='%.4f',padding=5,size=8)
    ax=plt.gca()
    # 设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    #设置x轴的位置
    ax.spines['bottom'].set_position(('data', 0))
    #设置x轴标签位置
    plt.show()

def main():

    #单侧长度L，窗口长度K
    L=10
    K=2*L+1

    #读取特征选择器
    selector1 = pickle.load(open('../../result/feature_selector/EN_Selector_PosNeg.pkl', 'rb'))
    selector2 = pickle.load(open('../../result/feature_selector/EN_Selector_IncreDecre.pkl', 'rb'))

    #查看各个特征编码方式中被选中的特征数量,并画出条形图
    #selected_number1=count_feature_selected(selector1,K)
    #selected_number2 = count_feature_selected(selector2, K)


    #读取弹性网络
    enet1 = pickle.load(open('../../result/feature_selector/EN_baseline_PosNeg.pkl', 'rb'))
    enet2 = pickle.load(open('../../result/feature_selector/EN_SMOTE_IncreDecre.pkl', 'rb'))
    #查看系数绝对值最大的十个特征
    index_list1=read_coef(enet1, feature_number=10)
    index_list2 = read_coef(enet2, feature_number=10)

    #查看特征的意义
    trans_list1=trans_index(index_list1, K)
    trans_list2=trans_index(index_list2, K)

    #绘制训练集上这些特征的分布图
    #plot_box(index_list1, trans_list1, 1)
    #plot_box(index_list2, trans_list2, 2)

    #取特征在正负样本上的均值作差，体现样本的差异性
    plot_difference(index_list1, trans_list1, 1)
    plot_difference(index_list2, trans_list2, 2)



if __name__ == '__main__':
    main()