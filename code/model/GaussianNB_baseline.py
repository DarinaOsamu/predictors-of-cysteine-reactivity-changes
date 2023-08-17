# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


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


def standard_scaler(data):
    # 对训练集进行标准化
    scaler = StandardScaler()
    scaler.fit_transform(data)
    return (scaler)


def GaussianNB_classifier(X_train, y_train):
    # 标签分布统计
    Label_D = pd.value_counts(y_train)
    print(Label_D)
    print('-' * 30)

    ##############################
    # 调用分类器
    clf = GaussianNB()

    clf.fit(X_train, y_train)
    y_predprob = clf.predict_proba(X_train)[:, 1]
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob))
    return (clf)


def main():
    # 读取训练集
    X1 = read_data('../../source_data/train_test_Incre.csv')
    X2 = read_data('../../source_data/train_test_Decre.csv')
    X3 = read_data('../../source_data/train_test_Unchange.csv')

    ##############################
    # 预测反应性是否改变

    # 合并训练集
    X1_train_test_list = [X1, X2, X3]
    X1_train_test = pd.concat(X1_train_test_list, ignore_index=True)
    # 查看训练集的样本数和特征数
    print(X1_train_test.shape)
    print('-' * 30)

    # 添加y值，区分Changed和Unchanged
    y1_train_test = [1] * (X1.shape[0] + X2.shape[0]) + [0] * (X3.shape[0])
    y1_train_test = pd.Series(y1_train_test, name='label')
    # 查看训练集的长度
    print(len(y1_train_test))
    print('-' * 30)

    # 无标化器时
    #scaler1=standard_scaler(X1_train_test)
    # 保存标化器
    #pickle.dump(scaler1, open('../../result/standard_scaler/standard_baseline_PosNeg.pkl','wb'))

    # 可调用标化器时，读取保存的标化器
    scaler1 = pickle.load(open('../../result/standard_scaler/standard_baseline_PosNeg.pkl', 'rb'))
    SS_X1_train_test = scaler1.transform(X1_train_test)
    SS_X1_train_test = pd.DataFrame(SS_X1_train_test)

    # 训练模型
    optimal_GaussianNB1=GaussianNB_classifier(SS_X1_train_test, y1_train_test)

    # 保存最优模型
    pickle.dump(optimal_GaussianNB1, open('../../result/classifier/GaussianNB_baseline_PosNeg.pkl','wb'))

    ##############################
    # 预测反应性升高还是降低

    # 合并训练集
    X2_train_test_list = [X1, X2]
    X2_train_test = pd.concat(X2_train_test_list, ignore_index=True)
    # 查看训练集的样本数和特征数
    print(X2_train_test.shape)
    print('-' * 30)

    # 添加y值，区分Incre和Decre
    y2_train_test = [1] * (X1.shape[0]) + [-1] * (X2.shape[0])
    y2_train_test = pd.Series(y2_train_test, name='label')
    # 查看训练集的长度
    print(len(y2_train_test))
    print('-' * 30)

    # 无标化器时
    #scaler2 = standard_scaler(X2_train_test)
    # 保存标化器
    #pickle.dump(scaler2, open('../../result/standard_scaler/standard_baseline_IncreDecre.pkl', 'wb'))

    # 可调用标化器时，读取保存的标化器
    scaler2 = pickle.load(open('../../result/standard_scaler/standard_baseline_IncreDecre.pkl', 'rb'))
    SS_X2_train_test = scaler2.transform(X2_train_test)
    SS_X2_train_test = pd.DataFrame(SS_X2_train_test)

    # 训练模型
    optimal_GaussianNB2 = GaussianNB_classifier(SS_X2_train_test, y2_train_test)

    # 保存最优模型
    pickle.dump(optimal_GaussianNB2, open('../../result/classifier/GaussianNB_baseline_IncreDecre.pkl', 'wb'))

    ##############################


if __name__ == '__main__':
    main()