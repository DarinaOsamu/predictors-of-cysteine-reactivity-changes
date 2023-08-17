# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 08:18:42 2022

@author: hp
"""

import pandas as pd
from imblearn.combine import SMOTETomek
import pickle
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score


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


def ElasticNet(X_train, y_train):
    enet = linear_model.ElasticNet(max_iter=10000)
    # 网格搜索
    # 要实验的超参数:4个alphas,4个l1_ratio
    # alphas = np.logspace(-3, 0, 10)
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1], 'l1_ratio': [0.1, 0.3, 0.5, 0.7]}
    # 创建一个网格搜索:16组参数组合，10折交叉验证
    grid_search = GridSearchCV(enet, param_grid, cv=10)

    # 开始搜索
    grid_search.fit(X_train, y_train)
    grid_search.score(X_train, y_train)  # 输出精度

    print(grid_search.best_params_, grid_search.best_score_)  # 最优参数与最优精度

    print(grid_search.best_estimator_)  # 访问最佳参数对应的模型，它是在整个训练集上训练得到的
    results = pd.DataFrame(grid_search.cv_results_)  # 网格搜索的结果
    print(results)

    ##############################

    # 获取最优模型
    optimal_enet = grid_search.best_estimator_
    print('score:', optimal_enet.score(X_train, y_train))

    ##############################

    # Estimate the coef_ on full data with optimal regularization parameter
    coef_ = optimal_enet.coef_
    # Plot coef
    plt.plot(coef_)
    plt.show()

    return (optimal_enet)


def Feature_Selector(optimal_enet, X, y):
    # 查找最佳阈值
    coef_ = optimal_enet.coef_

    # threshold = np.linspace(0,(abs(coef_)).max(),20)
    # threshold = np.linspace(0, 0.01, 20)
    threshold = np.linspace(0, 0.02, 40)
    # linspace(start,end,取出的数量)
    score = []
    for i in threshold:
        X_embedded = SelectFromModel(optimal_enet, threshold=i).fit_transform(X, y)
        once = cross_val_score(optimal_enet, X_embedded, y, cv=10).mean()  # 交叉验证
        score.append(once)
    i_threshold_optim = np.argmax(score)
    threshold_optim = threshold[i_threshold_optim]
    print("Optimal threshold : %s" % threshold_optim)
    plt.plot(threshold, score)
    plt.vlines(threshold_optim, plt.ylim()[0], np.max(score), color='k', linewidth=3)
    plt.show()

    # 特征选择
    selector = SelectFromModel(optimal_enet, threshold=threshold_optim).fit(X, y)

    # print(selector.get_support())
    X_embedded = selector.transform(X)
    # print(X.shape,X_embedded.shape)

    return (selector)


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
    # print(X1_train_test.shape)
    # print('-' * 30)

    # 添加y值，区分Changed和Unchanged
    y1_train_test = [1] * (X1.shape[0]) +[-1]*( X2.shape[0]) + [0] * (X3.shape[0])
    y1_train_test = pd.Series(y1_train_test, name='label')
    # 查看训练集的长度
    # print(len(y1_train_test))
    # print('-' * 30)

    # SMOTETomek 综合抽样
    print('不经过任何采样处理的原始 y1中的分类情况：{}'.format(pd.value_counts(y1_train_test)))
    Sampler = SMOTETomek(random_state=0)  # 综合采样
    X1_ST, y1_ST = Sampler.fit_resample(X1_train_test, y1_train_test)
    print('综合采样后，训练集 ST_y1中的分类情况：{}'.format(pd.value_counts(y1_ST)))

    # 读取保存的标化器
    scaler1 = pickle.load(open('../../result/standard_scaler/standard_SMOTE_direct3class.pkl', 'rb'))
    SS_X1_ST = scaler1.fit_transform(X1_ST)
    SS_X1_ST = pd.DataFrame(SS_X1_ST)

    # 训练弹性网络
    optimal_enet1=ElasticNet(SS_X1_ST,y1_ST)
    #pickle.dump(optimal_enet1, open('../../result/feature_selector/EN_SMOTE_direct3class.pkl','wb'))
    '''
    # 读取得到的最优弹性网络
    optimal_enet1 = pickle.load(open('../../result/feature_selector/EN_SMOTE_direct3class.pkl', 'rb'))
    coef_1 = optimal_enet1.coef_
    # Plot coef
    plt.plot(coef_1)
    plt.show()

    ##############################
    # 特征选择
    
    # 生成选择器
    #selector1 = Feature_Selector(optimal_enet1, SS_X1_ST, y1_ST)
    # 保存选择器
    #pickle.dump(selector1, open('../../result/feature_selector/EN_Selector_direct3class.pkl','wb'))

    # 读取得到的选择器
    selector1 = pickle.load(open('../../result/feature_selector/EN_Selector_direct3class.pkl', 'rb'))
    print(selector1.get_support())
    X1_embedded = selector1.transform(SS_X1_ST)
    print(SS_X1_ST.shape, X1_embedded.shape)
    '''
    ##############################

    '''
    feature1=pd.DataFrame(selector1.get_support())
    feature2=pd.DataFrame(selector2.get_support())
    feature_selected=pd.concat([feature1,feature2],axis=1)
    feature_selected.to_excel(excel_writer=r'../../result/feature/feature_selected.xlsx',index=False)
    '''


if __name__ == '__main__':
    main()