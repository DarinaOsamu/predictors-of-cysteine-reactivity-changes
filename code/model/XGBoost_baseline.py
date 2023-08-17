
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from xgboost import XGBClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import xgboost as xgb
# 网格搜索
from sklearn.model_selection import GridSearchCV


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
    return(scaler)

def XGB_classifier(X_train, y_train):
    # 标签分布统计
    Label_D = pd.value_counts(y_train)
    print(Label_D)
    print('-' * 30)

    ##############################
    # 调用分类器
    clf = XGBClassifier(random_state=0,n_estimators=500)
    #clf.fit(X_train, y_train)
    #y_predprob = clf.predict_proba(X_train)[:, 1]
    #print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob))
    #return(clf)

    #找出最佳迭代次数
    xgb_param = clf.get_xgb_params()
    xgtrain = xgb.DMatrix(X_train.values, label=y_train.values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=10,
                      metrics='auc', early_stopping_rounds=50)
    clf.set_params(n_estimators=cvresult.shape[0])
    print(clf.get_params()['n_estimators'])


    #clf.set_params(n_estimators=10)
    #clf.set_params(n_estimators=66)

    clf.fit(X_train, y_train)
    y_predprob = clf.predict_proba(X_train)[:, 1]
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob))
    return (clf)
    '''
    # 网格搜索
    # 要实验的超参数:'max_depth','min_child_weight'
    #param_grid = {'max_depth':range(3,10,2),'min_child_weight':range(1,6,2)}
    #param_grid = {'max_depth': range(1, 5), 'min_child_weight': range(4,7)}
    param_grid = {'max_depth': range(4, 7), 'min_child_weight': range(1, 3)}
    # 创建一个网格搜索:10折交叉验证
    grid_search = GridSearchCV(clf, param_grid, cv=10, scoring='roc_auc')

    # 开始搜索
    grid_search.fit(X_train, y_train)
    grid_search.score(X_train, y_train)  # 输出精度

    print(grid_search.best_params_, grid_search.best_score_)  # 最优参数与最优精度

    print(grid_search.best_estimator_)  # 访问最佳参数对应的模型，它是在整个训练集上训练得到的
    results = pd.DataFrame(grid_search.cv_results_)  # 网格搜索的结果
    print(results)
    print(results.loc[:, ['mean_test_score', 'std_test_score','rank_test_score']])

    ##############################
    # 获取最优模型
    optimal_XGB = grid_search.best_estimator_
    y_predprob = optimal_XGB.predict_proba(X_train)[:, 1]
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob))
    return(optimal_RF)
    '''

def plot_learning_curve(estimator,title, X, y):
    #绘制学习曲线
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=10,random_state=0)
    plt.figure("learning_curve")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Test score")
    plt.legend(loc="best")
    plt.show()




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
    optimal_XGB1=XGB_classifier(SS_X1_train_test, y1_train_test)

    #绘制学习曲线
    #plot_learning_curve(optimal_XGB1,"XGB-Change/Unchange", SS_X1_train_test,  y1_train_test)

    # 保存最优模型
    #pickle.dump(optimal_XGB1, open('../../result/classifier/XGBoost_baseline_PosNeg.pkl','wb'))

    ##############################

    # 预测反应性升高还是降低
    '''
    #合并训练集
    X2_train_test_list=[X1,X2]
    X2_train_test=pd.concat(X2_train_test_list,ignore_index=True)
    #查看训练集的样本数和特征数
    print(X2_train_test.shape)
    print('-' * 30)

    #添加y值，区分Incre和Decre
    #特殊：XGBoost负标签值必须为0
    y2_train_test=[1]*(X1.shape[0])+[0]*(X2.shape[0])
    y2_train_test=pd.Series(y2_train_test,name='label')
    #查看训练集的长度
    print(len(y2_train_test))
    print('-' * 30)

    # 无标化器时
    #scaler2=standard_scaler(X2_train_test)
    #保存标化器
    #pickle.dump(scaler2, open('../../result/standard_scaler/standard_baseline_IncreDecre.pkl','wb'))

    # 可调用标化器时，读取保存的标化器
    scaler2 = pickle.load(open('../../result/standard_scaler/standard_baseline_IncreDecre.pkl', 'rb'))
    SS_X2_train_test = scaler2.transform(X2_train_test)
    SS_X2_train_test = pd.DataFrame(SS_X2_train_test)

    #训练模型
    optimal_XGB2=XGB_classifier(SS_X2_train_test,y2_train_test)
    
    #绘制学习曲线
    plot_learning_curve(optimal_XGB2,"XGB-Increase/Decrease", SS_X2_train_test,  y2_train_test)

    # 保存最优模型
    pickle.dump(optimal_XGB2, open('../../result/classifier/XGBoost_baseline_IncreDecre.pkl','wb'))
    '''
    ##############################

if __name__ == '__main__':
    main()