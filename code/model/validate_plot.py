import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc


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

def ROC_plot(clf,X,y,linestyle,label,lw):

    # 绘制ROC曲线
    #metrics.RocCurveDisplay.from_estimator(clf, X, y,linestyle=linestyle)
    #plt.show()

    y_score = clf.predict_proba(X)[:, 1]
    fpr, tpr, threshold = roc_curve(y, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值


    label=label+'(AUC='+str(round(roc_auc,4))+')'
    plt.plot(fpr, tpr, linestyle=linestyle,label=label,lw=lw)  ###假正率为横坐标，真正率为纵坐标做曲线


    #plt.show()

def PR_plot(clf,X,y,linestyle,label,lw):
    # 绘制PR曲线
    y_scores = clf.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_scores)
    AUPR = average_precision_score(y, y_scores, average='macro', pos_label=1, sample_weight=None)

    label = label + '(AUPR=' + str(round(AUPR, 4)) + ')'
    plt.plot(recall, precision,linestyle=linestyle,label=label,lw=lw)

    #plt.show()



def main():
    # 使用训练出的模型在独立验证集上进行预测
    # 读取独立测试集
    validation = read_data('../../source_data/validation.csv')
    types = validation['type']
    print(pd.value_counts(types))
    print('-' * 30)

    ##############################
    # 分类：反应性是否改变

    # 划分X，转换y的标签值
    X1_validation = validation.iloc[:, :-1]
    y1_validation = [0 if i == 'Unchanged' else 1 for i in types]
    # 查看标签分布
    print(pd.value_counts(y1_validation))
    print('-' * 30)

    # 读取保存的标化器，标准化独立验证集的X
    scaler1 = pickle.load(open('../../result/standard_scaler/standard_baseline_PosNeg.pkl', 'rb'))

    SS_X1_validation = scaler1.transform(X1_validation)
    SS_X1_validation = pd.DataFrame(SS_X1_validation)

    # 读取保存的选择器，筛选特征
    selector1 = pickle.load(open('../../result/feature_selector/EN_Selector_PosNeg.pkl', 'rb'))
    SS_X1_validation = selector1.transform(SS_X1_validation)

    # 读取保存的分类器，预测独立验证集的y
    clf1_1= pickle.load(open('../../result/classifier/SVM_selected_PosNeg.pkl', 'rb'))
    clf1_2 = pickle.load(open('../../result/classifier/GaussianNB_selected_PosNeg.pkl', 'rb'))
    clf1_3 = pickle.load(open('../../result/classifier/LogisticRegression_selected_PosNeg.pkl', 'rb'))
    clf1_4 = pickle.load(open('../../result/classifier/RF_selected_PosNeg.pkl', 'rb'))
    clf1_5 = pickle.load(open('../../result/classifier/XGBoost_selected_PosNeg.pkl', 'rb'))

    # 预测
    #predicted1 = clf1.predict(SS_X1_validation)
    #print('-' * 30)


    # 绘制ROC曲线
    plt.figure(figsize=(5,5))
    ROC_plot(clf1_1,SS_X1_validation,y1_validation,linestyle='solid',label='SVM',lw=1)
    ROC_plot(clf1_2, SS_X1_validation, y1_validation, linestyle='dashdot',label='NB',lw=1)
    ROC_plot(clf1_3, SS_X1_validation, y1_validation, linestyle='dotted',label='LR',lw=1)
    ROC_plot(clf1_4, SS_X1_validation, y1_validation, linestyle='solid', label='RF',lw=2)
    ROC_plot(clf1_5, SS_X1_validation, y1_validation, linestyle='dotted', label='XGB',lw=2)

    plt.plot([0, 1], [0, 1], "black", linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    # 绘制PR曲线
    plt.figure(figsize=(5, 5))
    PR_plot(clf1_1, SS_X1_validation, y1_validation, linestyle='solid', label='SVM', lw=1)
    PR_plot(clf1_2, SS_X1_validation, y1_validation, linestyle='dashdot', label='NB', lw=1)
    PR_plot(clf1_3, SS_X1_validation, y1_validation, linestyle='dotted', label='LR', lw=1)
    PR_plot(clf1_4, SS_X1_validation, y1_validation, linestyle='solid', label='RF', lw=2)
    PR_plot(clf1_5, SS_X1_validation, y1_validation, linestyle='dotted', label='XGB', lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot([0, 1], [1, 0], color="black", linestyle="--")
    plt.legend(loc='lower left')
    plt.show()

    ##############################
    # 分类：反应性上升还是下降
    '''
    #获取反应性改变的数据
    Changed_validation=validation.loc[validation['type']!='Unchanged']
    #查看样本数和特征数
    print(Changed_validation.shape)
    print('-' * 30)

    Changed_types=Changed_validation['type']

    #划分X，转换y的标签值
    X2_validation=Changed_validation.iloc[:,:-1]
    y2_validation = [-1 if i=='Decreased' else 1 for i in Changed_types]
    #查看标签分布
    print(pd.value_counts(y2_validation))
    print('-' * 30)

    #读取保存的标化器，标准化独立验证集的X
    #scaler2=pickle.load(open('../../result/standard_scaler/standard_baseline_IncreDecre.pkl','rb'))
    scaler2=pickle.load(open('../../result/standard_scaler/standard_SMOTE_IncreDecre.pkl','rb'))
    SS_X2_validation=scaler2.transform(X2_validation)
    SS_X2_validation=pd.DataFrame(SS_X2_validation)

    # 读取保存的选择器，筛选特征
    selector2 = pickle.load(open('../../result/feature_selector/EN_Selector_IncreDecre.pkl', 'rb'))
    SS_X2_validation = selector2.transform(SS_X2_validation)

    #读取保存的分类器，预测独立验证集的y
    # 读取保存的分类器，预测独立验证集的y
    clf2_1 = pickle.load(open('../../result/classifier/SVM_selected_IncreDecre.pkl', 'rb'))
    clf2_2 = pickle.load(open('../../result/classifier/GaussianNB_selected_IncreDecre.pkl', 'rb'))
    clf2_3 = pickle.load(open('../../result/classifier/LogisticRegression_selected_IncreDecre.pkl', 'rb'))
    clf2_4 = pickle.load(open('../../result/classifier/RF_selected_IncreDecre.pkl', 'rb'))
    clf2_5 = pickle.load(open('../../result/classifier/XGBoost_selected_IncreDecre.pkl', 'rb'))

    #预测
    #predicted2 = clf2.predict(SS_X2_validation)
    #print('预测值：', predicted2[10:20])
    #print('实际值:', y2_validation[10:20])
    print('-' * 30)

    # 绘制ROC曲线
    plt.figure(figsize=(5, 5))
    ROC_plot(clf2_1, SS_X2_validation, y2_validation, linestyle='solid', label='SVM', lw=1)
    ROC_plot(clf2_2, SS_X2_validation, y2_validation, linestyle='dashdot', label='NB', lw=1)
    ROC_plot(clf2_3, SS_X2_validation, y2_validation, linestyle='dotted', label='LR', lw=1)
    ROC_plot(clf2_4, SS_X2_validation, y2_validation, linestyle='solid', label='RF', lw=2)
    ROC_plot(clf2_5, SS_X2_validation, y2_validation, linestyle='dotted', label='XGB', lw=2)
    plt.plot([0, 1], [0, 1], "black", linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    plt.show()

    # 绘制PR曲线
    plt.figure(figsize=(5, 5))
    PR_plot(clf2_1, SS_X2_validation, y2_validation, linestyle='solid', label='SVM', lw=1)
    PR_plot(clf2_2, SS_X2_validation, y2_validation, linestyle='dashdot', label='NB', lw=1)
    PR_plot(clf2_3, SS_X2_validation, y2_validation, linestyle='dotted', label='LR', lw=1)
    PR_plot(clf2_4, SS_X2_validation, y2_validation, linestyle='solid', label='RF', lw=2)
    PR_plot(clf2_5, SS_X2_validation, y2_validation, linestyle='dotted', label='XGB', lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot([0, 1], [1, 0], color="black", linestyle="--")
    plt.legend(loc='lower left')
    plt.show()
    '''
    ##############################



if __name__ == '__main__':
    main()