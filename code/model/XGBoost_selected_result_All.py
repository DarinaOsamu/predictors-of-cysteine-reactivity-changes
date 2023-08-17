import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve, average_precision_score


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

def Assessment(clf,X,y,predicted):
    # 计算概率
    preds = clf.predict_proba(X)[:, 1]

    # 计算其他指标
    accuracy = metrics.accuracy_score(y, predicted)
    balanced_accuracy = balanced_accuracy_score(y, predicted)
    precision = metrics.precision_score(y, predicted)
    recall = metrics.recall_score(y, predicted)
    f1 = metrics.f1_score(y, predicted)
    MCC = matthews_corrcoef(y, predicted)
    print('ACC=',accuracy,'balanced_ACC=',balanced_accuracy,'precision=', precision, 'recall=',recall, 'f1_score=',f1,'MCC=', MCC)

    # 获取验证集的混淆矩阵
    confusion_matrix = metrics.confusion_matrix(y, predicted)
    # print(confusion_matrix1)
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TP = confusion_matrix[1][1]
    #print('TN:', TN,'FP:', FP,'FN:', FN,'TP:', TP)

    # 灵敏度Sn，特异度Sp
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)
    print('Sn:', Sn)
    print('Sp:', Sp)

    # 绘制混淆矩阵
    metrics.ConfusionMatrixDisplay.from_predictions(y, predicted, cmap='Reds')
    plt.show()

    # 绘制ROC曲线
    metrics.RocCurveDisplay.from_estimator(clf, X, y)
    plt.show()

    #计算AUC
    print("AUC:", metrics.roc_auc_score(y, preds))

    # 绘制PR曲线
    y_scores = clf.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_scores)
    plt.figure("P-R Curve")
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall, precision)
    plt.plot([0, 1], [1, 0], color="black", linestyle="--")
    plt.show()

    # 计算AP
    AP = average_precision_score(y, y_scores, average='macro', pos_label=1, sample_weight=None)
    print('AP:', AP)

def Assessment_3class(y,predicted):
    # 获取验证集的混淆矩阵
    confusion_matrix = metrics.confusion_matrix(y, predicted)
    print(confusion_matrix)
    # 绘制混淆矩阵
    metrics.ConfusionMatrixDisplay.from_predictions(y, predicted, cmap='Reds')
    plt.show()
    # 获取验证集的准确率
    ACC = metrics.accuracy_score(y, predicted)
    print("准确率：", ACC)
    # 获取验证集的平衡准确率
    balanced_score = balanced_accuracy_score(y, predicted)
    print("平衡准确率：", balanced_score)

    # 精确率和召回率
    Pr = metrics.precision_score(y, predicted, average=None)
    Re = metrics.recall_score(y, predicted, average=None)
    print("Precision：", Pr)
    print("Recall：", Re)
    # 获取验证集的MCC
    MCC = matthews_corrcoef(y, predicted)
    print("MCC：", MCC)
    # 获取验证集的f1-score
    f1 = metrics.f1_score(y, predicted, average=None)
    print("f1-score：", f1)

def main():

    # 读取独立测试集
    validation = read_data('../../source_data/validation.csv')
    types = validation['type']
    # print(pd.value_counts(types))
    # print('-' * 30)
    # 读取训练集
    X1 = read_data('../../source_data/train_test_Incre.csv')
    X2 = read_data('../../source_data/train_test_Decre.csv')
    X3 = read_data('../../source_data/train_test_Unchange.csv')

    ##############################
    # 分类：反应性是否改变

    # 划分X，转换y的标签值
    X1_validation = validation.iloc[:, :-1]
    y1_validation = [0 if i == 'Unchanged' else 1 for i in types]
    y1_validation = pd.Series(y1_validation, name='label')

    # 合并训练集
    X1_train_test_list = [X1, X2, X3]
    X1_train_test = pd.concat(X1_train_test_list, ignore_index=True)
    # 查看训练集的样本数和特征数
    # print(X1_train_test.shape)
    # print('-' * 30)

    # 添加y值，区分Changed和Unchanged
    y1_train_test = [1] * (X1.shape[0] + X2.shape[0]) + [0] * (X3.shape[0])
    y1_train_test = pd.Series(y1_train_test, name='label')

    X1_All = pd.concat([X1_train_test, X1_validation], ignore_index=True)
    y1_All = pd.concat([y1_train_test, y1_validation], ignore_index=True)

    # 查看标签分布
    print(X1_All.shape)
    print(pd.value_counts(y1_All))
    print('-' * 30)

    # 读取保存的标化器，标准化独立验证集的X
    scaler1 = pickle.load(open('../../result/standard_scaler/standard_baseline_PosNeg.pkl', 'rb'))
    SS_X1_All = scaler1.transform(X1_All)
    SS_X1_All = pd.DataFrame(SS_X1_All)

    # 读取保存的选择器，筛选特征
    selector1 = pickle.load(open('../../result/feature_selector/EN_Selector_PosNeg.pkl', 'rb'))
    SS_X1_All = selector1.transform(SS_X1_All)

    # 读取保存的分类器，预测独立验证集的y
    clf1 = pickle.load(open('../../result/classifier/XGBoost_selected_PosNeg.pkl', 'rb'))

    # 预测
    predicted1 = clf1.predict(SS_X1_All)
    print('-' * 30)

    predicted1=pd.Series(predicted1)

    # 性能评估
    Assessment(clf1,SS_X1_All,y1_All,predicted1)

    ##############################
    # 分类：反应性上升还是下降
    '''
    # 获取反应性改变的数据
    Changed_validation = validation.loc[validation['type'] != 'Unchanged']
    # 查看样本数和特征数
    print(Changed_validation.shape)
    print('-' * 30)

    Changed_types = Changed_validation['type']

    # 划分X，转换y的标签值
    X2_validation = Changed_validation.iloc[:, :-1]
    y2_validation = [0 if i == 'Decreased' else 1 for i in Changed_types]
    y2_validation = pd.Series(y2_validation, name='label')

    # 合并训练集
    X2_train_test_list = [X1, X2]
    X2_train_test = pd.concat(X2_train_test_list, ignore_index=True)
    # 查看训练集的样本数和特征数
    # print(X2_train_test.shape)
    # print('-' * 30)

    # 添加y值，区分Incre和Decre
    y2_train_test = [1] * (X1.shape[0]) + [0] * (X2.shape[0])
    y2_train_test = pd.Series(y2_train_test, name='label')
    # 查看训练集的长度
    # print(len(y2_train_test))
    # print('-' * 30)

    X2_All = pd.concat([X2_train_test, X2_validation], ignore_index=True)
    y2_All = pd.concat([y2_train_test, y2_validation], ignore_index=True)

    # 查看标签分布
    print(pd.value_counts(y2_All))
    print('-' * 30)

    #读取保存的标化器，标准化独立验证集的X
    #scaler2=pickle.load(open('../../result/standard_scaler/standard_baseline_IncreDecre.pkl','rb'))
    scaler2=pickle.load(open('../../result/standard_scaler/standard_SMOTE_IncreDecre.pkl','rb'))
    SS_X2_All=scaler2.transform(X2_All)
    SS_X2_All=pd.DataFrame(SS_X2_All)

    # 读取保存的选择器，筛选特征
    #selector2 = pickle.load(open('../../result/feature_selector/EN_Selector_baseline_IncreDecre.pkl', 'rb'))
    selector2 = pickle.load(open('../../result/feature_selector/EN_Selector_IncreDecre.pkl', 'rb'))
    SS_X2_All = selector2.transform(SS_X2_All)

    #读取保存的分类器，预测独立验证集的y
    #clf2 = pickle.load(open('../../result/classifier/XGBoost_baseline_selected_IncreDecre.pkl', 'rb'))
    clf2=pickle.load(open('../../result/classifier/XGBoost_selected_IncreDecre.pkl','rb'))

    #预测
    predicted2 = clf2.predict(SS_X2_All)
    #print('预测值：', predicted2[10:20])
    #print('实际值:', y2_validation[10:20])
    print('-' * 30)

    #print(pd.value_counts(predicted2))
    #print('-' * 30)

    #性能评估
    Assessment(clf2,SS_X2_All,y2_All,predicted2)
    '''
    ##############################
    # 完整三分类流程

    # 首先预测反应性是否改变
    predicted1 = pd.Series(predicted1)
    # 查看标签分布
    # print(len(predict1))
    print(pd.value_counts(predicted1))
    print('-' * 30)

    # 划分X，转换y的标签值
    X_validation = validation.iloc[:, :-1]

    # 查看样本数和特征数
    # print(X_validation.shape)
    # print('-' * 30)

    y_validation = [0 if i == 'Unchanged' else (-1 if i == 'Decreased' else 1) for i in types]
    y_validation = pd.Series(y_validation, name='label')

    # 合并训练集
    X_train_test_list = [X1, X2, X3]
    X_train_test = pd.concat(X_train_test_list, ignore_index=True)
    # 查看训练集的样本数和特征数
    # print(X_train_test.shape)
    # print('-' * 30)

    # 添加y值，区分Changed和Unchanged
    y_train_test = [1] * (X1.shape[0]) + [-1] * (X2.shape[0]) + [0] * (X3.shape[0])
    y_train_test = pd.Series(y_train_test, name='label')

    X_All = pd.concat([X_train_test, X_validation], ignore_index=True)
    y_All = pd.concat([y_train_test, y_validation], ignore_index=True)

    # 查看标签分布
    print(pd.value_counts(y_All))
    print('-' * 30)



    # 读取保存的标化器，标准化独立验证集的X
    #scaler2 = pickle.load(open('../../result/standard_scaler/standard_baseline_IncreDecre.pkl', 'rb'))
    scaler2=pickle.load(open('../../result/standard_scaler/standard_SMOTE_IncreDecre.pkl','rb'))
    # 读取保存的选择器
    #selector2 = pickle.load(open('../../result/feature_selector/EN_Selector_baseline_IncreDecre.pkl', 'rb'))
    selector2 = pickle.load(open('../../result/feature_selector/EN_Selector_IncreDecre.pkl', 'rb'))
    # 读取保存的分类器
    #clf2 = pickle.load(open('../../result/classifier/XGBoost_baseline_selected_IncreDecre.pkl', 'rb'))
    clf2 = pickle.load(open('../../result/classifier/XGBoost_selected_IncreDecre.pkl', 'rb'))

    predicted2=[]
    for i in range(len(y_All)):
        predicted_result=[-2]
        if predicted1[i] != 0:
            # 标准化X
            SS_X2_All = scaler2.transform(X_All.iloc[i:i + 1, :])
            SS_X2_All = pd.DataFrame(SS_X2_All)


            # 特征选择
            SS_X2_All = selector2.transform(SS_X2_All)

            # 预测
            predicted_result = clf2.predict(SS_X2_All)
            predicted_result=predicted_result.tolist()

            # print('预测值：', predicted_result)
            # print('实际值:', [y_validation[i]])
            # print('-' * 30)

        predicted2+=predicted_result

    # 查看标签分布
    # print(predicted2)
    # print(len(predicted2))
    print('predicted2')
    print(pd.value_counts(predicted2))
    print('-' * 30)

    #修正标签值
    predicted2=pd.Series(predicted2)
    predicted2=predicted2.replace(0,-1)
    predicted2 = predicted2.replace(-2, 0)

    # 查看标签分布
    # print(predicted2)
    # print(len(predicted2))
    print('predicted2')
    print(pd.value_counts(predicted2))
    print('-' * 30)

    # 性能评估
    Assessment_3class(y_All,predicted2)

    ##############################


if __name__ == '__main__':
    main()