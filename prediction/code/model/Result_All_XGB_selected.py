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

def main():
    # 读取特征编码后的数据
    PSSM_code = read_data('../../source_data/feature/PSSM_code.csv')
    CKSAAP_0_code = read_data('../../source_data/feature/CKSAAP_0_code.csv')
    CKSAAP_1_code = read_data('../../source_data/feature/CKSAAP_1_code.csv')
    Atchley_code = read_data('../../source_data/feature/Atchley_code.csv')
    EBGW_code = read_data('../../source_data/feature/EBGW_code.csv')
    BLOSUM62_code = read_data('../../source_data/feature/BLOSUM62_code.csv')
    STP_dis_code = read_data('../../source_data/feature/STP_dis_code.csv')
    In_Disorder_code = read_data('../../source_data/feature/In_Disorder_code.csv')

    Feature_list = [PSSM_code, CKSAAP_0_code, CKSAAP_1_code, Atchley_code, EBGW_code, BLOSUM62_code, STP_dis_code,
                    In_Disorder_code]
    data_index = In_Disorder_code.loc[:, ['accession', 'position']]

    # 记录各特征维度数
    dimen = []
    n = 0
    for i in Feature_list:
        # print(i.shape[1]-2)
        Feature_list[n] = Feature_list[n].iloc[:, 2:]
        dimen += [i.shape[1] - 2]
        n += 1
    print(dimen)
    # 总特征数
    dimen_sum = sum(dimen)
    print(dimen_sum)


    # 特征合并
    dataset = pd.concat(Feature_list, axis=1)
    dataset = pd.concat([data_index, dataset], axis=1)
    validation=dataset.iloc[:,2:]
    print(validation.iloc[0:3,:])

    ##############################
    # 分类：反应性是否改变

    # 划分X
    X1_validation = validation

    # 读取保存的标化器，标准化独立验证集的X
    scaler1 = pickle.load(open('../../result/standard_scaler/standard_baseline_PosNeg.pkl', 'rb'))
    SS_X1_validation = scaler1.transform(X1_validation)
    SS_X1_validation = pd.DataFrame(SS_X1_validation)

    # 读取保存的选择器，筛选特征
    selector1 = pickle.load(open('../../result/feature_selector/EN_Selector_PosNeg.pkl', 'rb'))
    SS_X1_validation = selector1.transform(SS_X1_validation)

    # 读取保存的分类器，预测独立验证集的y
    clf1 = pickle.load(open('../../result/classifier/XGBoost_selected_PosNeg.pkl', 'rb'))

    # 预测
    predicted1 = clf1.predict(SS_X1_validation)
    print('-' * 30)

    predicted1 = pd.Series(predicted1)

    # 查看标签分布
    # print(len(predict1))
    print(pd.value_counts(predicted1))
    print('-' * 30)

    # 划分X
    X_validation = validation

    # 查看样本数和特征数
    # print(X_validation.shape)
    # print('-' * 30)

    # 读取保存的标化器，标准化独立验证集的X
    scaler2=pickle.load(open('../../result/standard_scaler/standard_SMOTE_IncreDecre.pkl','rb'))
    # 读取保存的选择器
    selector2 = pickle.load(open('../../result/feature_selector/EN_Selector_IncreDecre.pkl', 'rb'))
    # 读取保存的分类器
    clf2 = pickle.load(open('../../result/classifier/XGBoost_selected_IncreDecre.pkl', 'rb'))

    predicted2 = []
    for i in range(X_validation.shape[0]):
        predicted_result = [-2]
        if predicted1[i] != 0:
            # 标准化X
            SS_X2_validation = scaler2.transform(X_validation.iloc[i:i + 1, :])
            SS_X2_validation = pd.DataFrame(SS_X2_validation)

            # print(SS_X2_validation)
            # print(type(SS_X2_validation))
            # print('-' * 30)

            # 特征选择
            SS_X2_validation = selector2.transform(SS_X2_validation)

            # 预测
            predicted_result = clf2.predict(SS_X2_validation)
            predicted_result = predicted_result.tolist()

            # print('预测值：', predicted_result)
            # print('实际值:', [y_validation[i]])
            # print('-' * 30)

        predicted2 += predicted_result

    # 查看标签分布
    # print(predicted2)
    # print(len(predicted2))
    print('predicted2')
    print(pd.value_counts(predicted2))
    print('-' * 30)

    # 修正标签值
    predicted2 = pd.Series(predicted2)
    predicted2 = predicted2.replace(0, -1)
    predicted2 = predicted2.replace(-2, 0)

    # 查看标签分布
    # print(predicted2)
    # print(len(predicted2))
    print('predicted2')
    print(pd.value_counts(predicted2))
    print('-' * 30)

    #保存预测结果
    Result_all=pd.concat([data_index, predicted2], axis=1)
    print(Result_all.iloc[0:3,:])
    Result_all.to_csv('../../result/Result_all.csv', index=False)
    ##############################


if __name__ == '__main__':
    main()
