import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from random import normalvariate


def data_preprocessing(raw_data):
    '''
    传入pandas的dataframe,对dataframe进行切分，切分成feature cols和label set
    :param raw_data:
    :return: feature 特征矩阵 label label数组
    '''
    regularization = MinMaxScaler()

    feature = raw_data.iloc[:, :-1]
    feature = regularization.fit_transform(feature)
    feature = np.mat(feature)
    label = np.array(raw_data.iloc[:, -1])
    return feature, label


def sigmoid(x):
    '''
    定义sigmoid函数 1/(1+e^-x),常用的activation function。
    :param x: float type
    :return:
    '''
    return 1.0 / (1.0 + np.exp(-x))

def sgd_fm(featureMatrix, label, k, iter, learning_rate):
    '''
    对输出公式的权值进行更新
    :param featureMatrix: 特征矩阵
    :param label: 训练样本label
    :param k: 超参，定义v的维度。
    :param iter:迭代次数
    :param learning_rate:学习率
    :return:
    '''
    # m:数据集特征的行数，n:数据集特征的列数
    m, n = np.shape(featureMatrix)
    # 初始化W0,W,V
    # 偏置
    w0 = 0.0
    w = np.zeros((n, 1))
    v = normalvariate(0, 0.2) * np.ones((n, k))

    for it in range(iter):
        for i in range(m):
            # (Vi,f * Xi)^2
            VifMulXi = featureMatrix[i] * v
            # (Vi,f)^2*Xi^2
            TwoPowVifXi = np.multiply(featureMatrix[i], featureMatrix[i]) * np.multiply(v, v)
            # 交叉积，就是1/2开头的那个公式
            CrossTerm = np.sum((np.multiply(VifMulXi, VifMulXi) - TwoPowVifXi), axis=1) * 0.5
            # sigmoid 激活函数
            predict = sigmoid((w0 + w * featureMatrix[i] + CrossTerm)[0, 0])
            # 以下是权值更新，根据SGD，随机梯度下降。对w0,Wi,Vi,f求偏导
            w0 = w0 - learning_rate * predict
            for j in range(n):
                w[j] = w[j] - learning_rate * predict * featureMatrix[i, j]
                for f in range(k):
                    v[j, f] = v[j, f] - learning_rate * (predict * (
                                featureMatrix[i, j] * VifMulXi[0, f] - v[j, f] * featureMatrix[i, j] * featureMatrix[
                            i, j]))
    return w0, w, v


def predict(w0, w, v, featureCols, thold):
    '''
    预测函数，输入之前训练好的W0,W,V。
    :param w0: W0
    :param w: 一次权值
    :param v: 二次权值，交叉项
    :param featureCols:
    :param thold:
    :return:
    '''
    # 这里还原最后一次的输出公式。代入参数即可
    VifMulXi = featureCols * v
    TwoPowVifXi = np.multiply(featureCols, featureCols) * np.multiply(v, v)
    CrossTerm = np.sum((np.multiply(VifMulXi, VifMulXi) - TwoPowVifXi), axis=1) * 0.5
    predict = sigmoid((w0 + w * featureCols + CrossTerm)[0, 0])
    print("predict",predict)
    if predict > thold:
        return 1
    else:
        return 0



def train(data_train,  k, iter, learning_rate):
    '''

    :param data_train: 训练样本
    :param data_test: 测试样本
    :param k: 因式分解参数
    :param iter: 训练迭代次数
    :param learning_rate: 学习率
    :return:
    '''
    # Load datasets

    trainFeature, trainLabel = data_preprocessing(data_train)
    w0, w, v = sgd_fm(trainFeature, trainLabel, k, iter, learning_rate)
    # 返回训练好的权值
    return w0, w, v


if __name__ == '__main__':
    data_train = pd.read_csv('C:/Users/gjt76/iCloudDrive/FM/diabetes_train.txt', header=None)
    data_test = pd.read_csv('C:/Users/gjt76/iCloudDrive/FM/diabetes_test.txt', header=None)
    w0, w, v = train(data_train, 20, 300, 0.01)
    testFeature, testLabel = data_preprocessing(data_test)
    rowNum = np.shape(data_test)[0]
    for i in range(rowNum):
        yPredict = predict(w0,w,v,testFeature[i],0.5)
        print("预测值",yPredict,'实际值',testLabel[i])
