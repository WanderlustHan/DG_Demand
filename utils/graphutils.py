import math

import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def knn(inX, dataSet, k):
    dataSetSize = dataSet.shape[0]
    tileX = np.tile(inX, (dataSetSize, 1))
    diffMat = torch.from_numpy(tileX) - dataSet
    sqDiffMat = np.array(diffMat) ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    return sortedDistIndicies[0:k]
"""
    dataMat : 输入的adjacency matrix / global time series matrix，shape = (n, n)
    k : knn最近的k个邻居，超参，默认为15。knn用以计算W，D.
    t ：计算W的阈值，超参，默认为5.0
    m : 降维的维度大小，超参，默认为64.
"""
def laplaEigen(dataMat, k, t, m):
    n, _ = dataMat.shape
    W = torch.zeros([n, n])
    D = torch.zeros([n, n])
    for i in range(n):
        # 利用KNN找到最近的k个点，W[i,j]=exp(*) if dataMat[i,:] is in the k_index; else =0
        k_index = knn(dataMat[i, :], dataMat, k)
        # 计算 W，D. Wij=exp(-(xi-xj)**2/t); Dii=sum(Wij),j=[0,k)
        for j in range(k):
            sqDiffVector = dataMat[i, :] - dataMat[k_index[j], :]
            sqDiffVector = np.array(sqDiffVector) ** 2
            sqDistances = sqDiffVector.sum()
            W[i, k_index[j]] = W[k_index[j],i] = math.exp(-sqDistances / t)
        D[i, i] = torch.sum(W[i])
    L = D - W
    Dinv = np.linalg.inv(D)
    X = np.dot(Dinv, L)
    # 求解 L*f = lambda*D*f
    lamda, f = np.linalg.eig(X)  # 计算Dinv*L的特征值和特征向量 Dinv*L*f = lamda*f
    lamda_indicies = lamda.argsort() # 将lambda按照从小到大排序
    start = 0
    for i in range(n): # 找到特征值不为0的最小值的索引
        if lamda[lamda_indicies[i]] > 0:
            start = i
            break
    topk_f = f[:, lamda_indicies[start:m + start]] # 取出前[1,m+1]的m个特征值对应的特征向量
    # topk_f.shape = [n, m], 即降维后的矩阵。
    return lamda, torch.from_numpy(np.real(topk_f))


if __name__ == '__main__':
    A = torch.randn(170, 170)
    lamda, f = laplaEigen(A, 15, 5.0, 64)
    print(lamda, f)
