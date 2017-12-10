# -*- coding: utf8 -*-
from numpy import *
import operator
import os.path as path
import os
import matplotlib
from matplotlib import pyplot as plt


def create_dataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataset, labels, k):
    dataset_size = len(dataset)
    # 用inX来构造一个和预设的group格式相同的矩阵，并矩阵-dataset
    # 本质就是公式(x-x1)**2  (y-y1)**2，这里用矩阵一次性做了处理，先做括号里的减法
    diff_mat = tile(inX, (dataset_size, 1)) - dataset
    # 再把结果平方
    sq_diff_mat = diff_mat**2
    # 再把平方后的结果相加 也就是 (x-x1)**2 [+]这个+号 (y-y1)**2
    sq_dis = sq_diff_mat.sum(axis=1)
    # 加完后求开平方
    dis = sq_dis**0.5
    # 排序距离结果,索引值
    sorted_dis = dis.argsort()
    class_count = {}
    # 计算前k个里面每一类别出现的次数
    for i in range(k):
        v = labels[sorted_dis[i]]
        class_count[v] = class_count.get(v, 0) + 1
    sorted_class_count = sorted(
        class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# g, l = create_dataset()
# print(classify0([1, 1], g, l, 3))


# 读取训练例子
# 这个例子不够简洁，其实只需要读一次文件构造出结果即可，不过暂时按书上写
def file2matrix(filename):
    f = open(filename)
    rows = len(f.readlines())
    mat = zeros((rows, 3))
    class_label_vector = []
    f = open(filename)
    index = 0
    for line in f.readlines():
        line = line.strip()
        lines = line.split('\t')
        mat[index, :] = lines[0:3]
        class_label_vector.append(int(lines[-1]))
        index += 1
    return mat, class_label_vector


# normalize data to a num between 0-1
def auto_norm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# mat, labels = file2matrix(path.join(os.sys.path[0], 'datingTestSet2.txt'))

# mat, ranges, min_value = auto_norm(mat)

# print(mat)
# # fig for playing video game and icecream
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(mat[:, 1], mat[:, 2], 15 * array(labels), 15 * array(labels))

# # fig2 = plt.figure()
# # ax2 = fig2.add_subplot(1, 1, 1)
# # ax2.scatter(mat[:, 0], mat[:, 1], 15 * array(labels), 15 * array(labels))
# plt.show()


# test how accurate the algorithm is
def datingClassTest():
    hoRatio = 0.1
    mat, labels = file2matrix(path.join(os.sys.path[0], 'datingTestSet2.txt'))
    mat, ranges, min_value = auto_norm(mat)
    m = mat.shape[0]
    num_test_vecs = int(m * hoRatio)

    error_count = 0
    # 用前10%的数据结果，与之后的所有数据进行距离计算
    for i in range(num_test_vecs):
        result = classify0(mat[i, :], mat[num_test_vecs:m, :],
                           labels[num_test_vecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" %
              (result, labels[i]))

        if result != labels[i]:
            error_count += 1

    print("the total error: %d" % error_count)


datingClassTest()