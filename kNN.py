from numpy import *
import operator


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
    # 排序距离结果
    sorted_dis = dis.argsort()
    print(sorted_dis)


g, l = create_dataset()
classify0([0, 0], g, l, 3)
