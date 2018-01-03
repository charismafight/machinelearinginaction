from math import log
import operator
from collections import defaultdict


def entropy(data):
    label = defaultdict(int)
    for x in data:
        if x in label.keys():
            label[x] += 1
    result = 0.0
    for x in label.items():
        prob = float(int(x[1]) / len(data))
        result -= prob * log(prob, 2)
    return result


def get_dataset():
    dataSet = [[1, '1', 'yes'], [1, '1', 'yes'], [1, '0', 'no'],
               [0, '1', 'no'], [0, '1', 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def best_feature_index(data):
    """
    return the best feature column number
    """
    entropies = [entropy(x) for x in zip(*data)]
    return entropies.index(max(entropies))


# print(choose_best_feature(get_dataset()[0]))
# modify auto fetch config
def majority_count(classes):
    counts = defaultdict(int)
    for v in classes:
        counts[v] += 1
    sorted_class = sorted(
        counts.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class[0][0]


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def build_tree(dataset, labels):
    class_list = [x[-1] for x in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return majority_count(class_list)

    if len(set([x[-1] for x in dataset])) == 1:
        return dataset[0][-1]

    # get the best feature and contribute a dict with bestfeature name(from the label) as its key
    current_feature_index = best_feature_index(dataset)
    current_feature_name = labels[current_feature_index]
    my_tree = {current_feature_name: {}}
    # only delete column before loop,because looping(different value of the same attribute) should use the same label
    # so labels will be changed in the loop,we use a local variable to save it
    del(labels[current_feature_index])

    trans_dataset = list(map(set, zip(*dataset)))
    for f in trans_dataset[current_feature_index]:
        sub_label = labels[:]
        f_dataset = splitDataSet(dataset, current_feature_index, f)
        my_tree[current_feature_name][f] = build_tree(f_dataset, sub_label)
    return my_tree


print(build_tree(*get_dataset()))
