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


def choose_best_feature(data):
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
        if v in counts:
            counts[v] += 1
    sorted_class = sorted(
        counts.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class[0][0]
