from math import log


def entropy(data):
    label = {}
    for x in data:
        current_label = x[-1]
        if current_label not in label.keys():
            label[current_label] = 1
        else:
            label[current_label] += 1
    result = 0.0
    for x in label.items():
        prob = float(int(x[1]) / len(data))
        result -= prob * log(prob, 2)
    return result


def get_dataset():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def query(dataSet, col_num, value):
    return [x for x in ds if dataSet[col_num] == value]


ds = get_dataset()[0]
# ds[0][-1] = 'maybe'
# print(entropy(ds))
# test
