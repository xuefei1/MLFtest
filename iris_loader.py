import numpy as np

def load_iris(path, train_percent):
    with open(path) as f:
        raw = f.readlines()
        split = [l.rstrip().split(',') for l in raw]
    data = [[[float(l[0]), float(l[1]), float(l[2]), float(l[3])], iris_label_to_vect(l[4]), l[4]]
            for l in split]
    np.random.shuffle(data)
    class_0 = [d for d in data if d[1] == 0]
    class_1 = [d for d in data if d[1] == 1]
    class_2 = [d for d in data if d[1] == 2]
    train_amount_class_0 = int(len(class_0) * train_percent)
    train_amount_class_1 = int(len(class_1) * train_percent)
    train_amount_class_2 = int(len(class_2) * train_percent)
    train_data = []
    test_data = []
    train_data += class_0[:train_amount_class_0]
    train_data += class_1[:train_amount_class_1]
    train_data += class_2[:train_amount_class_2]
    test_data += class_0[train_amount_class_0:]
    test_data += class_1[train_amount_class_1:]
    test_data += class_2[train_amount_class_2:]
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    return train_data, test_data


def iris_label_to_vect(label):
    if label == 'Iris-setosa':
        return 0
    elif label == 'Iris-versicolor':
        return 1
    elif label == 'Iris-virginica':
        return 2
    return -1

if __name__ == '__main__':
    train_set, test_set = load_iris('iris.txt', 0.5)

