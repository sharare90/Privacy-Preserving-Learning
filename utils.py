import numpy as np

from settings import NUMBER_OF_ACTIVITIES, HISTORY_SIZE
from datasets import get_dataset_from_file


def build_dataset(data, history_size=HISTORY_SIZE):
    start_index = history_size
    end_index = len(data)

    stacked_data = []
    stacked_labels = []

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, 1)
        stacked_data.append(data[indices, :])
        stacked_labels.append(data[i, -NUMBER_OF_ACTIVITIES:])

    stacked_data = np.array(stacked_data)
    stacked_labels = np.array(stacked_labels)
    return stacked_data, stacked_labels


def num_days(data):
    data = data.data
    previous = data[0, 0]
    num_days = 0
    for line in data:
        current = list(line)[0]
        if float(current) < float(previous):
            num_days += 1
        previous = current
    return num_days


def get_confusion_matrix(preds, lbls):
    n = preds.shape[1]
    confusion_matrix = np.zeros((n, n))

    preds = np.argmax(preds, axis=1)
    lbls = np.argmax(lbls, axis=1)
    for i in range(len(preds)):
        confusion_matrix[lbls[i], preds[i]] += 1

    return confusion_matrix


if __name__ == "__main__":
    for i in range(101, 131):
        print(num_days(get_dataset_from_file(i)))

    # 59
    # 53
    # 57
    # 60
    # 36
    # 49
    # 28
    # 56
    # 60
    # 26
    # 54
    # 94
    # 488
    # 30
    # 296
    # 59
    # 221
    # 19
    # 30
    # 63
    # 9
    # 26
    # 30
    # 13
    # 57
    # 36
    # 26
    # 60
    # 28
    # 29


