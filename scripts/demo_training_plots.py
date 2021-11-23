import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from settings import NUMBER_OF_ACTIVITIES, LIST_OF_MAPPED_2_ACTIVITIES
from datasets import get_dataset_from_file, convert_to_one_hot, get_data_of_days

from settings import CLIENT_START_DAY


def smooth_data(data, smoothing_weight):
    last = data[0]
    for i in range(1, data.shape[0]):
        data[i] = last * smoothing_weight + (1 - smoothing_weight) * data[i]
        last = data[i]

    return data


def load_data():
    data_address_not_local = './results/not_local_90_homes_results/eva_accuracies.txt'
    data_address_local = './results/local_90_home_results/eva_accuracies.txt'
    # data_address_fl = './results/fl_90_homes_results/accuracies.txt'
    data_address_fl = './scripts/updatedlogs/fed_acc.txt'

    not_local_data = np.array(np.loadtxt(fname=data_address_not_local, delimiter=','))
    local_data = np.array(np.loadtxt(fname=data_address_local, delimiter=','))
    fl_data = np.array(np.loadtxt(fname=data_address_fl, delimiter=','))

    return local_data, not_local_data, fl_data


@st.cache(allow_output_mutation=True)
def load_pie_data(home_id, num_days):
    dataset_d = get_dataset_from_file(home_id)
    not_repeated = dataset_d.get_not_repeated_activity_data()
    one_hot = convert_to_one_hot(not_repeated)
    train_day_index = get_data_of_days(not_repeated, num_days)
    activities_d = one_hot[:train_day_index, -NUMBER_OF_ACTIVITIES:]
    # print("\n\nClient" + str(i) + "\n")
    stats = list()
    labels = list()
    for i, act in enumerate(LIST_OF_MAPPED_2_ACTIVITIES):
        activity_count = int(sum(activities_d[:, i]))
        stats.append(activity_count)
        labels.append(act)

    return stats, labels


FONTSIZE = 16
font = {'family': 'normal', 'size': FONTSIZE}



if __name__ == '__main__':
    local_data, not_local_data, fl_data = load_data()

    home_id = st.number_input(label='home_id', min_value=101, max_value=130)

    end_day = 78

    if home_id == 103:
        x = [d for d in range(5, end_day)]
        y_local = smooth_data(local_data[3:(end_day - 2), home_id - 101], 0.7)
        y_not_local = smooth_data(not_local_data[2:end_day - 3, home_id - 101], 0.7)
        y_fl = smooth_data(fl_data[:end_day - 5, home_id - 101], 0.7)
    else:
        x = [d for d in range(CLIENT_START_DAY[home_id], end_day)]
        y_local = smooth_data(local_data[:(end_day - CLIENT_START_DAY[home_id]), home_id - 101], 0.7)
        y_not_local = smooth_data(not_local_data[CLIENT_START_DAY[home_id] - 3:end_day-3, home_id - 101], 0.7)
        y_fl = smooth_data(fl_data[CLIENT_START_DAY[home_id] - 5:end_day - 5, home_id - 101], 0.7)

    figure = plt.figure(figsize=(5, 5))

    plt.plot(x, y_local, '-', label='Local', linewidth=4.0)
    # plt.plot(x, y_not_local, '*-', label='Centralized', linewidth=4.0)
    plt.plot(x, y_fl, '--', label='Federated', linewidth=4.0)

    point_x = -100
    point_y = 0
    for ind in range(len(x)):
        if y_local[ind] > y_fl[ind]:
            point_x = ind + CLIENT_START_DAY[home_id] - 1
            point_y = y_fl[ind - 1] + 0.03
            break

    if point_x > CLIENT_START_DAY[home_id]:
        plt.scatter(point_x, point_y, marker='v', c='chartreuse', s=80, zorder=10, label='Cross Point')
    plt.fill_between(x, y_local, y_fl, where=y_local < y_fl, color='#dde569', alpha=1, label='Regret')

    plt.xlabel("Day", fontsize=FONTSIZE)
    plt.ylabel("Categorical Accuracy", fontsize=FONTSIZE)
    plt.ylim([0.1, 1])
    plt.xlim(CLIENT_START_DAY[home_id], 80)
    plt.grid(True, axis='y')
    plt.title("Home " + str(home_id - 100), fontdict=font)
    plt.legend(loc=4, prop={'size': FONTSIZE})

    st.pyplot(dpi=180)

    for num_days in (2, 5, 10, 30, -1):
        if num_days == -1:
            num_days = st.sidebar.slider(label='Num days', value=2, min_value=2, max_value=100)
        # num_days =
        st.sidebar.markdown(f'num_days: {num_days}')
        stats, labels = load_pie_data(home_id, num_days)

        figure = plt.figure(figsize=(5, 5))
        patches, texts, autotexts = plt.pie(stats, labels=labels, autopct='%1.1f%%')
        for i in range(len(texts)):
            autotexts[i].set_fontsize(16)
            texts[i].set_fontsize(16)
        plt.title("Home " + str(home_id - 100), fontdict=font)
        st.sidebar.pyplot(dpi=180)
