import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import matplotlib
import json
from settings import CLIENT_START_DAY_NEW_DATA

def smooth_data(data, smoothing_weight):
    last = data[0]
    for i in range(1, data.shape[0]):
        data[i] = last * smoothing_weight + (1 - smoothing_weight) * data[i]
        last = data[i]

    return data


font = {'family': 'Times New Roman',
        # 'weight': 'bold',
        'size': 11}

matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42


def add_to_plot(data_address, label, style):
    data = json.load(open(data_address))
    data = np.array(data)
    x = np.array(data[:, 1])
    f1_score = np.array(data[:, 2])
    plt.plot(x, smooth_data(f1_score, 0.4), style, label=label, linewidth=2.0)


# data_address_local = './results/local_6_home_results_lre4_d1/eva_accuracies.txt'
data_address_local = './results/local_6_home_results_lre4_weighted_64/eva_accuracies.txt'


data_address_fl_local = './results/federated_local_6_home_results_after89_lre4_weighted/eva_accuracies.txt'
data_address_fl_local_new = './results/federated_local_6_home_results_after42_lre4_weighted/eva_accuracies.txt'

data_address_fl = './results/fl_6_homes_results_lre4/eva_accuracies.txt'
data_address_not_local = './results/not_local_6_homes_results_lre3_weighted/eva_my_accuracies.txt'
#
# data_address_local = './results/local_results_wtd_4/eva_my_accuracies.txt'

local_data = np.array(np.loadtxt(fname=data_address_local, delimiter=','))
not_local_data = np.array(np.loadtxt(fname=data_address_not_local, delimiter=','))
fl_local_data = np.array(np.loadtxt(fname=data_address_fl_local, delimiter=','))
fl_local_data_new = np.array(np.loadtxt(fname=data_address_fl_local_new, delimiter=','))

fl_data = np.array(np.loadtxt(fname=data_address_fl, delimiter=','))

#
# not_local_data = not_local_data.astype(np.float)
# local_data = local_data.astype(np.float)
# x_cent = [d for d in range(3, 60)]
# x_fed = [d for d in range(5, 60)]

# x = np.arange(0, 25, 0.1)
# fig, axis = plt.subplots(3, 4)
fig, axis = plt.subplots(2, 3)
end_day = 40
# to_plot = [0, 1, 2, 8, 13, 16, 17, 22, 24, 26, 28, 29]
# to_plot = [2, 5, 6, 7, 8, 10, 11, 13, 14, 16, 28, 29]
to_plot = [i for i in range(6)]
for p, i in enumerate(to_plot):
    if i in [0, 1, 2]:
        x = [d for d in range(5, end_day)]
        y_local = smooth_data(local_data[2:end_day - 3, i], 0.9)
        y_not_local = smooth_data(not_local_data[2:end_day - 3, i], 0.9)
        y_fl_local = smooth_data(fl_local_data[:end_day - 5, i], 0.9)
        y_fl_local_new = smooth_data(fl_local_data_new[:end_day - 5, i], 0.9)
        y_fl = smooth_data(fl_data[1:end_day - 4, i], 0.9)
    else:
        x = [d for d in range(CLIENT_START_DAY_NEW_DATA[i + 1], end_day)]
        y_local = smooth_data(local_data[CLIENT_START_DAY_NEW_DATA[i + 1] - 3:end_day - 3, i], 0.9)
        y_not_local = smooth_data(not_local_data[CLIENT_START_DAY_NEW_DATA[i + 1] - 3:end_day - 3, i], 0.9)
        y_fl_local = smooth_data(fl_local_data[CLIENT_START_DAY_NEW_DATA[i + 1] - 5:end_day - 5, i], 0.9)
        y_fl_local_new = smooth_data(fl_local_data_new[CLIENT_START_DAY_NEW_DATA[i + 1] - 5:end_day - 5, i], 0.9)
        y_fl = smooth_data(fl_local_data[CLIENT_START_DAY_NEW_DATA[i + 1] - 4:end_day - 4, i], 0.9)

    axis[p // 3][p % 3].plot(x, y_local, 'm--', label='Local', linewidth=2.0)
    axis[p // 3][p % 3].plot(x, y_not_local, '-', label='Centralized', linewidth=2.0)
    axis[p // 3][p % 3].plot(x, y_fl_local, '-', label='CASAS Federated_Local', linewidth=2.0)
    axis[p // 3][p % 3].plot(x, y_fl_local_new, '-', label='ADLs Federated_Local', linewidth=2.0)
    axis[p // 3][p % 3].plot(x, y_fl, '-', label='Federated', linewidth=2.0)

    point_x = -100
    point_y = 0

    fig.text(0.5, 0.01, "Day", ha='center')
    # plt.ylabel("Accuracy with windowing", fontsize=42)
    fig.text(0.01, 0.5, "Categorical Accuracy", va='center', rotation='vertical')
    axis[p // 3][p % 3].set_ylim([0.0, 0.9])
    # axis[p // 4][p % 4].set_xlim(CLIENT_START_DAY[i + 101], 70)
    axis[p // 3][p % 3].set_xlim(2, 43)
    # axis[p // 4][p % 4].set_xticks(np.arange(min(x), max(x) + 1, 5.0))
    # axis[p // 4][p % 4].set_xticks([i for i in range(CLIENT_START_DAY[i + 101], 70, 10)])
    axis[p // 3][p % 3].grid(True, axis='both')
    # plt.title("Prediction on Dataset of Client 9", fontdict=font)
    axis[p // 3][p % 3].set_title("Home " + str(i + 1), fontdict=font)

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = "center right")
plt.subplots_adjust(bottom=0.06, top=0.96, right=0.88, left=0.05, wspace=0.20, hspace=0.30)
plt.show()
