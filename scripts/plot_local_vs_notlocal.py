import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import matplotlib
import json
from settings import CLIENT_START_DAY


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


data_address_not_local = './results/not_local_90_homes_results/eva_accuracies.txt'

data_address_local = './results/local_90_home_results/eva_accuracies.txt'

data_address_fl = './results/fl_90_homes_results/accuracies.txt'

data_address_fl_local = './results/federated_local_30_home_results_1/eva_accuracies.txt'

# data_address_not_local = './results/not_local_results_wtd_4/eva_my_accuracies.txt'
#
# data_address_local = './results/local_results_wtd_4/eva_my_accuracies.txt'

not_local_data = np.array(np.loadtxt(fname=data_address_not_local, delimiter=','))
local_data = np.array(np.loadtxt(fname=data_address_local, delimiter=','))
fl_data = np.array(np.loadtxt(fname=data_address_fl, delimiter=','))
fl_local_data = np.array(np.loadtxt(fname=data_address_fl_local, delimiter=','))
#
# not_local_data = not_local_data.astype(np.float)
# local_data = local_data.astype(np.float)
# x_cent = [d for d in range(3, 60)]
# x_fed = [d for d in range(5, 60)]

# x = np.arange(0, 25, 0.1)
# fig, axis = plt.subplots(3, 4)
fig, axis = plt.subplots(5, 6)
end_day = 70
# to_plot = [0, 1, 2, 8, 13, 16, 17, 22, 24, 26, 28, 29]
# to_plot = [2, 5, 6, 7, 8, 10, 11, 13, 14, 16, 28, 29]
to_plot = [i for i in range(30)]
for p, i in enumerate(to_plot):
    if i == 2:
        x = [d for d in range(5, end_day)]
        y_local = smooth_data(local_data[3:(end_day - 2), i], 0.7)
        y_not_local = smooth_data(not_local_data[2:end_day - 3, i], 0.7)
        y_fl = smooth_data(fl_data[:end_day - 5, i], 0.7)
        y_fl_local = smooth_data(fl_local_data[:end_day - 5, i], 0.7)

    else:
        x = [d for d in range(CLIENT_START_DAY[i + 101], end_day)]
        y_local = smooth_data(local_data[:(end_day - CLIENT_START_DAY[i + 101]), i], 0.7)
        y_not_local = smooth_data(not_local_data[CLIENT_START_DAY[i + 101] - 3:end_day - 3, i], 0.7)
        y_fl = smooth_data(fl_data[CLIENT_START_DAY[i + 101] - 5:end_day - 5, i], 0.7)
        y_fl_local = smooth_data(fl_local_data[CLIENT_START_DAY[i + 101] - 5:end_day - 5, i], 0.7)

    # axis[p // 4][p % 4].plot(x, y_local, 'm--', label='Local', linewidth=2.0)
    # axis[p // 4][p % 4].plot(x, y_not_local, '-', label='Centralized', linewidth=2.0)
    # axis[p // 4][p % 4].plot(x, y_fl, '-', label='Federated', linewidth=2.0)
    # axis[p // 4][p % 4].plot(x, y_fl_local, '-', label='Federated_Local', linewidth=2.0)

    axis[p // 6][p % 6].plot(x, y_local, 'm--', label='Local', linewidth=2.0)
    axis[p // 6][p % 6].plot(x, y_not_local, '-', label='Centralized', linewidth=2.0)
    axis[p // 6][p % 6].plot(x, y_fl, '-', label='Federated', linewidth=2.0)
    axis[p // 6][p % 6].plot(x, y_fl_local, 'k-', label='Federated_Local', linewidth=2.0)

    point_x = -100
    point_y = 0
    for ind in range(len(x)):
        if y_local[ind] > y_fl[ind]:
            point_x = x[ind] - 1
            point_y = y_local[ind] + 0.04
            break

    point_xx = -100
    point_yy = 0
    for ind in range(len(x)):
        if y_local[ind] > y_fl_local[ind]:
            point_xx = x[ind] - 1
            point_yy = y_local[ind] + 0.04
            break

    if point_x >= CLIENT_START_DAY[i + 101] - 1:
        axis[p // 6][p % 6].scatter(point_x, point_y, marker='v', c='red', s=200, zorder=10,
                                    label='Federated Crossover')
        axis[p // 6][p % 6].scatter(point_xx, point_yy, marker='v', c='blue', s=200, zorder=10,
                                    label='Federated Local Crossover')
    axis[p // 6][p % 6].fill_between(x, y_local, y_fl, where=y_local <= y_fl, color='#dde569', alpha=1, label='Regret')

    fig.text(0.5, 0.01, "Day", ha='center')
    # plt.ylabel("Accuracy with windowing", fontsize=42)
    fig.text(0.01, 0.5, "Categorical Accuracy", va='center', rotation='vertical')
    axis[p // 6][p % 6].set_ylim([0.1, 0.9])
    # axis[p // 4][p % 4].set_xlim(CLIENT_START_DAY[i + 101], 70)
    axis[p // 6][p % 6].set_xlim(2, 70)
    # axis[p // 4][p % 4].set_xticks(np.arange(min(x), max(x) + 1, 5.0))
    # axis[p // 4][p % 4].set_xticks([i for i in range(CLIENT_START_DAY[i + 101], 70, 10)])
    axis[p // 6][p % 6].grid(True, axis='both')
    # plt.title("Prediction on Dataset of Client 9", fontdict=font)
    axis[p // 6][p % 6].set_title("Home " + str(i + 1), fontdict=font)

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc="center right")
plt.subplots_adjust(bottom=0.06, top=0.96, right=0.88, left=0.05, wspace=0.20, hspace=0.30)
plt.savefig('30homes.pdf', dpi=1000)
plt.show()
