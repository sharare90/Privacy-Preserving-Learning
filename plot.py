import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import matplotlib


def smooth_data(data, smoothing_weight):
    last = data[0]
    for i in range(1, data.shape[0]):
        data[i] = last * smoothing_weight + (1 - smoothing_weight) * data[i]
        last = data[i]

    return data


font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)

# sns.set_context("paper")
# plt.plot(smooth_data(loss_train, 0.9), label="Train", linewidth=3.0)
# plt.plot(smooth_data(loss_validation, 0.9), '--', label="Validation", linewidth=3.0)
# plt.xlabel("Number of epochs", fontsize=22, fontweight='bold')
# plt.ylabel("F1-score", fontsize=22, fontweight='bold')
# plt.legend(loc="best", prop={'size': 22})
# plt.xlim(0, 225)
# plt.ylim(0, 1)
# plt.grid(True, color="darkseagreen", linewidth=0.5)
# # plt.title("F1-score on SimulDataset 1", fontsize=22, fontweight='bold')
# plt.show()

# Plot loss based on size of data

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
train_acc = np.array([0.61, 0.65, 0.69, 0.71, 0.72, 0.73, 0.74, 0.74, 0.74, 0.75, 0.75, 0.75, 0.75, 0.76, 0.76, 0.76, 0.76, 0.76,
             0.76, 0.76, 0.77, 0.77, 0.77, 0.77])
val_acc = np.array([ 0.02, 0.17, 0.48, 0.53, 0.55, 0.56, 0.56, 0.57, 0.57, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58,
           0.58, 0.59, 0.59, 0.59, 0.59, 0.59])


plt.plot(x, smooth_data(train_acc, 0.6), 'orange', label="Train Categorical Accuracy", linewidth=3.0, )
plt.plot(x, smooth_data(val_acc, 0.6), 'blue', label="Validation Categorical Accuracy", linewidth=3.0)
plt.legend(loc="lower right", prop={'size': 20})
plt.xlabel("Round", fontsize=18, fontweight='bold')
plt.ylabel("Categorical Accuracy", fontsize=18, fontweight='bold')
plt.grid(True, color="darkseagreen", linewidth=0.5)
plt.show()
