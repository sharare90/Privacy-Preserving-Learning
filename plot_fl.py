import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import matplotlib
import json


def smooth_data(data, smoothing_weight):
    last = data[0]
    for i in range(1, data.shape[0]):
        data[i] = last * smoothing_weight + (1 - smoothing_weight) * data[i]
        last = data[i]

    return data


font = {'family': 'normal',
        # 'weight': 'bold',
        'size': 42}

matplotlib.rc('font', **font)


def add_to_plot(data_address, label, style):
    data = json.load(open(data_address))
    data = np.array(data)

    x = np.array(data[:, 1])
    f1_score = np.array(data[:, 2])
    plt.plot(x, smooth_data(f1_score, 0.7), style, label=label, linewidth=3.0)


# plt.plot(x, smooth_data(loss_validation, 0.6), '--', label="Validation F1-score", linewidth=7.0)

# Train
# avg_data_address = './plots/avg/evaluate/run-evaluate0-tag-evaluation categorical accuracy.json'
# sgd_data_address = './plots/sgd/evaluate/run-evaluate0-tag-evaluation categorical accuracy.json'
# avg_wt_data_address = './plots/avg_wt/evaluate/run-evaluate0-tag-evaluation categorical accuracy.json'
# avg_wd_data_address = './plots/avg_wd/evaluate/run-evaluate0-tag-evaluation categorical accuracy.json'

avg_data_address_tr = './plots/avg/train/run-train-tag-train categorical_accuracy.json'
sgd_data_address_tr = './plots/sgd/train/run-train-tag-train categorical_accuracy.json'
avg_wt_data_address_tr = './plots/avg_wt/train/run-train-tag-train categorical_accuracy.json'
avg_wd_data_address_tr = './plots/avg_wd/train/run-train-tag-train categorical_accuracy.json'

# add_to_plot(avg_data_address, 'Federated Averaging', '-')
# add_to_plot(avg_wt_data_address, 'Federated Averaging-with Time', 'r+')
# add_to_plot(avg_wd_data_address, 'Federated Averaging-with Day', 'g*')
# add_to_plot(sgd_data_address, 'Federated SGD', 'y--')


add_to_plot(avg_data_address_tr, 'Federated Averaging', '-')
add_to_plot(avg_wt_data_address_tr, 'Federated Averaging-with Time', 'r+')
add_to_plot(avg_wd_data_address_tr, 'Federated Averaging-with Day', 'g*')
add_to_plot(sgd_data_address_tr, 'Federated SGD', 'y--')


# add_to_plot(sgd_b256_data_address, 'Federated SGD, B=256', 'r--')
# plt.plot(x, smooth_data(loss_validation, 0.6), '--', label="Validation F1-score", linewidth=7.0)
# plt.legend(loc="upper right", prop={'size': 22})
plt.xlabel("Communication Rounds", fontsize=42)
plt.ylabel("Categorical Accuracy", fontsize=42)
plt.ylim([0, 1])
# plt.grid(True, color="black", linewidth=0.01)
# plt.title("Evaluation on Client 1", fontdict=font)
plt.title("Train on 21 Clients", fontdict=font)
plt.legend(loc=4, prop={'size': 32})
plt.show()
