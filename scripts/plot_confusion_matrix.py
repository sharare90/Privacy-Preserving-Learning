import matplotlib.pyplot as plt
import numpy as np
from settings import LIST_OF_MAPPED_ACTIVITIES, LIST_OF_MAPPED_2_ACTIVITIES

if __name__ == "__main__":
    # confusion_matrix_path = './confusion_matrices/122.csv'
    # np_array = np.genfromtxt(confusion_matrix_path, delimiter=',')
    confusion_matrix_path_1 = './confusion_matrices/1.csv'
    confusion_matrix_path_2 = './confusion_matrices/2.csv'
    confusion_matrix_path_3 = './confusion_matrices/3.csv'
    confusion_matrix_path_4 = './confusion_matrices/4.csv'
    confusion_matrix_path_5 = './confusion_matrices/5.csv'
    confusion_matrix_path_6 = './confusion_matrices/6.csv'
    # confusion_matrix_path_1 = './confusion_matrices/new_data_fed_confs/1.csv'
    # confusion_matrix_path_2 = './confusion_matrices/new_data_fed_confs/2.csv'
    # confusion_matrix_path_3 = './confusion_matrices/new_data_fed_confs/3.csv'
    # confusion_matrix_path_4 = './confusion_matrices/new_data_fed_confs/4.csv'
    # confusion_matrix_path_5 = './confusion_matrices/new_data_fed_confs/5.csv'
    # confusion_matrix_path_6 = './confusion_matrices/new_data_fed_confs/6.csv'
    # confusion_matrix_path_7 = './confusion_matrices/new_data_local_confs/128.csv'
    # confusion_matrix_path_8 = './confusion_matrices/new_data_local_confs/129.csv'
    # confusion_matrix_path_9 = './confusion_matrices/new_data_local_confs/130.csv'

    np_array_1 = np.genfromtxt(confusion_matrix_path_1, delimiter=',')
    np_array_2 = np.genfromtxt(confusion_matrix_path_2, delimiter=',')
    np_array_3 = np.genfromtxt(confusion_matrix_path_3, delimiter=',')
    np_array_4 = np.genfromtxt(confusion_matrix_path_4, delimiter=',')
    np_array_5 = np.genfromtxt(confusion_matrix_path_5, delimiter=',')
    np_array_6 = np.genfromtxt(confusion_matrix_path_6, delimiter=',')
    # np_array_7 = np.genfromtxt(confusion_matrix_path_7, delimiter=',')
    # np_array_8 = np.genfromtxt(confusion_matrix_path_8, delimiter=',')
    # np_array_9 = np.genfromtxt(confusion_matrix_path_9, delimiter=',')

    np_array = np_array_1 + np_array_2 + np_array_3 + np_array_4 + np_array_5 + np_array_6

    np_array = np_array_1[1:, 1:]
    # np_array[np_array > 300] = 200
    # 0'ed other activity cell because it was too big
    np_array[1, 1] = 0
    fig, ax = plt.subplots()
    im = ax.imshow(np_array)
    # plt.imshow(np_array)

    # Create colorbar
    # cbarlabel = "Confusion matrix for Client 9"
    cbarlabel = "Confusion matrix for all experiment clients with local training"
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=18)

    ax.set_xticks(np.arange(len(LIST_OF_MAPPED_2_ACTIVITIES)))
    ax.set_yticks(np.arange(len(LIST_OF_MAPPED_2_ACTIVITIES)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(LIST_OF_MAPPED_2_ACTIVITIES, fontdict={'fontsize': 18})
    ax.set_yticklabels(LIST_OF_MAPPED_2_ACTIVITIES, fontdict={'fontsize': 18})
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-20, ha="right",
             rotation_mode="anchor")
    # ax.set_title("Confusion matrix for Client 1")
    # fig.tight_layout()

    plt.show()
