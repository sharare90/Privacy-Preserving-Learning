import numpy as np
import sklearn
import os
import tensorflow as tf

import settings
from datasets import get_dataset_from_file, get_data_of_days, convert_to_one_hot
from lstm_model import LSTM_train_test
from utils import build_dataset


def create_local_train_test_dataset(dataset_num, num_days):
    dataset_d = get_dataset_from_file(dataset_num)
    not_repeated = dataset_d.get_not_repeated_activity_data()
    d = convert_to_one_hot(not_repeated)
    train_day_index = get_data_of_days(not_repeated, num_days)
    test_start_index = int(0.7 * len(d))
    if test_start_index >= train_day_index:
        train_d = d[:train_day_index, :]
    else:
        train_d = d[:test_start_index, :]
        # raise Exception("Number of days for training passed 70% limit of the train dataset.")
    test_d = d[test_start_index:, :]
    train_data_d, train_labels_d = build_dataset(train_d)
    test_data_d, test_labels_d = build_dataset(test_d)

    train_data_shuffled, train_labels_shuffled = sklearn.utils.shuffle(train_data_d, train_labels_d, random_state=0)
    return train_data_shuffled, train_labels_shuffled, test_data_d, test_labels_d


def get_window_accuracy(predicted, labels, window_size=3):
    preds = tf.argmax(predicted, axis=1)
    labs = tf.argmax(labels, axis=1)
    correct = 0
    for i in range(len(preds)):
        if preds[i] in labs[i: i + window_size]:
            correct += 1
    return correct / len(preds)


def run_federated_local():
    # logdir = "./logs_local_days_topk_24/"

    # num_clients = 9
    num_clients = 30
    # num_day_experiments = 30
    num_day_experiments = 36
    losses = np.zeros((num_day_experiments, num_clients))
    accuracies = np.zeros((num_day_experiments, num_clients))
    my_accuracies = np.zeros((num_day_experiments, num_clients))
    first_test_home = 101

    for num_days in range(34, 70):
        for client_id in range(first_test_home, 131):
            lstm = LSTM_train_test(output_size=settings.NUMBER_OF_ACTIVITIES)
            lstm.compile()
            try:
                data_train, labels_train, data_test, labels_test = create_local_train_test_dataset(client_id,
                                                                                                   num_days)
            except Exception as e:
                # raise (e)
                print(e)
                break

            size = labels_train.shape[0]
            # class_weights = {i: size / np.sum(labels_train[:, i]) for i in range(10)}
            # class_weights = {i: 1 for i in range(10)}
            # class_weights = {
            #     0: 1,
            #     1: 2,
            #     2: 4,
            #     3: 4,
            #     4: 2,
            #     5: 4,
            #     6: 2,
            #     7: 4,
            #     8: 4,
            #     9: 4,
            # }

            # class_weights[0] = 0.25
            # class_weights[1] = 0.5
            # class_weights[4] = 0.5
            # class_weights[6] = 0.5

            # print(class_weights)
            # train_summary_writer = tf.summary.create_file_writer(
            #     os.path.join(logdir, f'train_{num_days}_cl_{client_id}'))
            # evaluate_summary_writer = tf.summary.create_file_writer(
            #     os.path.join(logdir, f'evaluate_{num_days}_cl_{client_id}'))
            lstm.load_weights(
                '/home/sharare/PycharmProjects/FederatedLearning_Caching/saved_model/' + f'{num_days}')
            train_history = lstm.train(
                data_train=data_train,
                labels_train=labels_train,
                batch_size=settings.BATCH_SIZE,
                num_epochs=500
            )
            lstm.save_weights(
                '/home/sharare/PycharmProjects/FederatedLearning_Caching/saved_model_fl_local/' + str(
                    client_id) + '/' + f'{num_days}')

            loss, cat_accuracy = lstm.evaluate(data_test, labels_test)
            print("Evaluation on client " + str(client_id))
            predicted = lstm.predict(data_test)
            print('results on some data points\n\n')
            print(tf.argmax(predicted[:5], axis=1), tf.argmax(labels_test[:5], axis=1))
            print('\n\n')
            my_accuracy = get_window_accuracy(predicted, labels_test)

            losses[num_days - 34, client_id - first_test_home] = loss
            accuracies[num_days - 34, client_id - first_test_home] = cat_accuracy
            my_accuracies[num_days - 34, client_id - first_test_home] = my_accuracy

            # with evaluate_summary_writer.as_default():
            #     tf.summary.scalar('evaluate loss', loss, step=it)
            #     tf.summary.scalar('evaluate categorical accuracy', cat_accuracy, step=it)
            #     evaluate_summary_writer.flush()
            #
            # with train_summary_writer.as_default():
            #     tf.summary.scalar('train loss', train_history.history['loss'][0], step=it)
            #     tf.summary.scalar('train categorical accuracy', train_history.history['categorical_accuracy'][0],
            #                       step=it)
            #     train_summary_writer.flush()

    # for i in range(num_clients):
    os.makedirs('./results/federated_local_30_home_results_2/', exist_ok=True)
    np.savetxt("./results/federated_local_30_home_results_2/eva_losses.txt", losses, delimiter=',')
    np.savetxt("./results/federated_local_30_home_results_2/eva_accuracies.txt", accuracies, delimiter=',')
    np.savetxt("./results/federated_local_30_home_results_2/eva_my_accuracies.txt", my_accuracies, delimiter=',')

    return lstm


if __name__ == '__main__':
    # lstm = run_not_local()
    lstm = run_federated_local()
