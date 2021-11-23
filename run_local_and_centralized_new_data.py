import numpy as np
import sklearn
import os
import tensorflow as tf
import csv
from tqdm import tqdm

import settings
from datasets_adls import get_dataset_from_file, get_data_of_days, convert_to_one_hot
from lstm_model import LSTM_train_test
from utils import build_dataset, get_confusion_matrix
from settings import NUMBER_OF_ACTIVITIES, HISTORY_SIZE, TOTAL_NUM_OF_FEATURES, \
    CONFUSION_MATRIX_DIR, LIST_OF_MAPPED_2_ACTIVITIES, LIST_OF_DAYS_OF_UPDATES_NEW_DATA, AVAILABLE_CLIENTS_NEW_DATA, \
    CLIENT_START_DAY_NEW_DATA

np.random.seed(0)
tf.random.set_seed(0)


def create_centralized_train_dataset(dataset_nums, num_days):
    all_data = np.empty((0, HISTORY_SIZE, TOTAL_NUM_OF_FEATURES))
    all_labels = np.empty((0, NUMBER_OF_ACTIVITIES))

    print('Loading dataset')
    for i in tqdm(dataset_nums):
        dataset_d = get_dataset_from_file(i)
        not_repeated = dataset_d.get_not_repeated_activity_data()
        d = convert_to_one_hot(not_repeated)
        train_day_index = get_data_of_days(not_repeated, num_days[i])
        limit_index = int(0.7 * len(d))
        if limit_index >= train_day_index:
            train_d = d[:train_day_index, :]
        else:
            train_d = d[:limit_index, :]
        data_d, labels_d = build_dataset(train_d)
        if len(data_d) == 0:
            continue
        # data_d, labels_d = build_dataset(d)
        all_data = np.concatenate(([all_data, data_d]), axis=0)
        all_labels = np.concatenate(([all_labels, labels_d]), axis=0)
    all_data_shuffled, all_labels_shuffled = sklearn.utils.shuffle(all_data, all_labels, random_state=0)
    return all_data_shuffled, all_labels_shuffled


def create_local_test_dataset(dataset_num, split_point=0.7):
    dataset_d = get_dataset_from_file(dataset_num)
    not_repeated = dataset_d.get_not_repeated_activity_data()
    d = convert_to_one_hot(not_repeated)
    test_start_index = int(split_point * len(d))
    test_d = d[test_start_index:, :]
    test_data_d, test_labels_d = build_dataset(test_d)
    return test_data_d, test_labels_d


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


def check_labels(labels):
    print('checking labels')
    argmax_labels = np.argmax(labels, axis=1)
    for label in range(settings.NUMBER_OF_ACTIVITIES):
        print(f'{label}: {np.sum(argmax_labels == label)}')


def get_window_accuracy(predicted, labels, window_size=3):
    preds = tf.argmax(predicted, axis=1)
    labs = tf.argmax(labels, axis=1)
    correct = 0
    for i in range(len(preds)):
        if preds[i] in labs[i: i + window_size]:
            correct += 1
    return correct / len(preds)


def get_window_accuracy_on_dataset(model, dataset, window_size=3):
    predictions = model.predict(dataset)
    dataset_list = list(dataset)
    labels = np.zeros((len(dataset_list), 10))

    for i, item in enumerate(dataset):
        label = item[1]
        labels[i] = label

    return get_window_accuracy(predictions, labels, window_size=window_size)


def run_local_only():
    logdir = "./logs_local_days_topk_24/"

    # num_clients = 9
    num_clients = 6
    # num_day_experiments = 30
    num_day_experiments = 40
    losses = np.zeros((num_day_experiments, num_clients))
    accuracies = np.zeros((num_day_experiments, num_clients))
    my_accuracies = np.zeros((num_day_experiments, num_clients))
    first_test_home = 1

    for num_days in range(3, 43):
        for client_id in range(first_test_home, 7):
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

            train_history = lstm.train(
                data_train=data_train,
                labels_train=labels_train,
                batch_size=settings.BATCH_SIZE,
                num_epochs=500
            )

            loss, cat_accuracy = lstm.evaluate(data_test, labels_test)
            # loss, cat_accuracy = lstm.evaluate(data_train, labels_train)

            print("Evaluation on client " + str(client_id))
            predicted = lstm.predict(data_test)
            print('results on some data points\n\n')
            print(tf.argmax(predicted[:5], axis=1), tf.argmax(labels_test[:5], axis=1))
            print('\n\n')
            my_accuracy = get_window_accuracy(predicted, labels_test)

            losses[num_days - 3, client_id - first_test_home] = loss
            accuracies[num_days - 3, client_id - first_test_home] = cat_accuracy
            my_accuracies[num_days - 3, client_id - first_test_home] = my_accuracy

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
    os.makedirs('./results/local_6_home_results_lre4_weighted_16/', exist_ok=True)
    np.savetxt("./results/local_6_home_results_lre4_weighted_16/eva_losses.txt", losses, delimiter=',')
    np.savetxt("./results/local_6_home_results_lre4_weighted_16/eva_accuracies.txt", accuracies, delimiter=',')
    np.savetxt("./results/local_6_home_results_lre4_weighted_16/eva_my_accuracies.txt", my_accuracies, delimiter=',')

    return lstm


def run_not_local():
    logdir = "./logs_not_local_2/"

    # num_clients = 9
    num_clients = 6
    # num_day_experiments = 30
    num_day_experiments = 40
    losses = np.zeros((num_day_experiments, num_clients))
    accuracies = np.zeros((num_day_experiments, num_clients))
    my_accuracies = np.zeros((num_day_experiments, num_clients))
    first_test_home = 1
    data_keys = 1

    np.random.seed(0)
    tf.random.set_seed(0)

    lstm = LSTM_train_test(output_size=settings.NUMBER_OF_ACTIVITIES)
    lstm.compile()

    for current_days in range(3, 43):
        print(f"Current Day: {current_days}")
        np.random.seed(0)
        tf.random.set_seed(0)

        if current_days in LIST_OF_DAYS_OF_UPDATES_NEW_DATA:
            data_keys = current_days

        selected_clients = []
        num_days = {}

        for i in AVAILABLE_CLIENTS_NEW_DATA[data_keys]:
            num_d = current_days - CLIENT_START_DAY_NEW_DATA[i] + 1
            if num_d > 0:
                selected_clients.append(i)
                num_days[i] = num_d

        if len(selected_clients) == 0:
            continue
        try:
            data_train, labels_train = create_centralized_train_dataset(selected_clients, num_days)
            # data_train = data_train[:100, ...]
            # labels_train = labels_train[:100, ...]
            # for i in range(500):
            #     print("Iteration " + str(i))
            #     train_history = lstm.train(
            #         data_train=data_train, labels_train=labels_train, batch_size=settings.BATCH_SIZE
            #     )
            # loss, accuracy = lstm.evaluate(data_train, labels_train)
            # print(loss)
            # print(accuracy)
            # exit()
        except Exception as e:
            # raise (e)
            print(e)
            break

        size = labels_train.shape[0]

        # train_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, f'train_{num_days}'))

        train_history = lstm.train(
            data_train=data_train,
            labels_train=labels_train,
            batch_size=settings.BATCH_SIZE,
            num_epochs=500
        )

        for client_id in range(first_test_home, num_clients + 1):
            data_test, labels_test = create_local_test_dataset(client_id)
            print("Evaluation on client " + str(client_id))
            loss, cat_accuracy = lstm.evaluate(data_test, labels_test)
            predicted = lstm.predict(data_test)
            print('results on some data points\n\n')
            print(tf.argmax(predicted[:5], axis=1), tf.argmax(labels_test[:5], axis=1))
            print('\n\n')
            my_accuracy = get_window_accuracy(predicted, labels_test)

            losses[current_days - 3, client_id - first_test_home] = loss
            accuracies[current_days - 3, client_id - first_test_home] = cat_accuracy
            my_accuracies[current_days - 3, client_id - first_test_home] = my_accuracy

        # with train_summary_writer.as_default():
        #     tf.summary.scalar('train loss', train_history.history['loss'][0], step=it)
        #     tf.summary.scalar('train categorical accuracy', train_history.history['categorical_accuracy'][0],
        #                       step=it)
        #     train_summary_writer.flush()

    os.makedirs('./results/not_local_6_homes_results_lre3_weighted_00/', exist_ok=True)
    np.savetxt("./results/not_local_6_homes_results_lre3_weighted_00/eva_losses.txt", losses, delimiter=',')
    np.savetxt("./results/not_local_6_homes_results_lre3_weighted_00/eva_accuracies.txt", accuracies, delimiter=',')
    np.savetxt("./results/not_local_6_homes_results_lre3_weighted_00/eva_my_accuracies.txt", my_accuracies, delimiter=',')

    return lstm


def run_confusion_matrix(lstm):
    for client_id in range(1, 7):
        data_test, labels_test = create_local_test_dataset(client_id)
        predicted = lstm.predict(data_test)
        confusion_matrix = get_confusion_matrix(predicted, labels_test)

        os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)
        csv_file_address = os.path.join(CONFUSION_MATRIX_DIR, str(client_id) + '.csv')
        with open(csv_file_address, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([''] + LIST_OF_MAPPED_2_ACTIVITIES)
            for row in range(confusion_matrix.shape[0]):
                csv_writer.writerow(np.concatenate(([LIST_OF_MAPPED_2_ACTIVITIES[row]], confusion_matrix[row])))
        print(confusion_matrix)


if __name__ == '__main__':
    # lstm = run_not_local()
    lstm = run_local_only()
    # lstm = LSTM_train_test(output_size=settings.NUMBER_OF_ACTIVITIES)
    # lstm.compile()
    # lstm.load_weights('/home/sharare/PycharmProjects/FederatedLearning_Caching/saved_model_fl_new_data_lre4/42')
    run_confusion_matrix(lstm)
