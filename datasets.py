import csv
import numpy as np

import settings
from settings import SAMPLING, MAPPED_ACT2IDX, NUMBER_OF_ACTIVITIES, NUMBER_OF_HOURS, NUMBER_OF_DAYS, USE_TIME, USE_DAY, \
    ACT2IDX, MAPPED_2_ACT2IDX


class Dataset(object):
    def __init__(self, dataset_address):
        # TODO gets a csv file
        with open(dataset_address) as data_file:
            csv_data = csv.reader(data_file)
            self.data = np.array(list(csv_data))

    def get_seq_to_seq_data(self):
        pass

    def sample_data(self, freq):
        # TODO return data when sampled by frequency freq, for example
        # TODO [1, 2, 3, 4, 5, 6, 7, 8, 9] with freq 4 should be [1, 5, 7]
        return self.data[::freq]

    def get_not_repeated_activity_data(self):
        """
        self.data is N x 5 matrix.
        Columns correspond to time of the day (hour), seconds of the day, day of the week, action_window which is
        ignored and activity.

        ...activity 1 ...
        ...activity 1 ...
        ...activity 2 ...
        ...activity 2 ...
        ...activity 3 ...
        ...activity 2 ...
        ...activity 2 ...
        ...activity 2 ...

        will be turned into
        ...activity 1 ...
        ...activity 2 ...
        ...activity 3 ...
        ...activity 2 ...
        :return:
        np.ndarray of size M x 5.
        """
        latest_activity = self.data[0, -1]
        not_repeated = [self.data[0, :]]
        for i, line in enumerate(self.data):
            current_activity = list(line)[-1]
            if latest_activity != current_activity:
                not_repeated.append(line)
                latest_activity = current_activity
        not_repeated = np.stack(not_repeated)
        return not_repeated


def get_data_of_days(data, num_days):
    # TODO return the data of num_days from line = 0 to end_index
    """
    picks the first num_days of data.
    Data is N x 5 matrix
    :param data:
    :param num_days:
    :return:
    """
    previous_time = data[0, 0]
    count_for_days = 0
    for i, line in enumerate(data):
        if count_for_days < num_days:
            current_time = list(line)[0]
            if float(current_time) < float(previous_time):
                count_for_days += 1
            previous_time = current_time
        else:
            return i - 1
    return len(data)


def get_dataset_from_file(number):
    if settings.MAPPED:
        dataset_address = '/home/sharare/PycharmProjects/FederatedLearning_Caching/datasets/casas/mapped_2_csvs/csh'
    else:
        dataset_address = '/home/sharare/PycharmProjects/FederatedLearning_Caching/datasets/casas/processed_csvs/csh'

    dataset_address = dataset_address + str(number) + '.csv'
    dataset = Dataset(dataset_address)
    return dataset


def convert_to_one_hot(not_repeated):
    if settings.MAPPED:
        activities_index = [MAPPED_2_ACT2IDX[act] for act in not_repeated[:, -1]]
    else:
        activities_index = [ACT2IDX[act] for act in not_repeated[:, -1]]

    one_hot = np.eye(NUMBER_OF_ACTIVITIES)[activities_index]
    if USE_TIME:
        time_index = [int(float(t)) for t in not_repeated[:, 0]]
        one_hot_time = np.eye(NUMBER_OF_HOURS)[time_index]
        one_hot = np.concatenate((one_hot_time, one_hot), axis=1)
    if USE_DAY:
        day_index = [int(float(d)) for d in not_repeated[:, 2]]
        one_hot_day = np.eye(NUMBER_OF_DAYS)[day_index]
        one_hot = np.concatenate((one_hot_day, one_hot), axis=1)
    return one_hot


if __name__ == '__main__':
    dataset = Dataset('/home/sharare/PycharmProjects/FederatedLearning_Caching/datasets/casas/mapped_csvs/csh101.csv')
    activites = dataset.get_activities()
    print(activites.shape)
    print(activites)
    # print(dataset.sample_data(2))
