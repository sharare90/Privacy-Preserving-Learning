from tqdm import tqdm
from datetime import datetime

from settings import MAP_OF_ACTIVITIES_ADLS


for i in tqdm(range(1, 7)):

    with open('/home/sharare/PycharmProjects/FederatedLearning_Caching/datasets/adls/' + str(i) + '.csv') as casas_file, \
            open('/home/sharare/PycharmProjects/FederatedLearning_Caching/datasets/adls/mapped/' + str(i) + '.csv',
                 'w') as processed_file:
        casas_file.readline()
        for line in casas_file:
            line_arr = line.split(",")
            start_date_time = line_arr[3]
            end_date_time = line_arr[4]
            s_date_time_obj = datetime.strptime(start_date_time, '%d.%m.%y %H:%M:%S')
            e_date_time_obj = datetime.strptime(end_date_time, '%d.%m.%y %H:%M:%S')
            hour = s_date_time_obj.hour
            second = s_date_time_obj.second
            day_of_week = s_date_time_obj.weekday()
            FMT = '%H:%M:%S'
            window = datetime.strptime(end_date_time.split()[1], FMT) - datetime.strptime(
                start_date_time.split()[1], FMT)
            window_duration = window.seconds
            activity = line_arr[5]
            activity = activity.replace('\n', '')

            processed_file.write(
                str(hour) + "," + str(second) + "," + str(day_of_week) + "," + str(window_duration) + "," +
                MAP_OF_ACTIVITIES_ADLS[activity] + "\n")
