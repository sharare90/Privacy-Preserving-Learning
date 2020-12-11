from tqdm import tqdm

from settings import MAP_OF_ACTIVITIES

# for i in tqdm(range(101, 131)):
#
#     with open('./datasets/casas/processed_csvs/csh' + str(i) + '.csv') as casas_file, \
#             open('./datasets/casas/mapped_csvs/csh' + str(i) + '.csv', 'w') as processed_file:
#         for line in casas_file:
#             line_arr = line.split(",")
#             hour = line_arr[0]
#             second = line_arr[1]
#             day_of_week = line_arr[2]
#             window_duration = line_arr[3]
#             activity = line_arr[4]
#             activity = activity.replace('\n', '')
#             if activity != "NAN" and activity not in ['Step_Out', 'Other_Activity', 'Bed_Toilet_Transition']:
#                 processed_file.write(
#                     hour + "," + second + "," + day_of_week + "," + window_duration + "," +
#                     MAP_OF_ACTIVITIES[activity] + "\n")

for i in tqdm(range(101, 131)):

    with open('./datasets/casas/mapped_csvs/csh' + str(i) + '.csv') as casas_file, \
            open('./datasets/casas/mapped_2_csvs/csh' + str(i) + '.csv', 'w') as processed_file:
        for line in casas_file:
            line_arr = line.split(",")
            hour = line_arr[0]
            second = line_arr[1]
            day_of_week = line_arr[2]
            window_duration = line_arr[3]
            activity = line_arr[4]
            activity = activity.replace('\n', '')
            if activity in ['COOK','WASH_DISHES','HEALTH', 'GROOMING', 'PERSONAL_HYGIENE']:
                processed_file.write(
                    hour + "," + second + "," + day_of_week + "," + window_duration + "," +
                    MAP_OF_ACTIVITIES[activity] + "\n")
            else:
                processed_file.write(
                    hour + "," + second + "," + day_of_week + "," + window_duration + "," +
                    activity + "\n")