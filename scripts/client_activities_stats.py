from settings import LIST_OF_MAPPED_2_ACTIVITIES, NUMBER_OF_ACTIVITIES
from datasets import get_dataset_from_file, convert_to_one_hot, get_data_of_days


def client_activities_stats(client_activities):
    for i, act in enumerate(LIST_OF_MAPPED_2_ACTIVITIES):
        print(act + "," + str(int(sum(client_activities[:, i]))))


if __name__ == "__main__":
    for i in range(121, 122):
        dataset_d = get_dataset_from_file(i)
        not_repeated = dataset_d.get_not_repeated_activity_data()
        one_hot = convert_to_one_hot(not_repeated)
        train_day_index = get_data_of_days(not_repeated, 2)
        activities_d = one_hot[:train_day_index, -NUMBER_OF_ACTIVITIES:]
        print("\n\nClient" + str(i) + "\n")
        client_activities_stats(activities_d)
