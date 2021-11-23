import os

federated_results = '/home/sharare/PycharmProjects/FederatedLearning_Caching/results/fl_90_homes_results'


with open('/home/sharare/PycharmProjects/FederatedLearning_Caching/scripts/updatedlogs/fed_acc.txt', 'w') as fed_txt:
    for day in range(5, 78):
        day_result_path = os.path.join(federated_results, f'results_day_{day}_old.txt')

        if not os.path.exists(day_result_path):
            day_result_path = os.path.join(federated_results, f'results_day_{day}.txt')

        with open(day_result_path) as day_results:
            fed_txt.write(day_results.readline())
