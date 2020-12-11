import csv
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error, mean_squared_error, r2_score

import numpy as np
from settings import CLIENT_START_DAY


def find_days(y_test):
    num_clients = y_test.shape[0] // 45
    y_test_days = np.ones((num_clients, 1)) * 45

    for i in range(num_clients):
        for j in range(i * 45, (i + 1) * 45):
            if y_test[j] == 0:
                y_test_days[i] = j % 45
                break

    return y_test_days


data_address = "/home/sharare/PycharmProjects/FederatedLearning_Caching/datasets/classification_week.csv"
with open(data_address) as csv_file:
    data = csv.reader(csv_file, delimiter=',')
    data = np.array(list(data))

data = data[1:, 1:]
data_w_day = np.zeros((45 * 30, data.shape[1] + 2))
# import pdb
# pdb.set_trace()
for i in range(30 * 45):
    data_w_day[i, :-3] = data[i // 45, :-1].astype(np.float32)
    data_w_day[i, -3] = (i % 45)
    data_w_day[i, -2] = CLIENT_START_DAY[101 + i // 45]
    data_w_day[i, -1] = 1 if (i % 45) < int(data[i // 45, -1]) else 0
# X = data[1:, 1:-1].astype(np.float32)
# X_scaled = preprocessing.minmax_scale(X)
# y = data[1:, -1].astype(np.float32)

X = data_w_day[:, :-1].astype(np.float32)
X_scaled = X / 100
y = data_w_day[:, -1].astype(np.float32)

# class_weight={1.00: 1, 3.00: 3},
# labels = [1.0, 2.0]
# class_weight = {1.0: 11, 2.00: 19}
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=3),
    'SVM': SVC(random_state=3),
    'Nearest Neighbors': KNeighborsClassifier(n_neighbors=2),
    'Random Forest': RandomForestClassifier(n_estimators=10, random_state=3)
}
# models = {
#     'Decision Tree': DecisionTreeRegressor(),
#     'SVM': SVR(C=0.9, epsilon=0.1),
#     'Nearest Neighbors': KNeighborsRegressor(n_neighbors=3),
#     # 'Random Forest': RandomForestRegressor(n_estimators=10),
#     # 'Logistic': LogisticRegression(max_iter=500)
# }
kf = KFold(n_splits=5, shuffle=False, random_state=3)
for model_name, model in models.items():
    train_accs = []
    accs = []
    cls_reports = []
    for train_indices, test_indices in kf.split(data):
        # X_train, X_test = X[train_indices, :], X[test_indices, :]
        # y_train, y_test = y[train_indices], y[test_indices]

        new_train_indices = []
        new_test_indices = []

        for index in train_indices:
            for counter in range(45):
                new_train_indices.append(index * 45 + counter)
        new_train_indices = np.stack(new_train_indices)
        for index in test_indices:
            for counter in range(45):
                new_test_indices.append(index * 45 + counter)
        new_test_indices = np.stack(new_test_indices)

        train_indices = new_train_indices
        test_indices = new_test_indices

        X_train, X_test = X_scaled[train_indices, :], X_scaled[test_indices, :]
        y_train, y_test = y[train_indices], y[test_indices]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        # print('preds')
        # print(predictions)
        # print('labels')
        # print(list(map(int, y_test)))

        y_test_days = find_days(y_test)
        pred_days = find_days(predictions)

        # print(y_test_days)
        # print(pred_days)

        # print(predictions)
        # print(y_test.reshape((1, -1)))
        # print(model.score(X_test, y_test))
        train_accs.append(model.score(X_train, y_train))
        accs.append(model.score(X_test, y_test))
        cls_report = precision_recall_fscore_support(y_test, predictions)
        # diff = np.abs(y_test - predictions)
        # correct_count = np.count_nonzero(diff <= 3)
        # acc = correct_count / y_test.shape[0]
        # mae = mean_absolute_error(y_test, predictions)
        # mse = mean_squared_error(y_test, predictions)
        # r2 = r2_score(y_test, predictions)
        # cls_report = cls_report.reshape((-1, 1))
        cls_reports.append(cls_report)
        # cls_reports.append([mae, mse, r2, acc])
        # print(cls_report)

    cls_reports = np.stack(cls_reports)
    with np.printoptions(precision=3, suppress=True):
        print(model_name)
        print('Train Acc:')
        print(np.mean(train_accs))
        print('Acc:')
        print(np.mean(accs))
        print('Std:')
        print(np.std(accs))
        print('Classification Report')
        print('Precision\nRecall\nF1-Score\nSupport')
        print(np.mean(cls_reports, axis=0))

        print('======================================')
