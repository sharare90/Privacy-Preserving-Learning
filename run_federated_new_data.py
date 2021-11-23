import numpy as np
import h5py
import random
import os

import tensorflow as tf
import tensorflow_federated as tff
from datasets_adls import get_dataset_from_file, convert_to_one_hot, get_data_of_days
from run_local_and_centralized_new_data import get_window_accuracy_on_dataset
from utils import build_dataset
from settings import NUMBER_OF_ACTIVITIES, NUM_ROUNDS, BATCH_SIZE, BUFFER_SIZE, USE_DAY, USE_TIME, \
    HISTORY_SIZE, TOTAL_NUM_OF_FEATURES, LIST_OF_DAYS_OF_UPDATES_NEW_DATA, AVAILABLE_CLIENTS_NEW_DATA, \
    CLIENT_START_DAY_NEW_DATA, LOSS_WEIGHTS_ADLS

np.random.seed(0)
tf.random.set_seed(0)


def create_h5_data(number):
    np_data = get_dataset_from_file(number)
    activities_d = np_data.get_activities()
    hf = h5py.File('/home/sharare/PycharmProjects/FederatedLearning_Caching/datasets/casas/h5/' + str(number), 'w')
    hf.create_dataset('/home/sharare/PycharmProjects/FederatedLearning_Caching/datasets/casas/h5/' + str(number),
                      data=activities_d)
    hf.close()


def fl_train_dataset_for_client(client, num_days):
    dataset_d = get_dataset_from_file(client)
    # activities_d = dataset_d.get_activities()[client_round_dict[client] * history_size:
    #                                           (client_round_dict[client] + 2) * history_size, :]

    not_repeated = dataset_d.get_not_repeated_activity_data()
    d = convert_to_one_hot(not_repeated)
    train_day_index = get_data_of_days(not_repeated, num_days)
    limit_index = int(0.7 * len(d))
    if limit_index >= train_day_index:
        train_d = d[:train_day_index, :]
    else:
        train_d = d[:limit_index, :]
    data_d, labels_d = build_dataset(train_d)
    return data_d, labels_d


def fl_test_dataset_for_client(client, split_point=0.7):
    dataset_d = get_dataset_from_file(client)
    # activities_d = dataset_d.get_activities()[client_round_dict[client] * history_size:
    #                                           (client_round_dict[client] + 2) * history_size, :]

    not_repeated = dataset_d.get_not_repeated_activity_data()
    d = convert_to_one_hot(not_repeated)
    test_start_index = int(split_point * len(d))
    test_d = d[test_start_index:, :]
    data_d, labels_d = build_dataset(test_d)
    return data_d, labels_d


def sample_train_clients(num):
    return random.sample(range(101, 125), num)


def create_keras_model():
    return tf.keras.Sequential([

        tf.keras.layers.LSTM(
            256,
            activation='relu',
            return_sequences=False,
            unroll=True,
            use_bias=True,
            # activity_regularizer=tf.keras.regularizers.l1_l2(0.01),
            recurrent_initializer='he_normal',
            stateful=False,
            input_shape=(HISTORY_SIZE, TOTAL_NUM_OF_FEATURES)
        ),
        tf.keras.layers.Dense(256),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(NUMBER_OF_ACTIVITIES),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("softmax")
    ])


def load_model(batch_size):
    urls = {
        1: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch1.kerasmodel',
        8: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch8.kerasmodel'}
    assert batch_size in urls, 'batch_size must be in ' + str(urls.keys())
    url = urls[batch_size]
    local_file = tf.keras.utils.get_file(os.path.basename(url), origin=url)
    return tf.keras.models.load_model(local_file, compile=False)


def build_meta_fed_train_dataset(current_days, data_keys):
    def meta_learning_input(x, y):
        return x[0, ...], x[1, ...], y[0, ...], y[1, ...]

    meta_train_datasets = []
    for i in AVAILABLE_CLIENTS_NEW_DATA[data_keys]:
        num_days = current_days - CLIENT_START_DAY_NEW_DATA[i] + 1
        if num_days > 0:
            data, labels = fl_train_dataset_for_client(i, num_days)
            data_for_this_round = tf.data.Dataset.from_tensor_slices((data, labels))
            meta_data_batch = data_for_this_round.batch(2, drop_remainder=True)
            meta_data = meta_data_batch.map(meta_learning_input).shuffle(BUFFER_SIZE).batch(
                BATCH_SIZE, drop_remainder=True)
            meta_train_datasets.append(meta_data)
    return meta_train_datasets


def build_fed_train_dataset(current_days, data_keys):
    train_datasets = []
    for i in AVAILABLE_CLIENTS_NEW_DATA[data_keys]:
        num_days = current_days - CLIENT_START_DAY_NEW_DATA[i] + 1
        if num_days > 0:
            data, labels = fl_train_dataset_for_client(i, num_days)
            data_for_this_round = tf.data.Dataset.from_tensor_slices((data, labels))
            batch_data_for_this_round = data_for_this_round.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

            element_spec = batch_data_for_this_round.element_spec
            # print(element_spec)
            train_datasets.append(batch_data_for_this_round)
    return train_datasets


def build_fed_test_dataset():
    test_datasets = []
    for i in range(1, 7):
        data, labels = fl_test_dataset_for_client(i)
        data_for_this_round = tf.data.Dataset.from_tensor_slices((data, labels))
        batch_data_for_this_round = data_for_this_round.batch(1)
        element_spec = batch_data_for_this_round.element_spec
        # print(element_spec)
        test_datasets.append(batch_data_for_this_round)
    return test_datasets, element_spec


keras_model = create_keras_model()
# keras_model.load_weights("/home/sharare/PycharmProjects/FederatedLearning_Caching/saved_model/76")
test_datasets, element_spec = build_fed_test_dataset()


# keras_model = load_model(batch_size=BATCH_SIZE)


def create_tff_model():
    # TFF uses an `input_spec` so it knows the types and shapes
    # that your model expects.
    input_spec = element_spec
    keras_model_clone = tf.keras.models.clone_model(keras_model)

    return tff.learning.from_keras_model(
        keras_model_clone,
        input_spec=input_spec,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])


def initialize_fn():
    model = create_tff_model()
    return model.weights.trainable


def next_fn(server_weights, federated_dataset):
    # Broadcast the server weights to the clients.
    server_weights_at_client = tff.federated_broadcast(server_weights)

    # Each client computes their updated weights.
    client_weights = client_update(federated_dataset, server_weights_at_client)

    # The server averages these updates.
    mean_client_weights = np.mean(client_weights)

    # The server updates its model.
    server_weights = server_update(mean_client_weights)

    return server_weights


@tf.function
def client_update(model, dataset, server_weights, client_optimizer):
    """Performs training (using the server model weights) on the client's dataset."""
    # Initialize the client model with the current server weights.
    client_weights = model.weights.trainable
    # Assign the server weights to the client model.
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          client_weights, server_weights)

    # Use the client_optimizer to update the local model.
    for batch in dataset:
        with tf.GradientTape() as tape:
            # Compute a forward pass on the batch of data
            outputs = model.forward_pass(batch)

        # Compute the corresponding gradient
        grads = tape.gradient(outputs.loss, client_weights)
        grads_and_vars = zip(grads, client_weights)

        # Your code here: Apply the gradient using a client_optimizer.
        client_optimizer.apply_gradients(grads_and_vars)
    return client_weights


@tf.function
def server_update(model, mean_client_weights):
    """Updates the server model weights as the average of the client model weights."""
    model_weights = model.weights.trainable
    # Assign the mean client weights to the server model.
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          model_weights, mean_client_weights)
    return model_weights


trainer = tff.learning.build_federated_averaging_process(
    model_fn=create_tff_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(lr=0.001),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=0.5)
)

# trainer = tff.learning.build_federated_sgd_process(
#     model_fn=create_tff_model,
#     server_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=0.5)
# )

# tff.learning.build_federated_sgd_process()
# tff.learning.build_personalization_eval()
state = trainer.initialize()

logdir = "./logs_fl_wd_avg_sampling_new_data_d1/"
train_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
# evaluate_summary_writer = tf.summary.create_file_writer(os.path.join(logdir, 'evaluate'))
client_evaluate_summary_writers = [
    tf.summary.create_file_writer(os.path.join(logdir, 'evaluate' + str(i))) for i in range(len(test_datasets))
]


def keras_evaluate(state, round_num):
    # Take our global model weights and push them back into a Keras model to
    # use its standard `.evaluate()` method.
    clients_losses = []
    clients_accuracies = []
    window_accuracies = []
    keras_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])
    tff.learning.assign_weights_to_keras_model(keras_model, state.model)
    for i in range(len(test_datasets)):
        loss, accuracy = keras_model.evaluate(test_datasets[i], verbose=0)
        window_acc = get_window_accuracy_on_dataset(keras_model, test_datasets[i])

        print('\tEval {data:2d}: loss={l:.3f}, accuracy={a:.3f}'.format(data=i, l=loss, a=accuracy))
        with client_evaluate_summary_writers[i].as_default():
            tf.summary.scalar('evaluation loss', loss, step=round_num)
            tf.summary.scalar('evaluation categorical accuracy', accuracy, step=round_num)
        clients_losses.append(loss)
        clients_accuracies.append(accuracy)
        window_accuracies.append(window_acc)
    return clients_losses, clients_accuracies, window_accuracies, keras_model


losses = []
accuracies = []
my_accuracies = []
data_keys = 2
os.makedirs('./results/fl_6_homes_results_lre4_weighted/', exist_ok=True)
with train_summary_writer.as_default():
    for current_days in range(3, 43):
        if current_days in LIST_OF_DAYS_OF_UPDATES_NEW_DATA:
            data_keys = current_days
        build_meta_fed_train_dataset(current_days, data_keys)
        train_datasets = build_fed_train_dataset(current_days, data_keys)
        train_datasets = [
            tf_dataset for tf_dataset in train_datasets if tf.data.experimental.cardinality(tf_dataset) != 0
        ]
        if len(train_datasets) == 0:
            continue
        print(f"Current day: {current_days}")
        for round_num in range(NUM_ROUNDS):
            print('Round {r}'.format(r=round_num))
            client_losses, client_accuracies, window_accuracies, model = keras_evaluate(state, round_num)
            # sampled_clients = random.sample(train_datasets, 21)
            state, metrics = trainer.next(state, train_datasets)
            print('\tTrain: round {:2d}, metrics={}'.format(round_num, metrics))
            for name, value in metrics._asdict().items():
                tf.summary.scalar('train ' + name, value, step=round_num)
        losses.append(client_losses)
        accuracies.append(client_accuracies)
        my_accuracies.append(window_accuracies)

        model.save_weights(
            '/home/sharare/PycharmProjects/FederatedLearning_Caching/saved_model_fl_new_data_lre4_weighted/' + f'{current_days}')
        current_day_results = np.stack([client_accuracies, window_accuracies, client_losses], axis=0)
        np.savetxt(f"./results/fl_6_homes_results_lre4_weighted/results_day_{current_days}.txt",
                   current_day_results,
                   delimiter=',')

np.savetxt("./results/fl_6_homes_results_lre4_weighted/eva_losses.txt", losses, delimiter=',')
np.savetxt("./results/fl_6_homes_results_lre4_weighted/eva_accuracies.txt", accuracies, delimiter=',')
np.savetxt("./results/fl_6_homes_results_lre4_weighted/window_accuracies.txt", my_accuracies, delimiter=',')

keras_evaluate(state, NUM_ROUNDS + 1)
