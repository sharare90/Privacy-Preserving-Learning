import tensorflow as tf
import numpy as np
from datasets import Dataset
from settings import NUMBER_OF_ACTIVITIES, HISTORY_SIZE, TOTAL_NUM_OF_FEATURES

np.random.seed(0)
tf.random.set_seed(0)


class LSTM_train_test(object):
    def __init__(self, output_size):
        # TODO bring these hyperparameters to to __init__ arguments
        self.rnn_units = 256
        self.learning_rate = 1e-3

        self.output_size = output_size

        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                self.rnn_units,
                activation='relu',
                return_sequences=False,
                unroll=True,
                use_bias=True,
                # activity_regularizer=tf.keras.regularizers.l1_l2(0.01),
                # recurrent_initializer='he_normal',
                stateful=False,
                input_shape=(HISTORY_SIZE, TOTAL_NUM_OF_FEATURES)
            ),
            tf.keras.layers.Dense(256),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(self.output_size),
            tf.keras.layers.Activation("softmax")
        ])

    def compile(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                           metrics=[tf.keras.metrics.CategoricalAccuracy()])
        # tensorboard_call_back = tf.keras.callbacks.TensorBoard(log_dir='logs_non_fl_tr_b8_lr001')

    def train(self, data_train, labels_train, batch_size, num_epochs=1):
        earlystopping_call_back = tf.keras.callbacks.EarlyStopping(
            monitor='val_categorical_accuracy',
            min_delta=0.01,
            patience=50,
            mode='max',
            restore_best_weights=True
        )

        return self.model.fit(
            x=data_train,
            y=labels_train,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[earlystopping_call_back],
            verbose=1
        )

    def evaluate(self, data_test, labels_test):
        return self.model.evaluate(data_test, labels_test)

    def predict(self, data_test):
        return self.model.predict(data_test)
