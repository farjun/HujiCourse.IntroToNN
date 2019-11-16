from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import datetime

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D


class CNNModel(Model):


    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = Conv2D(32, (5, 5), activation='relu')
        self.max_pool_1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (5, 5), activation='relu')
        self.max_pool_2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


class LinearModel(Model):


    def __init__(self):
        super(LinearModel, self).__init__()
        self.max_pool_1 = MaxPooling2D((2, 2))
        self.max_pool_2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.max_pool_1(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

class SmallCNN(Model):

    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = Conv2D(32, (5, 5), activation='relu')
        self.max_pool_1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (5, 5), activation='relu')
        self.max_pool_2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')
        self.d2 = Dense(10, activation='softmax')


    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


def get_data(db, normalize=True):
    if db == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    else:
        return None

    if normalize:
        x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train), (x_test, y_test)


def main():
    # model = CNNModel()
    model = LinearModel()
    test_ds, train_ds = create_data_sets()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_step, train_loss, train_accuracy = get_train_step(model, loss_object)
    test_step, test_loss, test_accuracy = get_test_step(model, loss_object)

    # tensor board
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'logs/{model.name}/' + current_time + '/train'
    test_log_dir = f'logs/{model.name}/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    total_number_of_iteration = 20000
    report_every = 500
    train_counter = 0

    for epoch in range(report_every, total_number_of_iteration+report_every, report_every): # TODO it's not epoch!
        for images, labels in train_ds:
            train_step(images, labels)
            train_counter += 1
            if train_counter % report_every == 0:
                break
        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=train_counter)
            tf.summary.scalar("accuracy", train_accuracy.result() * 100, step=train_counter)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        with test_summary_writer.as_default():
            tf.summary.scalar("loss", test_loss.result(), step=train_counter)
            tf.summary.scalar("accuracy", test_accuracy.result() * 100, step=train_counter)
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    # train_summary_writer.close()
    # test_summary_writer.close()


def get_train_step(model, loss_object):
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    return train_step, train_loss, train_accuracy


def get_test_step(model, loss_object):
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    return test_step, test_loss, test_accuracy


def create_data_sets():
    (x_train, y_train), (x_test, y_test) = get_data('mnist')
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    return test_ds, train_ds


if __name__ == '__main__':
    main()
