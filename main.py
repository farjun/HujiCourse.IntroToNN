from __future__ import absolute_import, division, print_function, unicode_literals
import exModels
from datetime import datetime
import numpy as np
import tensorflow as tf


def get_data(db, normalize=True):
    if db == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    else:
        return None

    if normalize:
        x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train), (x_test, y_test)


def create_data_sets(maxTrainSize: int = 0):
    (x_train, y_train), (x_test, y_test) = get_data('mnist')
    if maxTrainSize:
        chosenIndexes = np.random.choice(x_train.shape[0], maxTrainSize)
        # TODO the if below redundant since you enter iff differ from 0.
        x_train = x_train if maxTrainSize == 0 else x_train[chosenIndexes]
        y_train = y_train if maxTrainSize == 0 else y_train[chosenIndexes]

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    return test_ds, train_ds


def get_train_step(model: tf.keras.Model, loss_object):
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # TODO change to standard usage with training=True
            predictions = model(images, training=True)
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
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    return test_step, test_loss, test_accuracy


def getSummaryWriters(modelName):
    # tensor board
    # TODO not sure why to use different directories.
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'logs/{modelName}/' + current_time + '/train'
    print("run: tensorboard --logdir ./" + train_log_dir + " --port 6006")
    test_log_dir = f'logs/{modelName}/' + current_time + '/test'
    print("run: tensorboard --logdir ./" + test_log_dir + " --port 6007")

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    return train_summary_writer, test_summary_writer


def trainAndTest(model, train_ds, test_ds, graphs_suffix="", summary_writers=None):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_step, train_loss, train_accuracy = get_train_step(model, loss_object)
    test_step, test_loss, test_accuracy = get_test_step(model, loss_object)

    # TODO missing close for the writer, while passing summary_writers as optional is problmatic, just make it mandatory.
    train_summary_writer, test_summary_writer = getSummaryWriters(
        model.name) if summary_writers is None else summary_writers

    total_number_of_iteration = 20000
    report_every = 500
    train_counter = 0

    for epoch in range(report_every, total_number_of_iteration + report_every, report_every):  # TODO it's not epoch!
        for images, labels in train_ds:
            train_step(images, labels)
            train_counter += 1
            if train_counter % report_every == 0:
                break

        with train_summary_writer.as_default():
            tf.summary.scalar(graphs_suffix + "loss", train_loss.result(), step=train_counter)
            tf.summary.scalar(graphs_suffix + "accuracy", train_accuracy.result() * 100, step=train_counter)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        with test_summary_writer.as_default():
            tf.summary.scalar(graphs_suffix + "loss", test_loss.result(), step=train_counter)
            tf.summary.scalar(graphs_suffix + "accuracy", test_accuracy.result() * 100, step=train_counter)
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


def runQ4():
    model = exModels.ReducedOverfittingCNNModel()
    summary_writers = getSummaryWriters(model.name)
    test_ds, train_ds = create_data_sets()
    trainAndTest(model, train_ds, test_ds, graphs_suffix="no dropout full data:", summary_writers=summary_writers)

    model = exModels.ReducedOverfittingCNNModel()
    test_ds, train_ds = create_data_sets(maxTrainSize=250)
    trainAndTest(model, train_ds, test_ds, graphs_suffix="no dropout partial data:", summary_writers=summary_writers)

    model = exModels.ReducedOverfittingCNNModel(dropoutRate=0.3)
    test_ds, train_ds = create_data_sets()
    trainAndTest(model, train_ds, test_ds, graphs_suffix="with dropout full data:", summary_writers=summary_writers)

    model = exModels.ReducedOverfittingCNNModel(dropoutRate=0.3)
    test_ds, train_ds = create_data_sets(maxTrainSize=250)
    trainAndTest(model, train_ds, test_ds, graphs_suffix="with dropout partial data:", summary_writers=summary_writers)
    # TODO missing close for the writes.


def main():
    runQ4()
    # model = exModels.CNNModel()
    # model = exModels.LinearModel()
    # model = exModels.SmallCNN()
    # model = exModels.ReducedCNNModel()
    # test_ds, train_ds = create_data_sets()
    # trainAndTest(model,train_ds, test_ds)


if __name__ == '__main__':
    main()
