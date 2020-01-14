import numpy as np
import tensorflow as tf
from datetime import datetime
import models as exModels
from tqdm.auto import tqdm

def get_data(normalize=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if normalize:
        x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train, x_test = x_train[..., tf.newaxis], x_test[..., tf.newaxis]

    return (x_train, y_train), (x_test, y_test)



def get_data_as_tensorslice():
    (x_train, y_train), (x_test, y_test) = get_data()

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    return test_ds, train_ds


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

def get_train_step(generator: tf.keras.Model, loss_object):
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = generator(images)
            loss = loss_object(images, predictions)
        gradients = tape.gradient(loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

        train_loss(loss)
        train_accuracy(images, predictions)

    return train_step, train_loss, train_accuracy

def trainEncoder(generator, train_ds,epochs = 40):
    loss_object = tf.keras.losses.MeanSquaredError()
    train_step, train_loss, train_accuracy = get_train_step(generator, loss_object)
    train_summary_writer, test_summary_writer = getSummaryWriters(generator.name)
    train_counter = 0
    for epoch in tqdm(range(epochs)):
        for images, labels in train_ds:
            train_step(images, labels)
            train_counter += 1

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=train_counter)
            tf.summary.scalar("accuracy", train_accuracy.result() * 100, step=train_counter)

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()


    generator.save_weights("./weights/v1")
    train_summary_writer.close()
    test_summary_writer.close()


def Q1():
    generator = exModels.CNNGenerator()
    test_ds, train_ds = get_data_as_tensorslice()
    exModels.printable_model(generator).summary()
    trainEncoder(generator, train_ds)

def main():
    Q1()

if __name__ == '__main__':
    main()