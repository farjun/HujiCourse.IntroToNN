from typing import Dict

import numpy as np
import tensorflow as tf
from datetime import datetime
import models as exModels
from tqdm.auto import tqdm
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# WEIGHTS_PATH = "./weights/AE/v1"
BATCH_SIZE = 32
WEIGHTS_PATH = "./weights/v1"
DenoisingAE_WEIGHTS_PATH = "./weights/DenoisingAE/v1"
GanGenerator_WEIGHTS_PATH = "./weights/GanGenerator/v1"
GanDiscriminator_WEIGHTS_PATH = "./weights/GanDiscriminator/v1"


def get_data(normalize=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if normalize:
        x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train, x_test = x_train[..., tf.newaxis], x_test[..., tf.newaxis]

    return (x_train, y_train), (x_test, y_test)


def get_data_as_tensorslice(normalize=True):
    (x_train, y_train), (x_test, y_test) = get_data(normalize)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)
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
    # TODO train_accuracy seems to be wrong maybe not , I'm not sure...
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

def train_AE(generator, train_ds, epochs=40, save_img_every=100, weights_path=WEIGHTS_PATH):
    loss_object = tf.keras.losses.MeanSquaredError()
    train_step, train_loss, train_accuracy = get_train_step(generator, loss_object)
    train_summary_writer, test_summary_writer = getSummaryWriters(generator.name)
    train_counter = 0
    for epoch in tqdm(range(epochs)):
        for images, labels in train_ds:
            train_step(images, labels)
            train_counter += 1
            if train_counter % save_img_every == 0:
                with train_summary_writer.as_default():
                    image_input = images
                    common = {
                        "step": train_counter,
                        "max_outputs": 3
                    }
                    tf.summary.image(
                        "generator_img",
                        generator(image_input),
                        **common
                    )
                    tf.summary.image(
                        "src_img",
                        image_input,
                        **common
                    )

            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss.result(), step=train_counter)
                tf.summary.scalar("accuracy", train_accuracy.result() * 100, step=train_counter)

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

    generator.save_weights(weights_path)
    train_summary_writer.close()
    test_summary_writer.close()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def get_gan_train_step(generator: tf.keras.Model, discriminator: tf.keras.Model, generator_loss_object, discriminator_loss_object ):
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    generator_train_loss = tf.keras.metrics.Mean(name='train_loss')
    discriminator_train_loss = tf.keras.metrics.Mean(name='train_loss')
    generator_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    discriminator_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(images, labels):
        noise = tf.random.normal((BATCH_SIZE, 1))
        with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:
            fake_im = generator(noise)

            real_im_output = discriminator(images)
            fake_im_output = discriminator(fake_im)

            generator_loss = generator_loss_object(fake_im_output)
            discriminator_loss = discriminator_loss_object(real_im_output, fake_im_output)

        generator_gradients = gen_tape.gradient(generator_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        generator_train_loss(generator_loss)
        discriminator_train_loss(discriminator_loss)

        generator_train_accuracy(real_im_output, tf.zeros_like(real_im_output))
        discriminator_train_accuracy(real_im_output, tf.ones_like(real_im_output))

    return train_step, generator_train_loss, discriminator_train_loss, generator_train_accuracy, discriminator_train_accuracy

def train_GAN(generator, discriminator, train_ds, epochs=40, save_img_every=100, gen_weights_path=WEIGHTS_PATH, disc_weights_path=WEIGHTS_PATH):
    train_step, gen_train_loss, disc_train_loss, gen_train_accuracy, disc_train_accuracy = get_gan_train_step(generator, discriminator, generator_loss, discriminator_loss)
    gen_train_summary_writer, t1 = getSummaryWriters(generator.name)
    disc_train_summary_writer, t2 = getSummaryWriters(discriminator.name)
    train_counter = 0
    for epoch in tqdm(range(epochs)):
        for images, labels in train_ds:
            train_step(images, labels)
            train_counter += 1
            if train_counter % save_img_every == 0:
                with gen_train_summary_writer.as_default():
                    tf.summary.scalar("gen - loss", gen_train_loss.result(), step=train_counter)
                    tf.summary.scalar("gen - accuracy", gen_train_accuracy.result() * 100, step=train_counter)

                with disc_train_summary_writer.as_default():
                    tf.summary.scalar("disc - loss", disc_train_loss.result(), step=train_counter)
                    tf.summary.scalar("disc - accuracy", disc_train_accuracy.result() * 100, step=train_counter)

        # Reset the metrics for the next epoch
        gen_train_loss.reset_states()
        disc_train_loss.reset_states()
        gen_train_accuracy.reset_states()
        disc_train_accuracy.reset_states()

    generator.save_weights(gen_weights_path)
    discriminator.save_weights(disc_weights_path)
    gen_train_summary_writer.close()
    gen_train_summary_writer.close()
    t1.close()
    t2.close()

def Q1(epochs=10, save_img_every=100):
    generator = exModels.CNNGenerator()
    test_ds, train_ds = get_data_as_tensorslice()
    exModels.printable_model(generator).summary()
    train_AE(generator, train_ds, epochs, save_img_every)
    visual_latent_space(generator, test_ds)


def visual_latent_space_from_save(weights_path=WEIGHTS_PATH):
    test_ds, train_ds = get_data_as_tensorslice()
    generator = exModels.CNNGenerator()
    generator.load_weights(weights_path)
    visual_latent_space(generator, test_ds)


def visual_latent_space(generator: exModels.CNNGenerator, test_ds):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA, LatentDirichletAllocation
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    import matplotlib.pyplot as plt
    tsne = TSNE(n_components=2)
    pca = PCA(n_components=2)
    lda = LinearDiscriminantAnalysis(n_components=2)
    reducer = tsne
    max_iters = 100
    iter_count = 0
    Xs = None
    Y = None
    for images, labels in test_ds:
        if iter_count % max_iters == max_iters - 1:
            break
        else:
            iter_count += 1
        imgs = generator.encode(images)
        imgs_numpy = imgs.numpy()
        labels_numpy = labels.numpy()
        if Xs is None and Y is None:
            Xs = imgs_numpy
            Y = labels_numpy
        else:
            Xs = np.vstack((Xs, imgs_numpy))
            Y = np.concatenate((Y, labels_numpy))
    vis_x, vis_y = reducer.fit_transform(Xs, Y).T
    for i in range(10):
        where = Y == i
        x_i = vis_x[where]
        y_i = vis_y[where]
        plt.plot(x_i, y_i, ".", label=str(i), markersize=14)
    plt.legend(np.arange(0, 10))
    plt.show()


def Q2(epochs=10, save_img_every=100, noise_attributes: Dict = None):
    generator = exModels.DenoisingAE(noise_attributes)
    test_ds, train_ds = get_data_as_tensorslice()
    exModels.printable_model(generator).summary()
    train_AE(generator, train_ds, epochs, save_img_every, weights_path=DenoisingAE_WEIGHTS_PATH)
    visual_latent_space(generator, test_ds)

def Q3(epochs=10, save_img_every=100):
    generator = exModels.Generator()
    discriminator = exModels.Discriminator()
    test_ds, train_ds = get_data_as_tensorslice()
    exModels.printable_model(generator).summary()
    exModels.printable_model(discriminator).summary()
    train_GAN(generator,discriminator , train_ds, epochs, save_img_every, gen_weights_path=GanGenerator_WEIGHTS_PATH, disc_weights_path=GanDiscriminator_WEIGHTS_PATH)

def main():
    # visual_latent_space_from_save()
    # Q1(epochs=10, save_img_every=100)
    # Q2(epochs=10, save_img_every=100)
    Q3(epochs=10, save_img_every=100)


if __name__ == '__main__':
    main()
