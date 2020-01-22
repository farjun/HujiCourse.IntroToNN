from typing import Dict
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from datetime import datetime
import models as exModels
from tqdm.auto import tqdm

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
from matplotlib import pyplot as plt

BATCH_SIZE = 32
AE_WEIGHTS_PATH = "./weights/AE/v1"
DenoisingAE_WEIGHTS_PATH = "./weights/DenoisingAE/v1"
GanGenerator_WEIGHTS_PATH = "./weights/GanGenerator/v1"
GanDiscriminator_WEIGHTS_PATH = "./weights/GanDiscriminator/v1"


def get_data(normalize=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if normalize:
        x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train[..., tf.newaxis], x_test[..., tf.newaxis]
    return (x_train, y_train), (x_test, y_test)


def get_data_as_tensorslice(normalize=True, shuffle_train=True):
    (x_train, y_train), (x_test, y_test) = get_data(normalize)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if shuffle_train:
        train_ds = train_ds.shuffle(60000)
    train_ds = train_ds.batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)
    return test_ds, train_ds


def getSummaryWriters(modelName, onlyTrain=False):
    # tensor board
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    train_log_dir = f'logs/{modelName}/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    print("run: tensorboard --logdir ./" + train_log_dir + " --port 6006" + " --samples_per_plugin images=1")

    if not onlyTrain:
        test_log_dir = f'logs/{modelName}/' + current_time + '/test'
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        print("run: tensorboard --logdir ./" + test_log_dir + " --port 6007" + " --samples_per_plugin images=1")
        return train_summary_writer, test_summary_writer

    return train_summary_writer


def get_train_step(generator: tf.keras.Model, loss_object):
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = generator(images)
            loss = loss_object(images, predictions)
        gradients = tape.gradient(loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

        train_loss(loss)

    return train_step, train_loss


def train_AE(generator, train_ds, epochs=40, save_img_every=100, weights_path=AE_WEIGHTS_PATH):
    loss_object = tf.keras.losses.MeanSquaredError()
    train_step, train_loss = get_train_step(generator, loss_object)
    train_summary_writer, test_summary_writer = getSummaryWriters(generator.name, onlyTrain=True)
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
                        "max_outputs": 6
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

        # Reset the metrics for the next epoch
        train_loss.reset_states()

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


def get_gan_train_step(generator: tf.keras.Model, discriminator: tf.keras.Model, generator_loss_object,
                       discriminator_loss_object):
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    generator_train_loss = tf.keras.metrics.Mean(name='gen-train_loss')
    discriminator_train_loss = tf.keras.metrics.Mean(name='disc-train_loss')
    generator_train_accuracy = tf.keras.metrics.BinaryAccuracy(name='gen-train_accuracy')
    discriminator_train_accuracy = tf.keras.metrics.BinaryAccuracy(name='disc-train_accuracy')

    @tf.function
    def train_step(images, labels):
        noise = tf.random.normal((BATCH_SIZE, 10))
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_im = generator(noise, training=True)

            real_im_output = discriminator(images, training=True)
            fake_im_output = discriminator(fake_im, training=True)

            generator_loss = generator_loss_object(fake_im_output)
            discriminator_loss = discriminator_loss_object(real_im_output, fake_im_output)

        generator_gradients = gen_tape.gradient(generator_loss, generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        generator_train_loss(generator_loss)
        discriminator_train_loss(discriminator_loss)

        generator_train_accuracy.update_state(tf.zeros_like(real_im_output), real_im_output)
        discriminator_train_accuracy.update_state(tf.ones_like(real_im_output), real_im_output)

    return train_step, generator_train_loss, discriminator_train_loss, generator_train_accuracy, discriminator_train_accuracy


def generate_and_save_images(generator, epoch, seed, saveFig=True):
    predictions = generator(seed)
    num_of_elements = 16
    fig = plt.figure(figsize=(4, 4))

    for i in range(num_of_elements):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 255.0, cmap='gray')
        plt.axis('off')
    if saveFig:
        plt.savefig('./GanGeneratedImages/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def train_GAN(generator, discriminator, train_ds, epochs=40, report_every=100, gen_weights_path=AE_WEIGHTS_PATH,
              disc_weights_path=AE_WEIGHTS_PATH, saveFig=True):
    train_step, gen_train_loss, disc_train_loss, gen_train_accuracy, disc_train_accuracy = get_gan_train_step(generator,
                                                                                                              discriminator,
                                                                                                              generator_loss,
                                                                                                              discriminator_loss)
    gan_train_summary_writer = getSummaryWriters(generator.name, onlyTrain=True)
    train_counter = 0
    seed = tf.random.normal((16, 10))
    for epoch in tqdm(range(1, epochs + 1)):
        for images, labels in train_ds:
            train_step(images, labels)
            train_counter += 1
            if train_counter % report_every == 0:
                with gan_train_summary_writer.as_default():
                    tf.summary.scalar("gen - loss", gen_train_loss.result(), step=train_counter)
                    tf.summary.scalar("gen - accuracy", gen_train_accuracy.result() * 100, step=train_counter)
                    tf.summary.scalar("disc - loss", disc_train_loss.result(), step=train_counter)
                    tf.summary.scalar("disc - accuracy", disc_train_accuracy.result() * 100, step=train_counter)

        generate_and_save_images(generator, epoch, seed, saveFig)

        # Reset the metrics for the next epoch
        gen_train_loss.reset_states()
        disc_train_loss.reset_states()
        gen_train_accuracy.reset_states()
        disc_train_accuracy.reset_states()

    generator.save_weights(gen_weights_path)
    discriminator.save_weights(disc_weights_path)
    gan_train_summary_writer.close()


def get_reducer(reducer: str):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA, LatentDirichletAllocation
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    factory = {
        "tsne": lambda: TSNE(2),
        "pca": lambda: PCA(2),
        "lda": lambda: LinearDiscriminantAnalysis(2)
    }
    if reducer not in factory.keys():
        raise NotImplementedError()
    return factory[reducer]()


def visual_latent_space_from_save(weights_path=AE_WEIGHTS_PATH):
    test_ds, train_ds = get_data_as_tensorslice()
    generator = exModels.CNNGenerator()
    generator.load_weights(weights_path)
    visual_latent_space(generator, test_ds)


def visual_latent_space(generator: exModels.CNNGenerator, test_ds, reducer="tsne", title_suffix=""):
    reducer = get_reducer(reducer)
    max_iters = -1
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
    title = f"visual latent space with {reducer}"
    if title_suffix:
        title += " " + title_suffix
    plt.title(title)
    plt.savefig(title.replace(" ", "_") + ".png")
    plt.show()


def Q1(epochs=10, save_img_every=100):
    generator = exModels.CNNGenerator()
    test_ds, train_ds = get_data_as_tensorslice()
    exModels.printable_model(generator).summary()
    train_AE(generator, train_ds, epochs, save_img_every, weights_path=AE_WEIGHTS_PATH)
    visual_latent_space(generator, test_ds)


def Q2(epochs=10, save_img_every=100, noise_attributes: Dict = None):
    generator = exModels.DenoisingAE(noise_attributes)
    test_ds, train_ds = get_data_as_tensorslice()
    exModels.printable_model(generator).summary()
    train_AE(generator, train_ds, epochs, save_img_every, weights_path=DenoisingAE_WEIGHTS_PATH)
    visual_latent_space(generator, test_ds)


def z_train_step(generator):
    l1_loss = tf.keras.losses.MeanAbsoluteError()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    generator_opt = tf.keras.optimizers.Adam(1e-4)
    latent_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    @tf.function
    def train_step(x, z):
        with  tf.GradientTape() as latent_tape, tf.GradientTape() as generator_tape:
            res_imgs = generator(z)
            loss = l1_loss(x, res_imgs)

            grads = generator_tape.gradient(loss, generator.trainable_variables)
            generator_opt.apply_gradients(zip(grads, generator.trainable_variables))

            grads = latent_tape.gradient(loss, [z])
            latent_opt.apply_gradients(zip(grads, [z]))
            train_loss(loss)

    return train_step, train_loss


def train_glo(test_ds, train_ds, generator, epochs, save_img_every):
    train_summary_writer, test_summary_writer = getSummaryWriters(generator.name)
    train_step, train_loss = z_train_step(generator)

    train_size = get_data()[0][0].shape[0]

    latent_dim = 10
    Z = np.random.normal(size=(train_size, latent_dim))
    Z /= np.linalg.norm(Z, axis=0)
    Z_input = tf.Variable(tf.zeros((BATCH_SIZE, latent_dim)), trainable=True, dtype=tf.float32)

    train_counter = 0
    for epoch in tqdm(range(1, epochs + 1)):
        slicing_idx = 0
        last_end = 0
        for images, labels in train_ds:
            size = images.get_shape()[0]
            if size != BATCH_SIZE:
                break
            start, end = last_end, last_end + size
            last_end = end
            Z_np = Z[start:end]
            Z_input.assign(Z_np)
            train_step(images, Z_input)
            Z[start:end] = Z_input.numpy()
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
                        generator(Z_input),
                        **common
                    )
                    tf.summary.image(
                        "src_img",
                        image_input,
                        **common
                    )
                with train_summary_writer.as_default():
                    tf.summary.scalar("loss", train_loss.result(), step=train_counter)
            slicing_idx += 1
        train_loss.reset_states()

    train_summary_writer.close()
    test_summary_writer.close()


def Q3(epochs=50, save_img_every=100, saveFig=True):
    generator = exModels.Generator(lastActivation='tanh')
    discriminator = exModels.Discriminator()
    test_ds, train_ds = get_data_as_tensorslice()
    # exModels.printable_model(generator).summary()
    # exModels.printable_model(discriminator).summary()
    train_GAN(generator, discriminator, train_ds, epochs, save_img_every, gen_weights_path=GanGenerator_WEIGHTS_PATH,
              disc_weights_path=GanDiscriminator_WEIGHTS_PATH, saveFig=saveFig)


def Q4(epochs=50, save_img_every=100):
    generator = exModels.GLO()
    test_ds, train_ds = get_data_as_tensorslice(shuffle_train=False)
    exModels.printable_model(exModels.GLO(), (10,)).summary()
    train_glo(test_ds, train_ds, generator, epochs, save_img_every)


from typing import List


def run_many_Q2(epochs=10, save_img_every=100, means: List[float] = None, stddevs: List[float] = None):
    if not means:
        means = [0, 0.5, 1, 2]
    if not stddevs:
        stddevs = [0.25, 0.5, 1]

    import itertools
    noise_attributes = {
        "mean": 0, "stddev": 0
    }
    for mean, stddev in itertools.product(means, stddevs):
        noise_attributes["mean"] = mean
        noise_attributes["stddev"] = stddev
        Q2(epochs=epochs, save_img_every=save_img_every, noise_attributes=noise_attributes)


def main():
    # visual_latent_space_from_save()
    Q1(epochs=10, save_img_every=100)
    Q2(epochs=10, save_img_every=100, noise_attributes={
        "mean": 0,
        "stddev": 1,
    })
    Q3(epochs=40, save_img_every=100)
    Q4(epochs=10, save_img_every=100)


if __name__ == '__main__':
    main()
