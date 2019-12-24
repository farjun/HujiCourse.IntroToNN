from tqdm.auto import tqdm
import alexnet_model.alexnet
from alexnet_model.alexnet import AlexnetModel
import numpy as np
from PIL import Image
from alexnet_model.classes import classes
import tensorflow as tf
from _datetime import datetime
from enums import NeuronChoice
import utils
from tensorflow.keras.losses import KLDivergence

IMAGE_NAME = 'poodle.png'


def getSummaryWriter(modelName):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_path = f'logs/{modelName}/' + current_time
    print("run: tensorboard --logdir ./" + base_path + " --port 6006")
    return tf.summary.create_file_writer(base_path)


def getImage(imageName, directory="./alexnet_weights/", normelize=None):
    I = Image.open(directory + imageName).resize([224, 224])
    I = np.asarray(I).astype(np.float32)
    I = I[:, :, :3]

    I = np.flip(I, 2)  # BGR
    I = I - [[[104.00698793, 116.66876762, 122.67891434]]]  # subtract mean - whitening
    # I = I - np.mean(I,axis=(0,1),keepdims=True)
    I = np.reshape(I, (1,) + I.shape)
    return I if not normelize else normelize(I)


def getModel(img_name, img_dir, weight_dir) -> (AlexnetModel, np.ndarray):
    model = AlexnetModel()
    I = getImage(img_name, img_dir)
    model(I)
    model.setAlexnetWeights(weight_dir)
    return model, I


def getQ2Loss(imageShape, normalizition_lambda=1e-3, resizeBy=2):
    numOfRows, numOfCols = utils.resizeShape(imageShape, resizeBy)

    def q2_loss(neuron, I):
        fuorierMatrix = utils.computeOneOverW(numOfRows, numOfCols)
        I = tf.image.resize(I, utils.resizeShape(imageShape, resizeBy))
        imFourierC0, imFourierC1, imFourierC2 = utils.image_fourier(I, rgb=True)
        kl = KLDivergence()
        diffs = kl(fuorierMatrix, imFourierC0) + kl(fuorierMatrix, imFourierC1) + kl(fuorierMatrix, imFourierC2)
        loss = neuron - tf.cast(normalizition_lambda * diffs, dtype=tf.float32)
        return loss

    return q2_loss


@tf.function
def q1_loss(neuron, I, normalizition_lambda=1e-3):
    return neuron - normalizition_lambda * (tf.norm(I) ** 2)


# TODO this function is exact as above but allow us to run twice with different neuron
def q1_loss_not_tf_function(neuron, I, normalizition_lambda=1e-3):
    return neuron - normalizition_lambda * (tf.norm(I) ** 2)


def getDistribution(distributionKey: str, shape=(1, 224, 224, 3)):
    if distributionKey == 'normal-1':
        return tf.random.normal(shape)

    if distributionKey == 'zeros':
        return tf.zeros(shape)
    raise ValueError("no such distributionKey: " + distributionKey)


def get_train_step(model: AlexnetModel, I, loss_object, optimizer, neuronChoice: NeuronChoice):
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            prediction, outputs = model(I)
            wanted_layer = outputs[neuronChoice.layer]
            layer_shape = wanted_layer.shape

            if len(layer_shape) == 2:  # affine layer:
                neuron = wanted_layer[0][neuronChoice.index]
                Sc = loss_object(neuron, I)

            else:  # conv layer
                neuron = wanted_layer[0]
                neuron_pixel = neuron[neuronChoice.row, neuronChoice.col, neuronChoice.filter]
                Sc = loss_object(neuron_pixel, I)

            actual_loss = -Sc
            gradients = tape.gradient(actual_loss, [I])
            optimizer.apply_gradients(zip(gradients, [I]))

    return train_step, loss_object


def spreadImPixels(I):
    plot_i = I - tf.reduce_min(I)
    return plot_i / tf.reduce_max(I)


def train(neuronChoice, loss_object, log_folder_name="Q1", distributionKey='normal-1', numberOfIterations=None,
          beforeImShow=None):
    model, I = getModel(IMAGE_NAME, "./alexnet_weights/", "./alexnet_weights/")
    I_v = tf.Variable(initial_value=getDistribution(distributionKey), trainable=True)
    I_v.initialized_value()
    optimizer = tf.keras.optimizers.Adam()
    summaryWriter = getSummaryWriter(log_folder_name)
    train_step, train_loss = get_train_step(model, I_v, loss_object, optimizer, neuronChoice)

    if numberOfIterations is None:
        iter_count = 5000 if neuronChoice.isConvLayer() else 50000
    else:
        iter_count = numberOfIterations

    for i in tqdm(range(1, iter_count + 1)):
        train_step()
        if i % 100 == 0:
            plot_i = beforeImShow(I_v) if beforeImShow else I_v
            with summaryWriter.as_default():
                tf.summary.image(neuronChoice.layer, plot_i, step=i)

    summaryWriter.close()


def Q1(clear_folder=True):
    base_path = "Q1"
    if clear_folder:
        rmtree_path("./logs/" + base_path)

    configs = [
        NeuronChoice(layer="conv1", filter=78, row=0, col=0),
        NeuronChoice(layer="conv2", filter=10, row=0, col=0),
        NeuronChoice(layer="conv3", filter=78, row=0, col=0),
        NeuronChoice(layer="conv4", filter=78, row=0, col=0),
        NeuronChoice(layer="dense1", index=10),
        NeuronChoice(layer="dense2", index=10),
        NeuronChoice(layer="dense3", index=99)  # class 99 is "goose"
    ]

    for neuronChoice in configs:
        print(neuronChoice.layer)
        train(
            neuronChoice,
            q1_loss_not_tf_function,
            log_folder_name=f"Q1/{neuronChoice.layer}",
            beforeImShow=fix_image_to_show
        )


def Q2():
    neuronChoice = NeuronChoice(layer="dense3", index=282)
    image_rows_cols_shape = getImage(IMAGE_NAME).shape[1:3]
    q2_loss = getQ2Loss(image_rows_cols_shape, resizeBy=8)
    train(neuronChoice, q2_loss, log_folder_name="Q2", distributionKey="zeros", numberOfIterations=2000)


def q3(target_index=None,
       reg_lambda=1e-3,
       learning_rate=0.001,
       report_evert=1,
       iterations=100,
       clear_prev_logs=True):
    path = "Q3-I"
    if clear_prev_logs:
        rmtree_path("./logs/" + path)

    image_name = "dog.png"
    model, I = getModel(image_name, "./alexnet_weights/", "./alexnet_weights/")
    c, _ = model(I)
    src_top_ind = np.argmax(c)

    target_index = target_index if target_index else (src_top_ind + 1) % len(classes)
    target_index = tf.constant(target_index)
    noise = tf.Variable(initial_value=tf.random.truncated_normal(I.shape))

    print(f"image={image_name} , prob={c[0][src_top_ind]}")
    print(f"image={image_name} ,target_index prob={c[0][target_index]}")
    print("Top1: %d, %s" % (src_top_ind, classes[src_top_ind]))

    image_writer = getSummaryWriter(f"{path}/images")
    true_idx_probability_writer = getSummaryWriter(f"{path}/true_idx_probability")
    target_idx_probability_writer = getSummaryWriter(f"{path}/target_idx_probability")

    step = get_adversarial_step(model, I, noise, target_index, reg_lambda, learning_rate)
    once = False
    for i in tqdm(range(1, iterations + 1), desc="Q3"):
        step()
        if i % report_evert == 0:
            c, _ = model(I + noise)
            with image_writer.as_default():
                tf.summary.image("Image", fix_image_to_show(I), step=i)
                tf.summary.image("Noise", fix_image_to_show(noise), step=i)
                tf.summary.image("Noise And Image", fix_image_to_show(I + noise), step=i)
            with target_idx_probability_writer.as_default():
                tf.summary.scalar("class probability", c[0][target_index], step=i)
            with true_idx_probability_writer.as_default():
                tf.summary.scalar("class probability", c[0][src_top_ind], step=i)

            if not once and c[0][target_index] > 0.98:
                print(f"We fooled the net first on step {i}")
                once = True


def rmtree_path(path):
    import shutil
    try:
        shutil.rmtree(path)
    except:
        pass


def fix_image_to_show(I):
    return normalize(rgb_to_bgr(I))


def normalize(I):
    I = I - tf.reduce_min(I)
    I = I / tf.reduce_max(I)
    return I


def rgb_to_bgr(I):
    return tf.reshape(np.flip(I[0], 2), I.shape)


def get_adversarial_step(model: tf.keras.Model, image, noise, label, reg_lambda=0.01, learning_rate=0.001):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            predictions, _ = model(image + noise, training=True)
            reg_loss = tf.reduce_mean(noise ** 2)
            loss = loss_object(label, predictions) + reg_lambda * reg_loss
        gradients = tape.gradient(loss, [noise])
        optimizer.apply_gradients(zip(gradients, [noise]))

    return train_step


def main():
    # Create an instance of the model
    model, I = getModel(IMAGE_NAME, "./alexnet_weights/", "./alexnet_weights/")
    c, _ = model(I)
    top_ind = np.argmax(c)
    print("Top1: %d, %s" % (top_ind, classes[top_ind]))


if __name__ == "__main__":
    Q1()
    # Q2()
    # q3(iterations=300, learning_rate=0.1, target_index=401)
