from tqdm.auto import tqdm
import alexnet_model.alexnet
from alexnet_model.alexnet import AlexnetModel, buildAlexnetThatOutputsAt
import numpy as np
from PIL import Image
from alexnet_model.classes import classes
import tensorflow as tf
from _datetime import datetime
EPSILON = 1e-30

def getTestTrainSummaryWriters(modelName):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_path = f'logs/{modelName}/' + current_time
    print("run: tensorboard --logdir ./" + base_path + '/train' + " --port 6006")
    print("run: tensorboard --logdir ./" + base_path + '/test' + " --port 6006")
    train_summary_writer = tf.summary.create_file_writer(base_path + '/train')
    test_summary_writer = tf.summary.create_file_writer(base_path + '/test')
    return train_summary_writer, test_summary_writer

def getSummaryWriter(modelName):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_path = f'logs/{modelName}/' + current_time
    print("run: tensorboard --logdir ./" + base_path + " --port 6006")

    return tf.summary.create_file_writer(base_path)

def getImage(imageName, directory="./alexnet_weights/"):
    I = Image.open(directory + imageName).resize([224, 224])
    I = np.asarray(I).astype(np.float32)
    I = I[:, :, :3]

    I = np.flip(I, 2)  # BGR
    I = I - [[[104.00698793, 116.66876762, 122.67891434]]]  # subtract mean - whitening
    # I = I - np.mean(I,axis=(0,1),keepdims=True)
    I = np.reshape(I, (1,) + I.shape)
    return I


def getModel(img_name, img_dir, weight_dir) -> (AlexnetModel, np.ndarray):
    model = AlexnetModel()
    I = getImage(img_name, img_dir)
    model(I)
    model.setAlexnetWeights(weight_dir)
    return model, I


@tf.function
def loss_object(neuron, I, normalizition_lambda=1e-3):
    return neuron - normalizition_lambda * (tf.norm(I) ** 2)


def get_train_step(model: AlexnetModel, I, loss_object, layer_name):
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            predictions, outputs = model(I, training=True)
            if layer_name not in outputs:
                print(f"No such layer {layer_name}, to possiblities are {outputs.keys()}")
            neuron :tf.keras.layers.Layer = outputs[layer_name]
            loss = loss_object(neuron, I)
            actual_loss = 1 / (loss + EPSILON)  # converting max problem to min problem , EPSILON for avoiding zero div
        gradients = tape.gradient(actual_loss, [I])
        optimizer.apply_gradients(zip(gradients, [I]))
        train_loss(loss)

    # return train_step, train_loss, train_accuracy
    return train_step, train_loss


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



def train():
    model, I = getModel("poodle.png", "./alexnet_weights/", "./alexnet_weights/")

    I_v = tf.Variable(initial_value=tf.zeros((1, 224, 224, 3)), trainable=True)
    train_step, train_loss = get_train_step(model, I_v, loss_object, "conv1")
    iter_count = 10000
    summaryWriter = getSummaryWriter("Q1-I")

    for i in tqdm(range(1,iter_count+1)):
        train_step()
        if i%100 == 0:
            with summaryWriter.as_default():
                tf.summary.image("outI", I_v, step=i)

def main():
    # Create an instance of the model
    model, I = getModel("poodle.png", "./alexnet_weights/", "./alexnet_weights/")
    c = model(I)
    top_ind = np.argmax(c)
    print("Top1: %d, %s" % (top_ind, classes[top_ind]))


if __name__ == "__main__":
    train()
    # main()
    # load_model("alexnet_weights")
