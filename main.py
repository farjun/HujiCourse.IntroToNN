import alexnet_model.alexnet
from alexnet_model.alexnet import AlexnetModel, buildAlexnetThatOutputsAt
import numpy as np
from PIL import Image
from alexnet_model.classes import classes
from enums import AlexnetLayers
import tensorflow as tf
from tensorflow.keras import  Input

def getImage(imageName, directory = "./alexnet_weights/"):
    I = Image.open(directory + imageName).resize([224, 224])
    I = np.asarray(I).astype(np.float32)
    I = I[:, :, :3]

    I = np.flip(I, 2)  # BGR
    I = I - [[[104.00698793, 116.66876762, 122.67891434]]]  # subtract mean - whitening
    # I = I - np.mean(I,axis=(0,1),keepdims=True)
    I = np.reshape(I, (1,) + I.shape)
    return I

def getModel():
    model = AlexnetModel()
    x = Input(shape=(224, 224, 3))
    model.call(x)
    return model

def get_train_step(model: tf.keras.Model, loss_object):
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(images, labels):
        I = tf.Variable(tf.random.uniform((1, 28, 28, 1)))
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        # gradients = tape.gradient(loss, model.trainable_variables)
        # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        gradients = tape.gradient(loss, [I, ])
        optimizer.apply_gradients(zip(gradients, [I, ]))
        # where I is a variable:
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

def main():
    # Create an instance of the model
    model = getModel()
    I = getImage("fish3.jpeg",directory = "./more_data/")

    model(I)  # Init graph
    model.setAlexnetWeights('./alexnet_weights/')
    c = model(I)
    print(model.getOutputAtLAyer(AlexnetLayers.conv3))


    top_ind = np.argmax(c)
    print("Top1: %d, %s" % (top_ind, classes[top_ind]))

if __name__ == "__main__":
    main()