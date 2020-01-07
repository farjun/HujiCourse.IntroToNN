#
# Base code for Introduction to Neural Networks (67103)
# loads weights from /cs/dataset/alexnet_weights
#
# Written by Raanan Fattal
#

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D
from tensorflow.keras import Model
from ex2.enums import AlexnetLayers

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class AlexnetModel(Model):
    def __init__(self):
        super(AlexnetModel, self).__init__()

        # OPS
        self.relu = Activation('relu')
        self.maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')
        # self.dropout = Dropout(0.4)
        self.softmax = Activation('softmax', )

        # Conv layers
        self.conv1 = Conv2D(name="conv1", filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(4, 4),
                            padding='same')
        self.conv2a = Conv2D(name="conv2a", filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')
        self.conv2b = Conv2D(name="conv2b", filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same')
        self.conv3 = Conv2D(name="conv3", filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv4a = Conv2D(name="conv4a", filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv4b = Conv2D(name="conv4b", filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv5a = Conv2D(name="conv5a", filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')
        self.conv5b = Conv2D(name="conv5b", filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')

        # Fully-connected layers

        self.flatten = Flatten()

        self.dense1 = Dense(4096, name="dense1", input_shape=(100,))
        self.dense2 = Dense(4096, name="dense2")
        self.dense3 = Dense(1000, name="dense3")

    def setAlexnetWeights(self, wdir):
        self.conv1.set_weights((np.load(wdir + 'conv1.npy'), np.load(wdir + 'conv1b.npy')))
        self.conv2a.set_weights((np.load(wdir + 'conv2_a.npy'), np.load(wdir + 'conv2b_a.npy')))
        self.conv2b.set_weights((np.load(wdir + 'conv2_b.npy'), np.load(wdir + 'conv2b_b.npy')))
        self.conv3.set_weights((np.load(wdir + 'conv3.npy'), np.load(wdir + 'conv3b.npy')))
        self.conv4a.set_weights((np.load(wdir + 'conv4_a.npy'), np.load(wdir + 'conv4b_a.npy')))
        self.conv5a.set_weights((np.load(wdir + 'conv5_a.npy'), np.load(wdir + 'conv5b_a.npy')))
        self.conv4b.set_weights((np.load(wdir + 'conv4_b.npy'), np.load(wdir + 'conv4b_b.npy')))
        self.conv5b.set_weights((np.load(wdir + 'conv5_b.npy'), np.load(wdir + 'conv5b_b.npy')))

        self.dense1.set_weights((np.load(wdir + 'dense1.npy'), np.load(wdir + 'dense1b.npy')))
        self.dense2.set_weights((np.load(wdir + 'dense2.npy'), np.load(wdir + 'dense2b.npy')))
        self.dense3.set_weights((np.load(wdir + 'dense3.npy'), np.load(wdir + 'dense3b.npy')))

    def getOutputAtLAyer(self, layer: AlexnetLayers):
        return self.get_layer(layer.name).output

        # Network definition

    def call(self, x, **kwargs):
        layers_by_name = dict()

        x = self.conv1(x)
        x = add_to_dict("conv1", self.relu(x), layers_by_name)

        x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
        x = self.maxpool(x)

        x = tf.concat((self.conv2a(x[:, :, :, :48]), self.conv2b(x[:, :, :, 48:])), 3)
        x = add_to_dict("conv2", self.relu(x), layers_by_name)

        x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = add_to_dict("conv3", self.relu(x), layers_by_name)
        x = tf.concat((self.conv4a(x[:, :, :, :192]), self.conv4b(x[:, :, :, 192:])), 3)
        x = add_to_dict("conv4", self.relu(x), layers_by_name)

        x = tf.concat((self.conv5a(x[:, :, :, :192]), self.conv5b(x[:, :, :, 192:])), 3)
        x = add_to_dict("conv5", self.relu(x), layers_by_name)
        x = self.maxpool(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = add_to_dict("dense1", self.relu(x), layers_by_name)
        x = self.dense2(x)
        x = add_to_dict("dense2", self.relu(x), layers_by_name)
        x = self.dense3(x)
        x = add_to_dict("dense3", self.relu(x), layers_by_name)
        x = add_to_dict("softmax", self.softmax(x), layers_by_name)
        return x, layers_by_name


import typing


def add_to_dict(key: str, item: tf.keras.layers.Layer, to_add: typing.Dict[str, tf.keras.layers.Layer]):
    if key in to_add:
        print(f"Warning: item {key} is already exist in to_add dictionary")
    to_add[key] = item
    return item

