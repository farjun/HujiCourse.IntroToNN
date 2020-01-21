from typing import Dict

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, BatchNormalization, LeakyReLU


def printable_model(model: Model, input_shape=(28, 28, 1)):
    x = Input(shape=input_shape)
    return Model(inputs=[x], outputs=model.call(x))


class CNNGenerator(Model):

    def __init__(self, lastActivation = 'sigmoid'):
        super(CNNGenerator, self).__init__()
        self.batchNormalization1 = BatchNormalization()
        self.batchNormalization2 = BatchNormalization()
        self.batchNormalization3 = BatchNormalization()
        self.batchNormalization4 = BatchNormalization()
        self.batchNormalization5 = BatchNormalization()
        self.batchNormalization6 = BatchNormalization()

        self.activation1 = LeakyReLU()
        self.activation2 = LeakyReLU()
        self.activation3 = LeakyReLU()
        self.activation4 = LeakyReLU()
        self.activation5 = LeakyReLU()
        self.activation6 = LeakyReLU()

        self.conv1 = Conv2D(32, (3, 3), strides=(2, 2), padding='same')
        self.conv2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same')
        self.flatten = Flatten()
        self.d1 = Dense(512)
        self.d2 = Dense(10)
        self.d3 = Dense(512)
        self.d4 = Dense(7 * 7 * 64)

        self.reshaper = Reshape((7, 7, 64))
        self.conv3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')
        self.conv4 = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation=lastActivation)

    def encode(self, x, **kwargs):
        x = self.conv1(x)
        x = self.batchNormalization1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.batchNormalization2(x)
        x = self.activation2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.batchNormalization3(x)
        x = self.activation3(x)
        x = self.d2(x)
        return x

    def decode(self, x, **kwargs):
        x = self.d3(x)
        x = self.batchNormalization4(x)
        x = self.activation4(x)
        x = self.d4(x)
        x = self.batchNormalization5(x)
        x = self.activation5(x)
        x = self.reshaper(x)
        x = self.conv3(x)
        x = self.batchNormalization6(x)
        x = self.activation6(x)
        x = self.conv4(x)
        return x

    def call(self, x, **kwargs):
        return self.decode(self.encode(x, **kwargs), **kwargs)


import tensorflow as tf


class DenoisingAE(CNNGenerator):
    def __init__(self, noise_attributes: Dict):
        super(DenoisingAE, self).__init__()
        self.noise_attributes = noise_attributes if noise_attributes else {}

    def call(self, x, **kwargs):
        x = x + tf.random.normal(tf.shape(x), **self.noise_attributes)
        return super().call(x)


class Discriminator(CNNGenerator):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.densePredict = Dense(1)

    def call(self, x, **kwargs):
        x = super().encode(x)
        x = self.flatten(x)
        return self.densePredict(x)

class Generator(CNNGenerator):
    def __init__(self, lastActivation = 'sigmoid'):
        super(Generator, self).__init__(lastActivation = lastActivation)

    def call(self, x, **kwargs):
        return super().decode(x)


class GLO(CNNGenerator):
    def __init__(self):
        super(GLO, self).__init__()

    def call(self, x, **kwargs):
        return self.decode(x,**kwargs)
