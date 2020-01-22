from typing import Dict

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, BatchNormalization, LeakyReLU


def printable_model(model: Model, input_shape=(28, 28, 1)):
    x = Input(shape=input_shape)
    return Model(inputs=[x], outputs=model.call(x))


class CNNGenerator(Model):

    def __init__(self, lastActivation='sigmoid'):
        super(CNNGenerator, self).__init__()
        self.batchNormalization1 = BatchNormalization()
        self.batchNormalization2 = BatchNormalization()
        self.batchNormalization3 = BatchNormalization()
        self.batchNormalization4 = BatchNormalization()
        self.batchNormalization5 = BatchNormalization()
        self.batchNormalization6 = BatchNormalization()

        self.conv1 = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu')
        self.conv2 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(10, activation='relu')
        self.d3 = Dense(512, activation='relu')
        self.d4 = Dense(7 * 7 * 64, activation='relu')

        self.reshaper = Reshape((7, 7, 64))
        self.conv3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')
        self.conv4 = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation=lastActivation)

    def encode(self, x, **kwargs):
        x = self.conv1(x)
        # x = self.batchNormalization1(x)
        x = self.conv2(x)
        # x = self.batchNormalization2(x)
        x = self.flatten(x)
        x = self.d1(x)
        # x = self.batchNormalization3(x)
        x = self.d2(x)
        return x

    def decode(self, x, **kwargs):
        x = self.d3(x)
        x = self.batchNormalization4(x)
        x = self.d4(x)
        x = self.batchNormalization5(x)
        x = self.reshaper(x)
        x = self.conv3(x)
        x = self.batchNormalization6(x)
        x = self.conv4(x)
        return x

    def call(self, x, **kwargs):
        return self.decode(self.encode(x, **kwargs), **kwargs)


import tensorflow as tf


class DenoisingAE(CNNGenerator):
    def __init__(self, noise_attributes: Dict = None):
        super(DenoisingAE, self).__init__()
        self.noise_attributes = noise_attributes if noise_attributes else {}

    def call(self, x, **kwargs):
        x = x + tf.random.normal(tf.shape(x), **self.noise_attributes)
        return super().call(x)

class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()

        # input is a vector of size [None, 10]
        # self.dense1 = Dense(512, activation='relu')
        self.dense1 = Dense(7 * 7 * 64, activation='relu')

        # this is the shape of the end of the convolutions before
        self.reshape1_decoder = Reshape(target_shape=(7, 7, 64))
        self.conv_decoder_3 = Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')
        self.conv_decoder_4 = Conv2DTranspose(1, 3, strides=2, padding='same', activation='tanh')

    def call(self, x, **kwargs):
        # [batch,10] => [batch,28,28,1]
        decoded = self.dense1(x)
        # decoded = self.dense2(decoded)
        decoded = self.reshape1_decoder(decoded)
        decoded = self.conv_decoder_3(decoded)
        decoded = self.conv_decoder_4(decoded)
        decoded = self.activation_4(decoded)
        return decoded

    def decode(self, encoded):
        # return the decoded version (728 tf version) of the
        # encoded parameter (10 tf vector)
        decoded = self.dense1(encoded)
        decoded = self.reshape1_decoder(decoded)
        decoded = self.conv_decoder_3(decoded)
        decoded = self.conv_decoder_4(decoded)

        return decoded

    def model(self):
        x = Input(shape=(10))
        return Model(inputs=[x], outputs=self.call(x))


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = Conv2D(32, 3, strides=2, activation='relu', padding='same')
        self.conv2 = Conv2D(64, 3, strides=2, activation='relu', padding='same')
        self.flatten = Flatten()
        self.dense3 = Dense(512, activation='relu')
        self.dense4 = Dense(1, activation='sigmoid')


    def call(self, x, **kwargs):
        encoded = self.conv1(x)
        encoded = self.conv2(encoded)
        encoded = self.flatten(encoded)
        encoded = self.dense3(encoded)
        encoded = self.dense4(encoded)
        return encoded

    def model(self):
        x = Input(shape=(28, 28, 1))
        return Model(inputs=[x], outputs=self.call(x))



class GLO(CNNGenerator):
    def __init__(self):
        super(GLO, self).__init__()

    def call(self, x, **kwargs):
        return self.decode(x, **kwargs)
