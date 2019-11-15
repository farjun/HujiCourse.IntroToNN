import tensorflow as tf
import keras
from keras import Model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D


class CNNModel(Model):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = Conv2D(32, (5, 5), activation='relu')
        self.max_pool_1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (5, 5), activation='relu')
        self.max_pool_2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, mask=None):
        x = self.conv1(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


if __name__ == '__main__':
    model = CNNModel()
    pass