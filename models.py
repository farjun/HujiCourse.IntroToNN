from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense,Flatten , Reshape
from matplotlib import pyplot as plt

def printable_model(model: Model, input_shape=(28, 28, 1)):
    x = Input(shape=input_shape)
    return Model(inputs=[x], outputs=model.call(x))

class CNNGenerator(Model):

    def __init__(self):
        super(CNNGenerator, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), strides=(2,2), activation='relu')
        self.conv2 = Conv2D(64, (3, 3), strides=(2,2), activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.d2 = Dense(10, activation='relu')
        self.d4 = Dense(7*7*12, activation='relu') # changed the size so it would fit the reshaper, we can change it later
        self.reshaper = Reshape((7,7,12))
        self.conv3 = Conv2DTranspose(64, (3, 3), strides=(2,2), padding='same', activation='relu')
        self.conv4 = Conv2DTranspose(1, (3, 3), strides=(2,2), padding='same', activation='sigmoid')


    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d4(x)
        x = self.reshaper(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x