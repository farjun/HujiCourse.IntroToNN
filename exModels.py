from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, AveragePooling2D


def printable_model(model: Model, input_shape=(28, 28, 1)):
    x = Input(shape=input_shape)
    return Model(inputs=[x], outputs=model.call(x))


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

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


class LinearModel(Model):

    def __init__(self):
        super(LinearModel, self).__init__()
        self.conv1 = Conv2D(32, (5, 5))
        self.max_pool_1 = AveragePooling2D((2, 2))
        self.conv2 = Conv2D(64, (5, 5))
        self.max_pool_2 = AveragePooling2D((2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(1024)
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.max_pool_1(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


class SmallCNN(Model):

    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = Conv2D(3, (5, 5), strides=2, activation='relu')
        self.flatten = Flatten()
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d2(x)
        return x


class ReducedCNNModel(Model):

    def __init__(self):
        super(ReducedCNNModel, self).__init__()
        self.conv1 = Conv2D(4, (5, 5), activation='relu')
        self.max_pool_1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(2, (5, 5), activation='relu')
        self.max_pool_2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(10, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

class ReducedOverfittingCNNModel(Model):

    def __init__(self, dropoutRate=0):
        super(ReducedOverfittingCNNModel, self).__init__()
        self.conv1 = Conv2D(8, (3, 3), activation='relu')
        self.max_pool_1 = MaxPooling2D((2, 2))
        self.max_pool_2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(124, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, training=None, mask=None):
        x = self.conv1(x)
        x = self.max_pool_1(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x

class DropoutOverfittingCNNModel(Model):

    def __init__(self, dropoutRate=0):
        super(DropoutOverfittingCNNModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu')
        self.dropout1 = Dropout(dropoutRate)
        self.max_pool_1 = MaxPooling2D((2, 2))
        self.max_pool_2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')
        self.dropout2 = Dropout(dropoutRate)
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, training=None, mask=None):
        x = self.conv1(x)
        x = self.dropout1(x) if training else x
        x = self.max_pool_1(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.dropout2(x) if training else x
        x = self.d2(x)
        return x

class SumCalculatorConcatinatedModel(Model):

    def __init__(self):
        super(SumCalculatorConcatinatedModel, self).__init__()
        self.conv1 = Conv2D(32, (5, 5), activation='relu')
        self.max_pool_1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (5, 5), activation='relu')
        self.max_pool_2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')
        self.d2 = Dense(19, activation='softmax')

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


class SumCalculatorStackedModel(Model):

    def __init__(self):
        super(SumCalculatorStackedModel, self).__init__()
        self.conv1 = Conv2D(32, (5, 5), activation='relu')
        self.max_pool_1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (5, 5), activation='relu')
        self.max_pool_2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x
