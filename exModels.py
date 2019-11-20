from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout


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
        self.max_pool_1 = MaxPooling2D((2, 2))
        self.max_pool_2 = MaxPooling2D((2, 2))
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
        self.conv1 = Conv2D(32, (3, 3), activation='relu')
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
        self.conv1 = Conv2D(32, (3, 3), activation='relu')
        self.max_pool_1 = MaxPooling2D((2, 2))
        self.max_pool_2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.d2 = Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.max_pool_1(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.d2(x)
        return x

class ReducedOverfittingCNNModel(Model):

    def __init__(self, dropoutRate = 0):
        super(ReducedOverfittingCNNModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu')
        self.max_pool_1 = MaxPooling2D((2, 2))
        self.max_pool_2 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.d2 = Dense(10, activation='softmax')
        self.dropout = Dropout(dropoutRate)

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.max_pool_1(x)
        x = self.max_pool_2(x)
        x = self.flatten(x)
        x = self.d2(x)
        x = self.dropout(x) if kwargs.get('run_type') == 'train' else x
        return x

