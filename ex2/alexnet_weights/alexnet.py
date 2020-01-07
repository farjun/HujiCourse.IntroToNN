#
# Base code for Introduction to Neural Networks (67103)
# loads weights from /cs/dataset/alexnet_weights
#
# Written by Raanan Fattal
#

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from PIL import Image
from ex2.alexnet_weights.classes import classes

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)


test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    # OPS
    self.relu = Activation('relu')
    self.maxpool = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid')
    #self.dropout = Dropout(0.4)
    self.softmax = Activation('softmax')

    # Conv layers
    self.conv1 = Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='same')
    self.conv2a = Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding='same')
    self.conv2b = Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding='same')
    self.conv3 = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same')
    self.conv4a = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same')
    self.conv4b = Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='same')
    self.conv5a = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same')
    self.conv5b = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same')
    
    # Fully-connected layers

    self.flatten = Flatten()

    self.dense1 = Dense(4096, input_shape=(100,))
    self.dense2 = Dense(4096)
    self.dense3 = Dense(1000)

    # Network definition
  def call(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta = 0.75, bias = 1.0)
    x = self.maxpool(x)

    x = tf.concat((self.conv2a(x[:,:,:,:48]), self.conv2b(x[:,:,:,48:])), 3)
    x = self.relu(x)
    x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta = 0.75, bias = 1.0)
    x = self.maxpool(x)

    x = self.conv3(x)
    x = self.relu(x)
    x = tf.concat((self.conv4a(x[:,:,:,:192]), self.conv4b(x[:,:,:,192:])), 3)
    x = self.relu(x)
    x = tf.concat((self.conv5a(x[:,:,:,:192]), self.conv5b(x[:,:,:,192:])), 3)
    x = self.relu(x)
    x = self.maxpool(x)
    
    x = self.flatten(x)
    
    x = self.dense1(x)
    x = self.relu(x)
    x = self.dense2(x)
    x = self.relu(x)
    x = self.dense3(x)
    
    return self.softmax(x)

# Create an instance of the model
model = MyModel()

im = Image.open("poodle.png")
#im = Image.open("fish.jpg")
#im = Image.open("dog.png")
im = im.resize([224,224])
I = np.asarray(im).astype(np.float32)
I=I[:,:,:3]

I = np.flip(I,2) # BGR
I=I-[[[104.00698793, 116.66876762, 122.67891434]]] # subtract mean - whitening
#I = I - np.mean(I,axis=(0,1),keepdims=True)

I = np.reshape(I,(1,)+I.shape)

model(I) # Init graph

wdir = '/cs/dataset/alexnet_weights/'

model.conv1.set_weights((np.load(wdir+'conv1.npy'),np.load(wdir+'conv1b.npy')))
model.conv2a.set_weights((np.load(wdir+'conv2_a.npy'),np.load(wdir+'conv2b_a.npy')))
model.conv2b.set_weights((np.load(wdir+'conv2_b.npy'),np.load(wdir+'conv2b_b.npy')))
model.conv3.set_weights((np.load(wdir+'conv3.npy'),np.load(wdir+'conv3b.npy')))
model.conv4a.set_weights((np.load(wdir+'conv4_a.npy'),np.load(wdir+'conv4b_a.npy')))
model.conv5a.set_weights((np.load(wdir+'conv5_a.npy'),np.load(wdir+'conv5b_a.npy')))
model.conv4b.set_weights((np.load(wdir+'conv4_b.npy'),np.load(wdir+'conv4b_b.npy')))
model.conv5b.set_weights((np.load(wdir+'conv5_b.npy'),np.load(wdir+'conv5b_b.npy')))

model.dense1.set_weights((np.load(wdir+'dense1.npy'),np.load(wdir+'dense1b.npy')))
model.dense2.set_weights((np.load(wdir+'dense2.npy'),np.load(wdir+'dense2b.npy')))
model.dense3.set_weights((np.load(wdir+'dense3.npy'),np.load(wdir+'dense3b.npy')))

c = model(I)

top_ind = np.argmax(c)
print("Top1: %d, %s" % (top_ind,classes[top_ind]))