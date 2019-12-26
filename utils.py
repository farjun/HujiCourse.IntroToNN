import numpy as np
import tensorflow as tf
import scipy as sp

def dftmtx(N):
    return sp.fft(sp.eye(N))

def resizeShape(curShape, resizeBy):
    return int(curShape[0]/resizeBy), int(curShape[1]/resizeBy)

def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1J / N )
    return np.power( omega, i * j ) / np.sqrt(N)

def image_fourier(I, rgb = False):

    imFourier = tf.math.abs(tf.signal.fft(tf.cast(I, dtype=tf.complex128)))
    if tf.math.reduce_mean(imFourier) != 0:
      imFourier = imFourier / tf.math.reduce_mean(imFourier)

    if rgb:
        return imFourier[0,:,:,0],imFourier[0,:,:,1],imFourier[0,:,:,2]

    return imFourier