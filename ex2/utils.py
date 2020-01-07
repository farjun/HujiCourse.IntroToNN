import numpy as np
import tensorflow as tf
import scipy as sp

def dftmtx(N):
    return sp.fft(sp.eye(N))

def resizeShape(curShape, resizeBy):
    return int(curShape[0]/resizeBy), int(curShape[1]/resizeBy)

def getDFT(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1J / N )
    res = np.power(omega, i * j) / np.sqrt(N)
    return res / np.mean(res)

def normellizeFourier(f):
    if tf.math.reduce_mean(f) != 0:
        return f / tf.math.reduce_mean(f)
    return f

def image_fourier_with_grayscale(I):
    grayI =  tf.image.rgb_to_grayscale(I)
    imFourier1 = normellizeFourier(tf.math.abs(tf.signal.fft(tf.cast(grayI, dtype=tf.complex128))))
    return imFourier1, imFourier1, imFourier1

def image_fourier(I, rgb = False):
    """
    :return: normelized image fourier for 3 or single channel
    """
    imFourier = tf.math.abs(tf.signal.fft(tf.cast(I, dtype=tf.complex64)))
    imFourier = normellizeFourier(imFourier)

    if rgb:
        return imFourier[0,:,:,0],imFourier[0,:,:,1],imFourier[0,:,:,2]

    return imFourier