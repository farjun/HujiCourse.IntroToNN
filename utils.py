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

def normellizeFourier(f):
    if tf.math.reduce_mean(f) != 0:
        return f / tf.math.reduce_mean(f)
    return f

def image_fourier(I):
    I_r, I_b, I_g =  I[0,:,:,0],I[0,:,:,1],I[0,:,:,2]
    imFourier1 = normellizeFourier(tf.math.abs(tf.signal.fft(tf.cast(I_r, dtype=tf.complex128))))
    imFourier2 = normellizeFourier(tf.math.abs(tf.signal.fft(tf.cast(I_b, dtype=tf.complex128))))
    imFourier3 = normellizeFourier(tf.math.abs(tf.signal.fft(tf.cast(I_g, dtype=tf.complex128))))
    return imFourier1,imFourier2, imFourier3

def image_fourier_with_grayscale(I):
    grayI =  tf.image.rgb_to_grayscale(I)
    imFourier1 = normellizeFourier(tf.math.abs(tf.signal.fft(tf.cast(grayI, dtype=tf.complex128))))
    return imFourier1, imFourier1, imFourier1