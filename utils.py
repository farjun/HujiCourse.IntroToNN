import numpy as np
import tensorflow as tf


def resizeShape(curShape, resizeBy):
    return int(curShape[0]/resizeBy), int(curShape[1]/resizeBy)

def computeOneOverW(numOfRows,numOfCols):
    # calculate the "natural frequency map"
    cy, cx = numOfRows/2, numOfCols/2
    x = np.linspace(0, numOfRows, numOfRows)
    y = np.linspace(0, numOfCols, numOfCols)

    Xv, Yv = np.meshgrid(x, y)
    oneoverw = 1/np.sqrt(((Xv-cx)**2+(Yv-cy)**2))
    return oneoverw/np.mean(oneoverw)

def image_fourier(I, rgb = False):
    imFourier = tf.math.abs(tf.signal.fft(tf.cast(I, dtype=tf.complex64)))
    if tf.math.reduce_mean(imFourier) != 0:
      imFourier = imFourier / tf.math.reduce_mean(imFourier)
    if rgb:
        return imFourier[0,:,:,0],imFourier[0,:,:,1],imFourier[0,:,:,2]

    return imFourier