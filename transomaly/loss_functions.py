import numpy as np
import keras.backend as K
import tensorflow as tf


def negloglike_with_error():
    """ Tensorflow negative loglikelihood loss function including data uncertainties with tensorflow probability """

    def loss(y_true, y_pred):
        npb = 2#int(y_true.shape[-1] / 2)  # Assumes no contextual information

        y_err = y_true[:, :, npb:2*npb]
        y_t = y_true[:, :, 0:npb]

        mu = y_pred.loc  # Distribution parameter for the mean
        sigma = y_pred.scale  # Distribution parameter for standard deviation

        mse = -0.5 * tf.square((y_t - mu)) / (tf.square(sigma) + tf.square(y_err))
        sigma_trace = -0.5 * (tf.math.log(tf.square(sigma) + tf.square(y_err)))
        log2pi = -0.5 * np.log(2 * np.pi)

        loglikelihood = mse + sigma_trace + log2pi

        return -loglikelihood

    return loss


def negloglike():
    """ Tensorflow negative loglikelihood loss function with tensorflow probability """
    # loss = lambda y, rv_y: -rv_y.log_prob(y)

    def loss(y_true, y_pred):
        # return -y_pred.log_prob(y_true)

        mu = y_pred.loc  # Distribution parameter for the mean
        sigma = y_pred.scale  # Distribution parameter for standard deviation

        mse = -0.5 * tf.square((y_true - mu)/sigma)
        sigma_trace = -0.5*tf.math.log(tf.square(sigma))
        log2pi = -0.5 * np.log(2 * np.pi)

        loglikelihood = mse + sigma_trace + log2pi

        return -loglikelihood

    return loss


def chisquare_loss():
    """ Compute chi-squared in form of a keras loss function that takes in uncertatinties. """

    def loss(y_true, y_pred):
        y_err = y_true[:, :, 2:4]
        y_t = y_true[:, :, 0:2]
        y_p = y_pred[:, :, 0:2]
        return K.sum(K.square((y_p - y_t)/y_err), axis=-1)

    return loss


def mean_squared_error():
    """ Compute mean squared in form of a keras loss function. """

    def loss(y_true, y_pred):
        y_err = y_true[:, :, 2:4]
        y_t = y_true[:, :, 0:2]
        y_p = y_pred[:, :, 0:2]
        return K.mean(K.square((y_p - y_t)), axis=-1)

    return loss


def mean_squared_error_over_error():
    """ Compute mean squared in form of a keras loss function. """

    def loss(y_true, y_pred):
        y_err = y_true[:, :, 2:4]
        y_t = y_true[:, :, 0:2]
        y_p = y_pred[:, :, 0:2]
        return K.mean(K.square((y_p - y_t)/y_err), axis=-1)

    return loss
