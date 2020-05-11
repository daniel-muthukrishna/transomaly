import numpy as np
import keras.backend as K
import tensorflow as tf


def negloglike():
    # loss = lambda y, rv_y: -rv_y.log_prob(y)

    def loss(y_true, y_pred):
        # # y_err = y_true[:, :, 2:4]
        # # y_t = y_true[:, :, 0:2]
        # # y_p = y_pred[:, :, 0:2]
        # return -y_pred.log_prob(y_true)

        mu = y_pred.loc  # Distribution parameter for the mean
        sigma = y_pred.scale  # Distribution parameter for standard deviation

        mse = -0.5 * tf.square((y_true - mu)/sigma)
        sigma_trace = -tf.math.log(sigma)
        log2pi = -0.5 * np.log(2*np.pi)

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


# # https://gist.github.com/sergeyprokudin/4a50bf9b75e0559c1fcd2cae860b879e
# n_dims = int(int(y_pred.shape[2]) / 2)
# mu = y_pred[:, :, 0:n_dims]
# logsigma = y_pred[:, :, n_dims:]
#
# mse = -0.5 * K.sum(K.square((y_true - mu) / K.exp(logsigma)), axis=-1)
# sigma_trace = -K.sum(logsigma, axis=-1)
# log2pi = -0.5 * n_dims * np.log(2 * np.pi)
#
# log_likelihood = mse + sigma_trace + log2pi
#
# return K.mean(-log_likelihood, axis=-1)
