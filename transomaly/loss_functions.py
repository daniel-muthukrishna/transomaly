import keras.backend as K


def negloglike():
    loss = lambda y, rv_y: -rv_y.log_prob(y)

    # def loss(y_true, y_pred):
    #     # y_err = y_true[:, :, 2:4]
    #     # y_t = y_true[:, :, 0:2]
    #     # y_p = y_pred[:, :, 0:2]
    #     return -y_pred.log_prob(y_true)

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
