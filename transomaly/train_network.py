import os
import numpy as np
# from keras.models import Sequential
# from keras.models import load_model
# from keras.layers import Dense, Input
# from keras.layers import LSTM, GRU
# from keras.layers import Dropout, BatchNormalization, Activation, TimeDistributed, Masking

import tensorflow.compat.v1 as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import Dense, Input, LSTM, GRU, Dropout, BatchNormalization, Activation, TimeDistributed, Masking
from tensorflow.python.keras.backend import set_session

import tensorflow_probability as tfp
tfd = tfp.distributions

import astrorapid

from transomaly.prepare_training_set import PrepareTrainingSetArrays
from transomaly.loss_functions import mean_squared_error, chisquare_loss, mean_squared_error_over_error
from transomaly.plot_metrics import plot_metrics, plot_history


def train_model(X_train, X_test, y_train, y_test, yerr_train, yerr_test, fig_dir='.', epochs=20, retrain=False,
                passbands=('g', 'r'), model_change='', reframe=False, probabilistic=False, train_from_last_stop=0):

    sess = tf.Session()

    model_name = f"keras_model_epochs{epochs+train_from_last_stop}_{model_change}"
    model_filename = os.path.join(fig_dir, model_name, f"{model_name}.hdf5")
    if not os.path.exists(os.path.join(fig_dir, model_name)):
        os.makedirs(os.path.join(fig_dir, model_name))

    npb = len(passbands)

    if probabilistic:
        negloglik = lambda y, rv_y: -rv_y.log_prob(y)
        lossfn = negloglik
    elif 'chi2' in model_change:
        lossfn = chisquare_loss()
    elif 'mse_oe' in model_change:
        lossfn = mean_squared_error_over_error()
    else:
        lossfn = mean_squared_error()

    if not retrain and os.path.isfile(model_filename):
        set_session(sess)
        if probabilistic:
            with sess.as_default():
                model = load_model(model_filename, custom_objects={'loss': lossfn})
        else:
            model = load_model(model_filename, custom_objects={'loss': lossfn})
    else:
        with sess.as_default():
            if train_from_last_stop:
                old_model_name = f"keras_model_epochs{train_from_last_stop}_{model_change}"
                old_model_filename = os.path.join(fig_dir, old_model_name, f"{old_model_name}.hdf5")
                model = load_model(old_model_filename, custom_objects={'loss': mean_squared_error()})
                history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
                                    batch_size=250, verbose=2, inital_epoch=train_from_last_stop)
                print(model.summary())
                model.save(model_filename)
            else:
                model = Sequential()

                model.add(Masking(mask_value=0., input_shape=(X_train.shape[1], X_train.shape[2])))

                model.add(LSTM(100, return_sequences=True))
                # # model.add(Dropout(0.2, seed=42))
                # # model.add(BatchNormalization())

                # model.add(LSTM(100, return_sequences=True))
                # # model.add(Dropout(0.2, seed=42))
                # # model.add(BatchNormalization())
                # # model.add(Dropout(0.2, seed=42))

                if reframe is True:
                    model.add(LSTM(100))
                    # model.add(Dropout(0.2, seed=42))
                    # model.add(BatchNormalization())
                    # model.add(Dropout(0.2, seed=42))
                    model.add(Dense(npb))
                else:
                    model.add(LSTM(100, return_sequences=True))
                    # # model.add(Dropout(0.2, seed=42))
                    # # model.add(BatchNormalization())
                    # # model.add(Dropout(0.2, seed=42))
                    if probabilistic:

                        # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
                        def prior_trainable(kernel_size, bias_size=0, dtype=None):
                            print(dtype)
                            n = kernel_size + bias_size
                            print(kernel_size, bias_size)
                            return tf.keras.Sequential([
                                tfp.layers.VariableLayer(n, dtype=dtype),
                                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                                    tfd.Normal(loc=t, scale=1),
                                    reinterpreted_batch_ndims=1)),
                            ])

                        # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
                        def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
                            print(dtype)
                            n = kernel_size + bias_size
                            print(kernel_size, bias_size)
                            c = np.log(np.expm1(1.))
                            return tf.keras.Sequential([
                                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                                tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                                    tfd.Normal(loc=t[..., :n],
                                               scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                                    reinterpreted_batch_ndims=1)),
                            ])

                        # every second column is the mean; every other column is the stddev
                        model.add(TimeDistributed(Dense(npb*2)))
                        # model.add(TimeDistributed(tfp.layers.DenseFlipout(npb * 2)))
                        # model.add(TimeDistributed(tfp.layers.VariationalGaussianProcess(npb * 2)))
                    else:
                        model.add(TimeDistributed(Dense(npb*1)))

                if probabilistic:
                    model.add(tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., ::2],
                                                                                 scale=tf.math.softplus(t[..., 1::2]))),)
                    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
                    model.compile(loss=negloglik, optimizer='adam')
                else:
                    if 'chi2' in model_change:
                        model.compile(loss=chisquare_loss(), optimizer='adam')
                    elif 'mse_oe' in model_change:
                        model.compile(loss=mean_squared_error_over_error(), optimizer='adam')
                    elif reframe is True:
                        model.compile(loss='mse', optimizer='adam')
                    else:
                        model.compile(loss=mean_squared_error(), optimizer='adam')
                history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=200, verbose=2)

                # import pdb; pdb.set_trace()
                print(model.summary())
                model.save(model_filename)

        plot_history(history, model_filename)

    return model, model_name, sess


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    for npred in range(1, 2):
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(SCRIPT_DIR, '..', 'data/real_ZTF_data_from_osc')
        save_dir = os.path.join(SCRIPT_DIR, '..', 'data/saved_real_ZTF_light_curves')
        training_set_dir = os.path.join(SCRIPT_DIR, '..', 'data/training_set_files')
        fig_dir = os.path.join(SCRIPT_DIR, '..', 'plots')
        get_data_func = astrorapid.get_training_data.get_real_ztf_training_data
        passbands = ('g', 'r')
        contextual_info = ()
        known_redshift = True if 'redshift' in contextual_info else False
        nprocesses = 1
        class_nums = ('Ia',)
        otherchange = ''
        nsamples = 1
        extrapolate_gp = True
        redo = False
        train_epochs = 5
        retrain = False
        reframe_problem = False
        # npred = 1
        probabilistic = False
        train_from_last_stop = 0
        normalise = True
        nn_architecture_change = f"real_data_{'probabilistic_' if probabilistic else ''}predictpoint{npred}timestepsinfuture_normalised{normalise}_mse_nodropout_100lstmneurons"

        fig_dir = os.path.join(fig_dir,
                               "model_{}_ci{}_ns{}_c{}".format(otherchange, contextual_info, nsamples, class_nums))
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        preparearrays = PrepareTrainingSetArrays(passbands, contextual_info, data_dir, save_dir, training_set_dir, redo,
                                                 get_data_func)
        X_train, X_test, y_train, y_test, Xerr_train, Xerr_test, yerr_train, yerr_test, \
        timesX_train, timesX_test, labels_train, labels_test, objids_train, objids_test = \
            preparearrays.make_training_set(class_nums, nsamples, otherchange, nprocesses, extrapolate_gp,
                                            reframe=reframe_problem, npred=npred, normalise=normalise)

        model, model_name, tf_sess = train_model(X_train, X_test, y_train, y_test, yerr_train, yerr_test,
                                                 fig_dir=fig_dir, epochs=train_epochs,
                                                 retrain=retrain, passbands=passbands,
                                                 model_change=nn_architecture_change, reframe=reframe_problem,
                                                 probabilistic=probabilistic, train_from_last_stop=train_from_last_stop)

        # plot_metrics(model, model_name, X_test, y_test, timesX_test, yerr_test, labels_test, objids_test, passbands=passbands,
        #             fig_dir=fig_dir, nsamples=nsamples, data_dir=data_dir, save_dir=save_dir, nprocesses=nprocesses, plot_gp=True, extrapolate_gp=extrapolate_gp, reframe=reframe_problem, plot_name='', npred=npred, probabilistic=probabilistic, tf_sess=tf_sess)

        plot_metrics(model, model_name, X_train, y_train, timesX_train, yerr_train, labels_train, objids_train,
                     passbands=passbands,
                     fig_dir=fig_dir, nsamples=nsamples, data_dir=data_dir, save_dir=save_dir, nprocesses=nprocesses,
                     plot_gp=True, extrapolate_gp=extrapolate_gp, reframe=reframe_problem, plot_name='_training_set',
                     npred=npred, probabilistic=probabilistic, tf_sess=tf_sess, known_redshift=known_redshift,
                     get_data_func=get_data_func)

        # Test on other classes  #51,60,62,70 AndOtherTypes
        X_train, X_test, y_train, y_test, Xerr_train, Xerr_test, yerr_train, yerr_test, \
        timesX_train, timesX_test, labels_train, labels_test, objids_train, objids_test = \
            preparearrays.make_training_set(class_nums=(1, 51,), nsamples=1, otherchange='getKnAndOtherTypes',
                                            nprocesses=nprocesses, extrapolate_gp=extrapolate_gp,
                                            reframe=reframe_problem, npred=npred, normalise=normalise)
        plot_metrics(model, model_name, X_train, y_train, timesX_train, yerr_train, labels_train, objids_train,
                     passbands=passbands,
                     fig_dir=fig_dir, nsamples=nsamples, data_dir=data_dir, save_dir=save_dir, nprocesses=nprocesses,
                     plot_gp=True, extrapolate_gp=extrapolate_gp, reframe=reframe_problem, plot_name='anomaly',
                     npred=npred, probabilistic=probabilistic, tf_sess=tf_sess, known_redshift=known_redshift,
                     get_data_func=get_data_func)


if __name__ == '__main__':
    main()