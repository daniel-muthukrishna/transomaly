import os
import numpy as np
import h5py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Masking, Dropout, Input

import tensorflow_probability as tfp
tfd = tfp.distributions

# from bayesian_tcn import TCN, tcn_full_summary
from tcn import TCN, tcn_full_summary

import astrorapid

from transomaly.prepare_training_set import PrepareTrainingSetArrays
from transomaly.loss_functions import mean_squared_error, chisquare_loss, mean_squared_error_over_error, negloglike, negloglike_with_error
from transomaly.plot_metrics import plot_metrics, plot_history


def build_model(X_train, passbands=('g', 'r'), reframe=False, probabilistic=False, nunits=100, bayesian=False,
                dropout_rate=0.0):
    if bayesian:
        mc_dropout = True
    else:
        mc_dropout = None
    npb = len(passbands)

    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    hidden = Masking(mask_value=0.)(inputs)

    hidden = TCN(nunits, return_sequences=True, kernel_size=2, nb_stacks=1, dilations=[1, 2, 4, 8],
                 padding='causal', use_skip_connections=True, dropout_rate=dropout_rate, activation='sigmoid')(hidden, training=mc_dropout)
    hidden = Dropout(dropout_rate)(hidden, training=mc_dropout)

    if reframe is True:
        hidden = LSTM(nunits)(hidden)
        hidden = Dense(npb)(hidden)
    else:
        if probabilistic:
            hidden = TimeDistributed(Dense(npb * 2))(hidden)
        else:
            hidden = TimeDistributed(Dense(npb * 1))(hidden)

    if probabilistic:
        outputs = tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :npb], scale=1e-3 + tf.math.softplus(0.01 * t[..., npb:])))(hidden)
    else:
        outputs = hidden

    model = Model(inputs, outputs)

    return model


def train_model(X_train, X_test, y_train, y_test, yerr_train, yerr_test, fig_dir='.', epochs=20, retrain=False,
                passbands=('g', 'r'), model_change='', reframe=False, probabilistic=False, train_from_last_stop=0,
                batch_size=50, nunits=100, use_uncertainties=False, bayesian=False, dropout_rate=0.0, learning_rate=0.01):

    model_name = f"keras_model_epochs{epochs+train_from_last_stop}_{model_change}"
    model_filename = os.path.join(fig_dir, model_name, f"{model_name}.hdf5")
    if not os.path.exists(os.path.join(fig_dir, model_name)):
        os.makedirs(os.path.join(fig_dir, model_name))

    if probabilistic:
        if use_uncertainties:
            lossfn = negloglike_with_error()
        else:
            lossfn = negloglike()
    elif 'chi2' in model_change:
        lossfn = chisquare_loss()
    elif 'mse_oe' in model_change:
        lossfn = mean_squared_error_over_error()
    else:
        lossfn = mean_squared_error()

    if not retrain and os.path.isfile(model_filename):
        model = load_model(model_filename, custom_objects={'loss': lossfn, 'TCN': TCN})
    else:
        if train_from_last_stop:
            old_model_name = f"keras_model_epochs{train_from_last_stop}_{model_change}"
            old_model_filename = os.path.join(fig_dir, old_model_name, f"{old_model_name}.hdf5")
            model = load_model(model_filename, custom_objects={'loss': lossfn, 'TCN': TCN})
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
                                batch_size=batch_size, verbose=2, inital_epoch=train_from_last_stop)
        else:
            model = build_model(X_train, passbands, reframe, probabilistic, nunits, bayesian, dropout_rate=dropout_rate)
            model.compile(loss=lossfn, optimizer=tf.optimizers.Adam(learning_rate=learning_rate))
            tcn_full_summary(model, expand_residual_blocks=True)
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=2)

        print(model.summary())
        model.save(model_filename)

        plot_history(history, model_filename)

    return model, model_name


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(SCRIPT_DIR, '..', 'data/ZTF_20190512')
    save_dir = os.path.join(SCRIPT_DIR, '..', 'data/saved_light_curves')
    training_set_dir = os.path.join(SCRIPT_DIR, '..', 'data/training_set_files')
    get_data_func = astrorapid.get_training_data.get_data_from_snana_fits
    fig_dir = os.path.join(SCRIPT_DIR, '..', 'plots')
    passbands = ('g', 'r')
    contextual_info = ()
    known_redshift = True if 'redshift' in contextual_info else False
    nprocesses = None
    class_nums = (1,)
    otherchange = 'spline_interpolation'  # single_object_1_50075859'
    use_gp_interp = False
    nsamples = 1
    extrapolate_gp = True
    redo = False
    train_epochs = 20
    retrain = False
    reframe_problem = False
    npred = 1
    probabilistic = True
    batch_size = 128
    learning_rate = 0.01
    nunits = 30
    train_from_last_stop = 0
    normalise = True
    use_uncertainties = True
    bayesian = True
    dropout_rate = 0.3

    nn_architecture_change = f"mcdropout_TCN_{'probabilistic_' if probabilistic else ''}bayesian{bayesian}_uncertainties{use_uncertainties}_predictfuture{npred}point_normalised{normalise}_dropout{dropout_rate}_units{nunits}_batchsize{batch_size}_lr{learning_rate}"

    fig_dir = os.path.join(fig_dir, "model_{}_ci{}_ns{}_c{}".format(otherchange, contextual_info, nsamples, class_nums))
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    preparearrays = PrepareTrainingSetArrays(passbands, contextual_info, data_dir, save_dir, training_set_dir, redo,
                                             get_data_func, use_gp_interp=use_gp_interp)
    X_train, X_test, y_train, y_test, Xerr_train, Xerr_test, yerr_train, yerr_test, \
    timesX_train, timesX_test, labels_train, labels_test, objids_train, objids_test = \
        preparearrays.make_training_set(class_nums, nsamples, otherchange, nprocesses, extrapolate_gp,
                                        reframe=reframe_problem, npred=npred, normalise=normalise,
                                        use_uncertainties=use_uncertainties)

    model, model_name = train_model(X_train, X_test, y_train, y_test, yerr_train, yerr_test, fig_dir=fig_dir,
                                    epochs=train_epochs,
                                    retrain=retrain, passbands=passbands, model_change=nn_architecture_change,
                                    reframe=reframe_problem, probabilistic=probabilistic,
                                    train_from_last_stop=train_from_last_stop, batch_size=batch_size, nunits=nunits,
                                    use_uncertainties=use_uncertainties, bayesian=bayesian, dropout_rate=dropout_rate)

    plot_metrics(model, model_name, X_test, y_test, timesX_test, yerr_test, labels_test, objids_test,
                 passbands=passbands,
                 fig_dir=fig_dir, nsamples=nsamples, data_dir=data_dir, save_dir=save_dir, nprocesses=nprocesses,
                 plot_gp=True, extrapolate_gp=extrapolate_gp, reframe=reframe_problem, plot_name='', npred=npred,
                 probabilistic=probabilistic, known_redshift=known_redshift, get_data_func=get_data_func,
                 normalise=normalise, bayesian=bayesian)

    plot_metrics(model, model_name, X_train, y_train, timesX_train, yerr_train, labels_train, objids_train,
                 passbands=passbands,
                 fig_dir=fig_dir, nsamples=nsamples, data_dir=data_dir, save_dir=save_dir, nprocesses=nprocesses,
                 plot_gp=True, extrapolate_gp=extrapolate_gp, reframe=reframe_problem, plot_name='_training_set',
                 npred=npred, probabilistic=probabilistic, known_redshift=known_redshift, get_data_func=get_data_func,
                 normalise=normalise, bayesian=bayesian)

    # Test on other classes  #51,60,62,70 AndOtherTypes
    X_train, X_test, y_train, y_test, Xerr_train, Xerr_test, yerr_train, yerr_test, \
    timesX_train, timesX_test, labels_train, labels_test, objids_train, objids_test = \
        preparearrays.make_training_set(class_nums=(1,51,), nsamples=1, otherchange='getKnAndOtherTypes', nprocesses=nprocesses, extrapolate_gp=extrapolate_gp, reframe=reframe_problem, npred=npred, normalise=normalise, use_uncertainties=use_uncertainties)
    plot_metrics(model, model_name, X_train, y_train, timesX_train, yerr_train, labels_train, objids_train, passbands=passbands,
                 fig_dir=fig_dir, nsamples=nsamples, data_dir=data_dir, save_dir=save_dir, nprocesses=nprocesses, plot_gp=True, extrapolate_gp=extrapolate_gp, reframe=reframe_problem, plot_name='anomaly', npred=npred, probabilistic=probabilistic, known_redshift=known_redshift, get_data_func=get_data_func, normalise=normalise, bayesian=bayesian)


    # class_nums_test_on = (1, 2, 12, 14, 3, 13, 41, 43, 51, 60, 61, 62, 63, 64, 70)  # , 80, 81, 83)
    # ignore_class_names_test_on = []  # ignore_test_on class_names
    # model_filepaths = {'SNIa': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(1,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    'SNIa-x': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(43,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    # 'Ia-91bg': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(41,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    'SNII': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(2, 12, 14)/keras_model_epochs500_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs500_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    'SNIbc': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(3, 13)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    # 'Kilonovae': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(51,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    'SLSN-I': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(60,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    # 'PISN': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(61,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    # 'ILOT': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(62,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    # 'CART': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(63,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    'TDE': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(64,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    'AGN': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(70,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    # 'RRLyrae': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(80,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    # 'Mdwarf': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(81,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    # 'Eclip. Bin.': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(83,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
    #                    }
    # similarity_matrix, similarity_matrix_std = get_similarity_matrix(class_nums_test_on, model_filepaths, preparearrays, nprocesses, extrapolate_gp, nsamples, ignore_class_names_test_on)
    # plot_similarity_matrix(similarity_matrix, similarity_matrix_std)
    # # plot_similarity_scatter_plot(similarity_matrix)

if __name__ == '__main__':
    main()


# {'SNIa-norm': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(1,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        'SNII': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(2, 12, 14)/keras_model_epochs500_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs500_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        'SNIbc': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(3, 13)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        'SNIa-91bg': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(41,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        'SNIa-x': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(43,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        # 'Kilonovae': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(51,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        'SLSN-I': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(60,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        # 'PISN': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(61,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        # 'ILOT': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(62,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        # 'CART': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(63,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        'TDE': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(64,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        'AGN': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(70,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        # 'RRLyrae': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(80,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        # 'Mdwarf': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(81,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        # 'Eclip. Bin.': os.path.join(SCRIPT_DIR, '..', 'plots', 'model_8020split_ci()_ns1_c(83,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5'),
#                        }
