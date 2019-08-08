import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Input
from keras.layers import LSTM, GRU
from keras.layers import Dropout, BatchNormalization, Activation, TimeDistributed
import keras.backend as K

from transomaly.prepare_arrays import PrepareTrainingSetArrays
from transomaly.fit_gaussian_processes import get_data, save_gps

COLPB = {'g': 'tab:green', 'r': 'tab:red'}
MARKPB = {'g': 'o', 'r': 's', 'z': 'd'}
ALPHAPB = {'g': 0.3, 'r': 1., 'z': 1}


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


def train_model(X_train, X_test, y_train, y_test, yerr_train, yerr_test, fig_dir='.', epochs=20, retrain=False,
                passbands=('g', 'r'), model_change=''):

    model_name = f"keras_model_epochs{epochs}_{model_change}"
    model_filename = os.path.join(fig_dir, model_name, f"{model_name}.hdf5")
    if not os.path.exists(os.path.join(fig_dir, model_name)):
        os.makedirs(os.path.join(fig_dir, model_name))

    npb = len(passbands)

    if not retrain and os.path.isfile(model_filename):
        if 'chi2' in model_change:
            model = load_model(model_filename, custom_objects={'loss': chisquare_loss()})
        elif 'mse_oe'in model_change:
            model = load_model(model_filename, custom_objects={'loss': mean_squared_error_over_error()})
        else:
            model = load_model(model_filename, custom_objects={'loss': mean_squared_error()})
    else:
        model = Sequential()

        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.4, seed=42))
        model.add(BatchNormalization())

        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.4, seed=42))
        model.add(BatchNormalization())
        model.add(Dropout(0.4, seed=42))

        model.add(LSTM(100, return_sequences=True))
        model.add(Dropout(0.4, seed=42))
        model.add(BatchNormalization())
        model.add(Dropout(0.4, seed=42))

        model.add(TimeDistributed(Dense(npb)))
        if 'chi2' in model_change:
            model.compile(loss=chisquare_loss(), optimizer='adam')
        elif 'mse_oe' in model_change:
            model.compile(loss=mean_squared_error_over_error(), optimizer='adam')
        else:
            model.compile(loss=mean_squared_error(), optimizer='adam')
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, verbose=2)

        print(model.summary())
        model.save(model_filename)

        # Plot loss vs epochs
        plt.figure()
        trainloss = history.history['loss']
        valloss = history.history['val_loss']
        plt.plot(trainloss)
        plt.plot(valloss)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(model_filename.replace('.hdf5', '.pdf'))
        # Plot zoomed figure
        lenloss = len(trainloss)
        zoomloss = int(lenloss/2.)
        plt.figure()
        plt.plot(np.arange(zoomloss, lenloss), trainloss[zoomloss:])
        plt.plot(np.arange(zoomloss, lenloss), valloss[zoomloss:])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f"{model_filename.replace('.hdf5', '_zoomed.pdf')}")

    return model, model_name


def plot_metrics(model, model_name, X_test, y_test, timesX_test, yerr_test, labels_test, objids_test, passbands, fig_dir, nsamples, data_dir,  save_dir, nprocesses, plot_gp=False, extrapolate_gp=True):
    nobjects, ntimesteps, npassbands = yerr_test.shape
    y_pred = model.predict(X_test)

    # Get reduced chi_squared
    chi2_hist = []
    chi2_reduced_allobjects = 0
    reduce_count = 0
    save_object_chi2 = []
    save_chi2 = {}
    for idx in range(nobjects):
        for pbidx, pb in enumerate(passbands):
            m = yerr_test[idx, :, pbidx] != 0  # ignore zeros (where no data exists)
            yt = y_test[idx, :, pbidx][m]
            yp = y_pred[idx, :, pbidx][m]
            ye = yerr_test[idx, :, pbidx][m]
            if len(yt) == 0:
                reduce_count += 1
                print(f"No values for {objids_test[idx]} {pb}-band")
                continue
            chi2 = sum(((yt - yp) / ye) ** 2)
            chi2_reduced = chi2 / len(yt)
            chi2_reduced_allobjects += chi2_reduced
        chi2_hist.append(chi2_reduced/npassbands)
        save_object_chi2.append((idx, objids_test[idx], chi2_reduced/npassbands))
        save_chi2[objids_test[idx]] = chi2_reduced/npassbands
    chi2_reduced_allobjects = chi2_reduced_allobjects / ((nobjects * npassbands) - reduce_count)
    print(f"Reduced chi-squared for model is {chi2_reduced_allobjects}")
    print(f"Median reduced chi-squared for model is {np.median(chi2_hist)}")
    with open(os.path.join(fig_dir, model_name, 'model_info.txt'), 'w') as file:
        file.write(f"{model_name}\n")
        file.write(f"Reduced chi-squared: {chi2_reduced_allobjects}\n")
        file.write(f"Median reduced chi-squared: {np.median(chi2_hist)}\n")
    plt.figure()
    plt.hist(chi2_hist, bins=max(100, int(max(chi2_hist)/10)), range=(0, int(np.mean(chi2_hist) + 3**np.std(chi2_hist))))
    plt.legend()
    plt.xlabel("chi-squared")
    plt.savefig(os.path.join(fig_dir, model_name, 'chi_squared_distribution.pdf'))
    plt.show()

    save_object_chi2 = sorted(save_object_chi2, key=lambda x: x[2])
    print(save_object_chi2[:100])

    # Get raw light curve data
    light_curves = {}
    gp_fits = {}
    for classnum in np.unique(labels_test):
        print(f"Getting lightcurves for class:{classnum}")
        light_curves[classnum] = get_data(classnum, data_dir, save_dir, nprocesses)
        if plot_gp is True and nsamples == 1:
            gp_fits[classnum] = save_gps(light_curves, save_dir, classnum, passbands, plot=False,
                                     nprocesses=nprocesses, redo=False, extrapolate=extrapolate_gp)

    # Plot predictions vs time per class
    font = {'family': 'normal',
            'size': 36}
    matplotlib.rc('font', **font)

    for idx in np.arange(0, 100):
    # for (idx, oid, chi2_red) in save_object_chi2[:100]:
    #     print(idx, oid, chi2_red)
        sidx = idx * nsamples  # Assumes like samples are in order
        print("Plotting example vs time", idx, objids_test[sidx])
        argmax = -1  #timesX_test[sidx].argmax()  # -1

        # Get raw light curve observations
        lc = light_curves[labels_test[sidx]][objids_test[sidx]]
        if plot_gp is True and nsamples == 1:
            gp_lc = gp_fits[labels_test[sidx]][objids_test[sidx]]

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 15), sharex=True)

        for pbidx, pb in enumerate(passbands):
            for s in range(nsamples):
                lw = 3 if s == 0 else 0.5
                alpha = 1 if s == 0 else 0.1
                plotlabeltest = "ytest:{}".format(pb) if s == 0 else ''
                plotlabelpred = "ypred:{}".format(pb) if s == 0 else ''
                marker = None  # MARKPB[pb] if s == 0 else None
                ax1.plot(timesX_test[sidx+s][:argmax], y_test[sidx+s][:, pbidx][:argmax], c=COLPB[pb], lw=lw,
                         label=plotlabeltest, marker=marker, markersize=10, alpha=alpha, linestyle='-')
                ax1.plot(timesX_test[sidx+s][:argmax], y_pred[sidx+s][:, pbidx][:argmax], c=COLPB[pb], lw=lw,
                         label=plotlabelpred, marker=marker, markersize=10, alpha=alpha, linestyle=':')
            ax1.errorbar(lc[pb]['time'].dropna(), lc[pb]['flux'].dropna(), yerr=lc[pb]['fluxErr'].dropna(),
                         fmt=".", capsize=0, color=COLPB[pb], label='_nolegend_')
            ax1.text(0.05, 0.95, f"$\chi^2 = {round(save_chi2[objids_test[idx]], 3)}$", horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
            if plot_gp is True and nsamples == 1:
                gp_lc[pb].compute(lc[pb]['time'].dropna(), lc[pb]['fluxErr'].dropna())
                pred_mean, pred_var = gp_lc[pb].predict(lc[pb]['flux'].dropna(), timesX_test[sidx+s][:argmax], return_var=True)
                pred_std = np.sqrt(pred_var)
                ax1.fill_between(timesX_test[sidx+s][:argmax], pred_mean + pred_std, pred_mean - pred_std, color=COLPB[pb], alpha=0.3,
                                 edgecolor="none")

        # Plot anomaly scores
        chi2_samples = []
        for s in range(nsamples):
            chi2 = 0
            for pbidx in range(npassbands):
                m = yerr_test[sidx+s, :, pbidx][:argmax] != 0  # ignore zeros (where no data exists)
                yt = y_test[sidx+s, :, pbidx][:argmax][m]
                yp = y_pred[sidx+s, :, pbidx][:argmax][m]
                ye = yerr_test[sidx+s, :, pbidx][:argmax][m]
                try:
                    chi2 += ((yp - yt)/ye)**2
                except ValueError as e:
                    pbidx -= 1
                    m = yerr_test[sidx + s, :, pbidx][:argmax] != 0
                    print(f"Failed chi2 object {objids_test[sidx+s]}", e)
            chi2_samples.append(chi2 / npassbands)
        anomaly_score_samples = chi2_samples
        anomaly_score_mean = np.mean(anomaly_score_samples, axis=0)
        anomaly_score_std = np.std(anomaly_score_samples, axis=0)
        ax2.text(0.05, 0.95, f"$\chi^2 = {round(np.sum(anomaly_score_mean)/len(yt), 3)}$", horizontalalignment='left',
                 verticalalignment='center', transform=ax2.transAxes)

        ax2.plot(timesX_test[sidx][:argmax][m], anomaly_score_mean, lw=3)
        ax2.fill_between(timesX_test[sidx][:argmax][m], anomaly_score_mean + anomaly_score_std, anomaly_score_mean - anomaly_score_std, alpha=0.3, edgecolor="none")

        ax1.legend(frameon=True, fontsize=33)
        ax1.set_ylabel("Relative flux")
        ax2.set_ylabel("Anomaly score")
        ax2.set_xlabel("Time since trigger [days]")
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)

        plt.savefig(os.path.join(fig_dir, model_name, f"lc_{objids_test[sidx]}_{idx}.pdf"))
        plt.close()

    print(model_name)
    print(f"Reduced chi-squared for model is {chi2_reduced_allobjects}")
    print(f"Median reduced chi-squared for model is {np.median(chi2_hist)}")


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(SCRIPT_DIR, '..', 'data/ZTF_20190512')
    save_dir = os.path.join(SCRIPT_DIR, '..', 'data/saved_light_curves')
    training_set_dir = os.path.join(SCRIPT_DIR, '..', 'data/training_set_files')
    fig_dir = os.path.join(SCRIPT_DIR, '..', 'plots')
    passbands = ('g', 'r')
    contextual_info = ()
    nprocesses = None
    class_nums = (1,)
    otherchange = 'singleobject_1_99285690_extrapolatedgp'  #'5050testvalidation' #
    nsamples = 5000
    extrapolate_gp = False
    redo = False
    train_epochs = 5
    retrain = False
    nn_architecture_change = 'mse'  # 'chi2'  # 'mse'

    fig_dir = os.path.join(fig_dir, "model_{}_ci{}_ns{}_c{}".format(otherchange, contextual_info, nsamples, class_nums))
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    preparearrays = PrepareTrainingSetArrays(passbands, contextual_info, data_dir, save_dir, training_set_dir, redo)
    X_train, X_test, y_train, y_test, Xerr_train, Xerr_test, yerr_train, yerr_test, \
    timesX_train, timesX_test, labels_train, labels_test, objids_train, objids_test = \
        preparearrays.make_training_set(class_nums, nsamples, otherchange, nprocesses, extrapolate_gp)

    model, model_name = train_model(X_train, X_test, y_train, y_test, yerr_train, yerr_test, fig_dir=fig_dir, epochs=train_epochs,
                        retrain=retrain, passbands=passbands, model_change=nn_architecture_change)

    plot_metrics(model, model_name, X_test, y_test, timesX_test, yerr_test, labels_test, objids_test, passbands=passbands,
                 fig_dir=fig_dir, nsamples=nsamples, data_dir=data_dir, save_dir=save_dir, nprocesses=nprocesses, plot_gp=True, extrapolate_gp=extrapolate_gp)

    #X_train, y_train, timesX_train, yerr_train, labels_train, objids_train
    #X_test, y_test, timesX_test, yerr_test, labels_test, objids_test


if __name__ == '__main__':
    main()
