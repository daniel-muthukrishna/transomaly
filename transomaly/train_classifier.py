import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import json
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Input
from keras.layers import LSTM, GRU
from keras.layers import Dropout, BatchNormalization, Activation, TimeDistributed, Masking
import keras.backend as K

from transomaly.prepare_training_set import PrepareTrainingSetArrays
from transomaly.fit_gaussian_processes import save_gps
from transomaly.get_training_data import get_data
from transomaly.loss_functions import mean_squared_error, chisquare_loss, mean_squared_error_over_error
from transomaly import helpers

COLPB = {'g': 'tab:green', 'r': 'tab:red'}
MARKPB = {'g': 'o', 'r': 's', 'z': 'd'}
ALPHAPB = {'g': 0.3, 'r': 1., 'z': 1}


def train_model(X_train, X_test, y_train, y_test, yerr_train, yerr_test, fig_dir='.', epochs=20, retrain=False,
                passbands=('g', 'r'), model_change='', reframe=False):

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

        model.add(Masking(mask_value=0.))

        model.add(LSTM(100, return_sequences=True))
        # model.add(Dropout(0.2, seed=42))
        # model.add(BatchNormalization())

        model.add(LSTM(100, return_sequences=True))
        # model.add(Dropout(0.2, seed=42))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.2, seed=42))

        if reframe is True:
            model.add(LSTM(100))
            # model.add(Dropout(0.2, seed=42))
            # model.add(BatchNormalization())
            # model.add(Dropout(0.2, seed=42))
            model.add(Dense(npb))
        else:
            model.add(LSTM(100, return_sequences=True))
            # model.add(Dropout(0.2, seed=42))
            # model.add(BatchNormalization())
            # model.add(Dropout(0.2, seed=42))
            model.add(TimeDistributed(Dense(npb)))

        if 'chi2' in model_change:
            model.compile(loss=chisquare_loss(), optimizer='adam')
        elif 'mse_oe' in model_change:
            model.compile(loss=mean_squared_error_over_error(), optimizer='adam')
        elif reframe is True:
            model.compile(loss='mse', optimizer='adam')
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


def plot_metrics(model, model_name, X_test, y_test, timesX_test, yerr_test, labels_test, objids_test, passbands, fig_dir, nsamples, data_dir,  save_dir, nprocesses, plot_gp=False, extrapolate_gp=True, reframe=False, plot_name='', npred=49):
    nobjects, ntimesteps, npassbands = X_test.shape
    y_pred = model.predict(X_test)

    if reframe is True:
        X_test = X_test[::npred]
        nobjects, ntimesteps, npassbands = X_test.shape
        y_test = y_test.reshape(nobjects, npred, npassbands)[:,::-1,:]
        y_pred = y_pred.reshape(nobjects, npred, npassbands)[:,::-1,:]
        yerr_test = yerr_test.reshape(nobjects, npred, npassbands)[:,::-1,:]
    else:
        npred = ntimesteps
        # test that it's only using previous data
        plt.figure()
        trial_X = np.copy(X_test[0:1])
        out = model.predict(trial_X)
        trial_X[:, -20:] = np.zeros(trial_X[:, -20:].shape)
        out2 = model.predict(trial_X)
        plt.plot(y_test[0, :, 0], label='original data')
        plt.plot(out[0, :, 0], label='prediction on all data')
        plt.plot(out2[0, :, 0], label='prediction not using last 20 time-steps')
        plt.legend()
        plt.savefig(os.path.join(fig_dir, model_name, "test_using_previous_timesteps_only{}".format(plot_name)))

    # nsamples = 1 ##

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
    with open(os.path.join(fig_dir, model_name, f'model_info{plot_name}.txt'), 'w') as file:
        file.write(f"{model_name}\n")
        file.write(f"Reduced chi-squared: {chi2_reduced_allobjects}\n")
        file.write(f"Median reduced chi-squared: {np.median(chi2_hist)}\n")
    plt.figure()
    plt.hist(chi2_hist, bins=max(100, int(max(chi2_hist)/10)), range=(0, int(np.mean(chi2_hist) + 3*np.std(chi2_hist))))
    plt.legend()
    plt.xlabel("chi-squared")
    plt.savefig(os.path.join(fig_dir, model_name, 'chi_squared_distribution{}.pdf'.format(plot_name)))

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
        argmax = None  #timesX_test[sidx].argmax()  # -1

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
                if reframe:
                    ax1.plot(timesX_test[sidx + s][:-1][:argmax], X_test[sidx + s][:, pbidx][:-1][:argmax], c=COLPB[pb], lw=lw,
                             label=plotlabeltest, marker=marker, markersize=10, alpha=alpha, linestyle='-')
                ax1.plot(timesX_test[sidx+s][1:][-npred:][:argmax], y_test[sidx+s][:, pbidx][:argmax], c=COLPB[pb], lw=lw,
                         label=plotlabeltest, marker=None, markersize=10, alpha=alpha, linestyle='-')
                ax1.plot(timesX_test[sidx+s][1:][-npred:][:argmax], y_pred[sidx+s][:, pbidx][:argmax], c=COLPB[pb], lw=lw,
                         label=plotlabelpred, marker=None, markersize=10, alpha=alpha, linestyle=':')
            ax1.errorbar(lc[pb]['time'].dropna(), lc[pb]['flux'].dropna(), yerr=lc[pb]['fluxErr'].dropna(),
                         fmt=".", capsize=0, color=COLPB[pb], label='_nolegend_')

            if plot_gp is True and nsamples == 1:
                gp_lc[pb].compute(lc[pb]['time'].dropna(), lc[pb]['fluxErr'].dropna())
                pred_mean, pred_var = gp_lc[pb].predict(lc[pb]['flux'].dropna(), timesX_test[sidx+s][:argmax], return_var=True)
                pred_std = np.sqrt(pred_var)
                ax1.fill_between(timesX_test[sidx+s][:argmax], pred_mean + pred_std, pred_mean - pred_std, color=COLPB[pb], alpha=0.3,
                                 edgecolor="none")
        ax1.text(0.05, 0.95, f"$\chi^2 = {round(save_chi2[objids_test[idx]], 3)}$", horizontalalignment='left',
                 verticalalignment='center', transform=ax1.transAxes)

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

        ax2.plot(timesX_test[sidx][1:][-npred:][:argmax][m], anomaly_score_mean, lw=3, marker='o')
        ax2.fill_between(timesX_test[sidx][1:][-npred:][:argmax][m], anomaly_score_mean + anomaly_score_std, anomaly_score_mean - anomaly_score_std, alpha=0.3, edgecolor="none")

        ax1.legend(frameon=True, fontsize=33)
        ax1.set_ylabel("Relative flux")
        ax2.set_ylabel("Anomaly score")
        ax2.set_xlabel("Time since trigger [days]")
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.savefig(os.path.join(fig_dir, model_name, f"lc_{objids_test[sidx]}_{idx}{plot_name}.pdf"))
        plt.close()

    print(model_name)
    print(f"Reduced chi-squared for model is {chi2_reduced_allobjects}")
    print(f"Median reduced chi-squared for model is {np.median(chi2_hist)}")


def similarity_metric(model, X_test, y_test, yerr_test, labels_test, objids_test, nsamples):
    nobjects, ntimesteps, npassbands = X_test.shape
    y_pred = model.predict(X_test)
    class_nums = np.unique(labels_test)
    sntypes_map = helpers.get_sntypes()
    class_names = [sntypes_map[class_num] for class_num in class_nums]

    anomaly_scores = {key: [] for key in class_names}
    for idx in range(nobjects):
        sidx = idx * nsamples  # Assumes like samples are in order
        argmax = None  # timesX_test[sidx].argmax()  # -1

        # Get anomaly scores
        chi2_samples = []
        for s in range(nsamples):
            chi2 = np.zeros(ntimesteps)
            npb = 0
            for pbidx in range(npassbands):
                m = yerr_test[sidx + s, :, pbidx][:argmax] != 0  # ignore zeros (where no data exists)
                yt = y_test[sidx + s, :, pbidx][:argmax][m]
                yp = y_pred[sidx + s, :, pbidx][:argmax][m]
                ye = yerr_test[sidx + s, :, pbidx][:argmax][m]
                try:
                    chi2 += ((yp - yt) / ye) ** 2
                    npb += 1
                except ValueError as e:
                    print(f"Failed chi2 object {objids_test[sidx + s]}", e)
            chi2_samples.append(chi2 / npb)
        anomaly_score_samples = chi2_samples
        anomaly_score_mean = np.mean(anomaly_score_samples, axis=0)
        anomaly_score_std = np.std(anomaly_score_samples, axis=0)
        anomaly_score_max = max(anomaly_score_mean)

        class_name = sntypes_map[labels_test[sidx]]
        anomaly_scores[class_name].append(anomaly_score_max)

    similarity_score = {key: [] for key in class_names}
    similarity_score_std = {key: [] for key in class_names}
    for c in class_names:
        similarity_score[c] = np.median(anomaly_scores[c])
        similarity_score_std[c] = np.std(anomaly_scores[c])

    return similarity_score, similarity_score_std


def get_similarity_matrix(class_nums, model_filepaths, preparearrays, nprocesses, extrapolate_gp, nsamples, ignore_class_names_test_on=[]):

    X_train, X_test, y_train, y_test, Xerr_train, Xerr_test, yerr_train, yerr_test, \
    timesX_train, timesX_test, labels_train, labels_test, objids_train, objids_test = \
        preparearrays.make_training_set(class_nums=class_nums, nsamples=1, otherchange='getKnAndOtherTypes', nprocesses=nprocesses, extrapolate_gp=extrapolate_gp, reframe=False, npred=49)

    similarity_matrix = {classnum: [] for classnum in model_filepaths.keys()}
    similarity_matrix_std = {classnum: [] for classnum in model_filepaths.keys()}
    for class_name, model_filepath in model_filepaths.items():
        print(class_name)
        if not os.path.exists(model_filepath):
            print("No model found at", model_filepath)
            continue

        saved_scores_fp = os.path.join(os.path.dirname(model_filepath), f'similarity_scores_{class_nums}.json')

        if os.path.exists(saved_scores_fp):
            print("Using saved similarity scores")
            with open(saved_scores_fp, 'r') as fp:
                similarity_score = json.load(fp)
            with open(saved_scores_fp.replace('similarity_scores_', 'similarity_scores_std_'), 'r') as fp:
                similarity_score_std = json.load(fp)
        else:
            print("Saving similarity scores...")
            if 'chi2' in model_filepath:
                model = load_model(model_filepath, custom_objects={'loss': chisquare_loss()})
            elif 'mse_oe' in model_filepath:
                model = load_model(model_filepath, custom_objects={'loss': mean_squared_error_over_error()})
            else:
                model = load_model(model_filepath, custom_objects={'loss': mean_squared_error()})

            similarity_score, similarity_score_std = similarity_metric(model, X_test, y_test, yerr_test, labels_test,
                                                                       objids_test, nsamples)
            with open(saved_scores_fp, 'w') as fp:
                json.dump(similarity_score, fp)
            with open(saved_scores_fp.replace('similarity_scores_', 'similarity_scores_std_'), 'w') as fp:
                json.dump(similarity_score_std, fp)

        similarity_matrix[class_name] = similarity_score
        similarity_matrix_std[class_name] = similarity_score_std

    similarity_matrix = pd.DataFrame(similarity_matrix)
    similarity_matrix_std = pd.DataFrame(similarity_matrix_std)

    similarity_matrix.to_csv('similarity_matrix.csv')
    similarity_matrix_std.to_csv('similarity_matrix_std.csv')

    print(similarity_matrix)

    similarity_matrix = similarity_matrix.drop(ignore_class_names_test_on)

    return similarity_matrix, similarity_matrix_std



def plot_similarity_scatter_plot(similarity_matrix):
    font = {'family': 'normal',
            'size': 10}
    matplotlib.rc('font', **font)
    CLASS_COLOR = {'SNIa-norm': 'tab:green', 'SNIbc': 'tab:orange', 'SNII': 'tab:blue', 'SNIIn': 'blue',
                   'SNIa-91bg': 'tab:red', 'SNIa-x': 'tab:purple', 'point-Ia': 'tab:brown', 'Kilonova': '#aaffc3',
                   'SLSN-I': 'tab:olive', 'PISN': 'tab:cyan', 'ILOT': '#FF1493', 'CART': 'navy', 'TDE': 'tab:pink',
                   'AGN': 'bisque'}

    xrange, yrange = similarity_matrix.shape

    # similarity_matrix = np.log(similarity_matrix)
    # similarity_matrix=(similarity_matrix-similarity_matrix.min())/(similarity_matrix.max()-similarity_matrix.min())
    fig, ax = plt.subplots()
    for j in range(yrange):
        yname = similarity_matrix.columns.values[j]
        for i in range(xrange):
            xname = similarity_matrix.index.values[i]
            xplot, yplot = similarity_matrix[yname][xname], j
            plt.scatter(xplot, yplot, s=100, color=CLASS_COLOR[xname])
            plt.annotate(xname, xy=(xplot, yplot), fontsize=10)
    ax.yaxis.set_major_locator(plt.MaxNLocator(yrange))
    ax.set_yticklabels(np.insert(similarity_matrix.columns.values, 0, 0))
    # ax.xaxis.set_major_formatter(plt.NullFormatter())
    plt.show()




def plot_similarity_matrix(similarity_matrix, similarity_matrix_std):
    font = {'family': 'normal',
            'size': 36}
    matplotlib.rc('font', **font)

    xrange, yrange = similarity_matrix.shape
    similarity_matrix = similarity_matrix.T
    similarity_matrix = similarity_matrix[
        ['SNIa', 'SNIa-x', 'SNII', 'SNIbc', 'SLSN-I', 'TDE', 'AGN', 'SNIIn', 'Ia-91bg', 'CART', 'TDE', 'PISN',
         'Kilonova']]
    xlabels = similarity_matrix.columns.values
    ylabels = similarity_matrix.index.values

    maxval = min(20, similarity_matrix.values.max())
    plt.figure(figsize=(15,12))
    plt.imshow(similarity_matrix, cmap=plt.cm.RdBu_r, vmin=0, vmax=maxval)#, norm=colors.LogNorm())

    cb = plt.colorbar()
    # cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=27)
    plt.xticks(np.arange(xrange), xlabels, rotation=90, fontsize=27)
    plt.yticks(np.arange(yrange), ylabels, fontsize=27)

    thresh_q3 = 0.75 * maxval
    thresh_q1 = 0.25 * maxval
    for i in range(xrange):
        for j in range(yrange):
            c = similarity_matrix.iloc[j, i]
            if c > 100:
                cell_text = f"{c:.0f}"
            elif c > 10:
                cell_text = f"{c:.1f}"
            else:
                cell_text = f"{c:.2f}"
            plt.text(i, j, cell_text, va='center', ha='center',
                     color="white" if c < thresh_q1 or c > thresh_q3 else "black", fontsize=14)

    plt.ylabel('Trained on')
    plt.xlabel('Tested on')
    plt.tight_layout()
    print("Saving matrix plot...")
    plt.savefig("similarity_matrix.pdf")




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
    otherchange = '8020split'  # 'singleobject_1_50075859_gp_samples_extrapolated_gp'  # '8020split' #  #'5050testvalidation' #
    nsamples = 1  # 5000
    extrapolate_gp = True
    redo = False
    train_epochs = 150
    retrain = False
    reframe_problem = False
    npred = 49
    nn_architecture_change = 'unnormalised_{}mse_predict_last{}_timesteps_nodropout_100lstmneurons'.format('reframe_Xy_' if reframe_problem else '', npred)  # 'normalise_mse_withmasking_1000lstmneurons'  # 'chi2'  # 'mse'

    fig_dir = os.path.join(fig_dir, "model_{}_ci{}_ns{}_c{}".format(otherchange, contextual_info, nsamples, class_nums))
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    preparearrays = PrepareTrainingSetArrays(passbands, contextual_info, data_dir, save_dir, training_set_dir, redo)
    X_train, X_test, y_train, y_test, Xerr_train, Xerr_test, yerr_train, yerr_test, \
    timesX_train, timesX_test, labels_train, labels_test, objids_train, objids_test = \
        preparearrays.make_training_set(class_nums, nsamples, otherchange, nprocesses, extrapolate_gp, reframe=reframe_problem, npred=npred)

    model, model_name = train_model(X_train, X_test, y_train, y_test, yerr_train, yerr_test, fig_dir=fig_dir, epochs=train_epochs,
                        retrain=retrain, passbands=passbands, model_change=nn_architecture_change, reframe=reframe_problem)

    plot_metrics(model, model_name, X_test, y_test, timesX_test, yerr_test, labels_test, objids_test, passbands=passbands,
                 fig_dir=fig_dir, nsamples=nsamples, data_dir=data_dir, save_dir=save_dir, nprocesses=nprocesses, plot_gp=True, extrapolate_gp=extrapolate_gp, reframe=reframe_problem, plot_name='', npred=npred)

    plot_metrics(model, model_name, X_train, y_train, timesX_train, yerr_train, labels_train, objids_train, passbands=passbands,
                 fig_dir=fig_dir, nsamples=nsamples, data_dir=data_dir, save_dir=save_dir, nprocesses=nprocesses, plot_gp=True, extrapolate_gp=extrapolate_gp, reframe=reframe_problem, plot_name='_training_set', npred=npred)

    # Test on other classes  #51,60,62,70 AndOtherTypes
    X_train, X_test, y_train, y_test, Xerr_train, Xerr_test, yerr_train, yerr_test, \
    timesX_train, timesX_test, labels_train, labels_test, objids_train, objids_test = \
        preparearrays.make_training_set(class_nums=(51,60,62,70,80), nsamples=1, otherchange='getKnAndOtherTypes', nprocesses=nprocesses, extrapolate_gp=extrapolate_gp, reframe=reframe_problem, npred=npred)
    plot_metrics(model, model_name, X_train, y_train, timesX_train, yerr_train, labels_train, objids_train, passbands=passbands,
                 fig_dir=fig_dir, nsamples=nsamples, data_dir=data_dir, save_dir=save_dir, nprocesses=nprocesses, plot_gp=True, extrapolate_gp=extrapolate_gp, reframe=reframe_problem, plot_name='anomaly', npred=npred)


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
