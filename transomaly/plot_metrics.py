import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import json

from transomaly.fit_gaussian_processes import save_gps
from astrorapid.get_training_data import get_data
from transomaly import helpers

from astropy.stats import median_absolute_deviation

# matplotlib.use('TkAgg')


COLPB = {'g': 'tab:green', 'r': 'tab:red', 'gpred': 'turquoise', 'rpred': 'tab:pink'}
MARKPB = {'g': 'o', 'r': 's', 'z': 'd'}
ALPHAPB = {'g': 0.3, 'r': 1., 'z': 1}


def plot_history(history, model_filename):
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
    zoomloss = int(lenloss / 2.)
    plt.figure()
    plt.plot(np.arange(zoomloss, lenloss), trainloss[zoomloss:])
    plt.plot(np.arange(zoomloss, lenloss), valloss[zoomloss:])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.ylim(bottom=min(valloss)-abs(0.1*min(valloss)), top=1.1*max(np.array(valloss[zoomloss:])[abs(valloss[zoomloss:]-np.median(valloss[zoomloss:])) < 5 * median_absolute_deviation(valloss[zoomloss:])]))
    plt.savefig(f"{model_filename.replace('.hdf5', '_zoomed.pdf')}")
    # Plot zoomed figure reduced y axis
    lenloss = len(trainloss)
    zoomloss = int(0.75*lenloss)
    plt.figure()
    plt.plot(np.arange(zoomloss, lenloss), trainloss[zoomloss:])
    plt.plot(np.arange(zoomloss, lenloss), valloss[zoomloss:])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.ylim(bottom=min(valloss)-abs(0.1*min(valloss)), top=1.1*max(np.array(valloss[zoomloss:])[abs(valloss[zoomloss:]-np.median(valloss[zoomloss:])) < 5 * median_absolute_deviation(valloss[zoomloss:])]))
    plt.savefig(f"{model_filename.replace('.hdf5', '_zoomed2.pdf')}")


def plot_metrics(model, model_name, X_test, y_test, timesX_test, yerr_test, labels_test, objids_test, passbands,
                 fig_dir, nsamples, data_dir, save_dir, nprocesses, plot_gp=False, extrapolate_gp=True, reframe=False,
                 plot_name='', npred=49, probabilistic=False, known_redshift=False, get_data_func=None, normalise=False,
                 bayesian=False):
    print(model_name)
    nobjects, ntimesteps, nfeatures = X_test.shape
    npassbands = len(passbands)

    sampled_ypred = []
    sampled_ystd = []
    draws = []
    if probabilistic:
        X_test = np.asarray(X_test, np.float32)
        y_test = np.asarray(y_test, np.float32)
        yhat = model(X_test)
        y_pred = np.asarray(yhat.mean())
        y_pred_std = np.asarray(yhat.stddev())
        if bayesian:
            ns = 100
            for i in range(ns):
                sampled_yhat = model(X_test)
                sampled_ypred.append(np.asarray(sampled_yhat.mean()))
                sampled_ystd.append(np.asarray(sampled_yhat.stddev()))
                draws.append(np.random.normal(sampled_yhat.mean(), sampled_yhat.stddev()))
            # plot_mean_ypred = np.mean(np.array(sampled_ypred), axis=0)
            # plot_sigma_ypred = np.std(np.array(sampled_ypred), axis=0)
            plot_mean_ypred = np.mean(np.array(draws), axis=0)
            plot_sigma_ypred = np.std(np.array(draws), axis=0)
        else:
            yhat = model(X_test)
            plot_mean_ypred = np.asarray(yhat.mean())
            plot_sigma_ypred = np.asarray(yhat.stddev())
    else:
        y_pred = model.predict(X_test)

    if not reframe:
        npred = ntimesteps

    # Get raw light curve data
    light_curves = {}
    gp_fits = {}
    for classnum in np.unique(labels_test):
        print(f"Getting lightcurves for class:{classnum}")
        light_curves[classnum] = get_data(get_data_func=get_data_func, class_num=classnum, data_dir=data_dir,
                                          save_dir=save_dir, passbands=passbands, known_redshift=known_redshift,
                                          nprocesses=nprocesses, redo=False, calculate_t0=False)
        if plot_gp is True and nsamples == 1:
            gp_fits[classnum] = save_gps(light_curves, save_dir, classnum, passbands, plot=False,
                                         nprocesses=nprocesses, redo=False, extrapolate=extrapolate_gp)

    # # Plot predictions vs time per class
    # font = {'family': 'normal',
    #         'size': 36}
    # matplotlib.rc('font', **font)

    for idx in np.arange(0, 10):
        sidx = idx * nsamples  # Assumes like samples are in order
        print("Plotting example vs time", idx, objids_test[sidx])
        argmax = None  # timesX_test[sidx].argmax()  # -1

        # Get raw light curve observations
        lc = light_curves[labels_test[sidx]][objids_test[sidx]]
        if plot_gp is True and nsamples == 1:
            gp_lc = gp_fits[labels_test[sidx]][objids_test[sidx]]

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 15), sharex=True)

        for pbidx, pb in enumerate(passbands):
            pbmask = lc['passband'] == pb

            for s in range(1):  # nsamples):
                lw = 3 if s == 0 else 0.5
                alpha = 1 if s == 0 else 0.1
                plotlabeltest = "ytest:{}".format(pb) if s == 0 else ''
                plotlabelpred = "ypred:{}".format(pb) if s == 0 else ''
                marker = None  # MARKPB[pb] if s == 0 else None
                if reframe:
                    ax1.plot(timesX_test[sidx + s][:-1][:argmax], X_test[sidx + s][:, pbidx][:-1][:argmax], c=COLPB[pb],
                             lw=lw,
                             label=plotlabeltest, marker=marker, markersize=10, alpha=alpha, linestyle='-')
                ax1.plot(timesX_test[sidx + s][1:][-npred:][:argmax], y_test[sidx + s][:, pbidx][:argmax], c=COLPB[pb],
                         lw=lw,
                         label=plotlabeltest, marker='o', markersize=10, alpha=alpha, linestyle='-')
                if probabilistic:
                    if bayesian:
                        for sp in range(ns):
                            ax1.errorbar(timesX_test[sidx + s][1:][-npred:][:argmax],
                                         sampled_ypred[sp][sidx + s][:, pbidx][:argmax],
                                         yerr=sampled_ystd[sp][sidx + s][:, pbidx][:argmax],
                                         color=COLPB[f'{pb}pred'], lw=0.5, marker='*', markersize=10, alpha=1 / 256,
                                         linestyle=':')
                    ax1.errorbar(timesX_test[sidx + s][1:][-npred:][:argmax],
                                 plot_mean_ypred[sidx + s][:, pbidx][:argmax],
                                 yerr=plot_sigma_ypred[sidx + s][:, pbidx][:argmax],
                                 color=COLPB[f'{pb}pred'], lw=lw, label=plotlabelpred, marker='x', markersize=20,
                                 alpha=1, linestyle=':')

                else:
                    ax1.plot(timesX_test[sidx + s][1:][-npred:][:argmax], y_pred[sidx + s][:, pbidx][:argmax],
                             c=COLPB[f'{pb}pred'], lw=lw,
                             label=plotlabelpred, marker='*', markersize=10, alpha=alpha, linestyle=':')

        if not normalise:
            ax1.errorbar(lc[pbmask]['time'].data, lc[pbmask]['flux'].data, yerr=lc[pbmask]['fluxErr'].data,
                         fmt="x", capsize=0, color=COLPB[pb], label='_nolegend_', markersize=15, )

            if plot_gp is True and nsamples == 1:
                gp_lc[pb].compute(lc[pbmask]['time'].data, lc[pbmask]['fluxErr'].data)
                pred_mean, pred_var = gp_lc[pb].predict(lc[pbmask]['flux'].data, timesX_test[sidx + s][:argmax],
                                                        return_var=True)
                pred_std = np.sqrt(pred_var)
                ax1.fill_between(timesX_test[sidx + s][:argmax], pred_mean + pred_std, pred_mean - pred_std,
                                 color=COLPB[pb],
                                 alpha=0.05,
                                 edgecolor="none")
            # ax1.text(0.05, 0.95, f"$\chi^2 = {round(save_chi2[objids_test[idx]], 3)}$", horizontalalignment='left',
            #          verticalalignment='center', transform=ax1.transAxes)
            plt.xlim(-70, 80)

        # Plot anomaly scores
        chi2_samples = []
        for s in range(1):  # nsamples):
            chi2 = 0
            for pbidx in range(npassbands):
                m = yerr_test[sidx + s, :, pbidx][:argmax] != 0  # ignore zeros (where no data exists)
                yt = y_test[sidx + s, :, pbidx][:argmax][m]
                yp = y_pred[sidx + s, :, pbidx][:argmax][m]
                ye = yerr_test[sidx + s, :, pbidx][:argmax][m]
                try:
                    chi2 += ((yp - yt) / ye) ** 2
                except ValueError as e:
                    pbidx -= 1
                    m = yerr_test[sidx + s, :, pbidx][:argmax] != 0
                    print(f"Failed chi2 object {objids_test[sidx + s]}", e)
            chi2_samples.append(chi2 / npassbands)
        anomaly_score_samples = chi2_samples
        anomaly_score_mean = np.mean(anomaly_score_samples, axis=0)
        anomaly_score_std = np.std(anomaly_score_samples, axis=0)
        ax2.text(0.05, 0.95, f"$\chi^2 = {round(np.sum(anomaly_score_mean) / len(yt), 3)}$", horizontalalignment='left',
                 verticalalignment='center', transform=ax2.transAxes)

        ax2.plot(timesX_test[sidx][1:][-npred:][:argmax][m], anomaly_score_mean, lw=3, marker='o')
        ax2.fill_between(timesX_test[sidx][1:][-npred:][:argmax][m], anomaly_score_mean + anomaly_score_std,
                         anomaly_score_mean - anomaly_score_std, alpha=0.3, edgecolor="none")

        ax1.legend(frameon=True, fontsize=33)
        ax1.set_ylabel("Relative flux")
        ax2.set_ylabel("Anomaly score")
        ax2.set_xlabel("Time since trigger [days]")
        plt.tight_layout()
        fig.subplots_adjust(hspace=0)
        plt.savefig(os.path.join(fig_dir, model_name, f"lc_{objids_test[sidx]}_{idx}{plot_name}.pdf"))
        plt.close()

    print(model_name)


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

