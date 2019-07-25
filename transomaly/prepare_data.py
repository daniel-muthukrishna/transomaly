import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import celerite
from celerite import terms

from transomaly import helpers
from transomaly.read_light_curves_from_snana_fits import read_light_curves_from_snana_fits_files


def get_data(class_num, data_dir='data/ZTF_20190512/', passbands=('g', 'r'), nprocesses=1):
    save_lc_filepath = os.path.join(data_dir, "..", "saved_light_curves", f"lc_classnum_{class_num}.pickle")

    if os.path.exists(save_lc_filepath):
        with open(save_lc_filepath, "rb") as fp:  # Unpickling
            light_curves = pickle.load(fp)
    else:
        class_dir = os.path.join(data_dir, 'ZTF_MSIP_MODEL{:02d}'.format(class_num))
        files = os.listdir(class_dir)

        head_files = []
        phot_files = []
        for file in files:
            filepath = os.path.join(data_dir, class_dir, file)
            if filepath.endswith('HEAD.FITS'):
                head_files.append(filepath)
            elif filepath.endswith('PHOT.FITS'):
                phot_files.append(filepath)
            print(filepath)

        light_curves = read_light_curves_from_snana_fits_files(head_files, phot_files, passbands, known_redshift=False, nprocesses=nprocesses)

        with open(save_lc_filepath, "wb") as fp:  # Pickling
            pickle.dump(light_curves, fp)

    return light_curves


def fit_gaussian_process(args):
    lc, objid, passbands, plot = args

    gp_lc = {}
    if plot:
        plt.figure()
    for pbidx, pb in enumerate(passbands):
        time = lc[pb]['time'].dropna()
        flux = lc[pb]['flux'].dropna()
        fluxerr = lc[pb]['fluxErr'].dropna()

        kernel = terms.Matern32Term(log_sigma=0.1, log_rho=0.1)
        gp_lc[pb] = celerite.GP(kernel)
        gp_lc[pb].compute(time, fluxerr)
        # print("Initial log likelihood: {0}".format(gp_lc[pb].log_likelihood(flux)))

        # Optimise parameters
        from scipy.optimize import minimize
        def neg_log_like(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.log_likelihood(y)
        initial_params = gp_lc[pb].get_parameter_vector()
        bounds = gp_lc[pb].get_parameter_bounds()
        r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(flux, gp_lc[pb]))
        gp_lc[pb].set_parameter_vector(r.x)
        # print(r)

        # print("Final log likelihood: {0}".format(gp_lc[pb].log_likelihood(flux)))

        # Plot GP fit
        if plot:
            # Predict with GP
            x = np.linspace(min(time), max(time), 5000)
            pred_mean, pred_var = gp_lc[pb].predict(flux, x, return_var=True)
            pred_std = np.sqrt(pred_var)

            color = {'g': 'tab:green', 'r': "tab:orange"}
            # plt.plot(time, flux, "k", lw=1.5, alpha=0.3)
            plt.errorbar(time, flux, yerr=fluxerr, fmt=".", capsize=0, color=color[pb])
            plt.plot(x, pred_mean, color=color[pb])
            plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color[pb], alpha=0.3,
                             edgecolor="none")

    if plot:
        plt.xlabel("Days since trigger")
        plt.ylabel("Flux")
        plt.savefig(f'/Users/danmuth/PycharmProjects/transomaly/plots/gp_fits/gp_{objid}.pdf')
        plt.close()

    return gp_lc


def save_gps(light_curves, save_dir='data/', class_num=None, passbands=('g', 'r'), plot=False, nprocesses=1):
    save_gp_filepath = os.path.join(save_dir, "saved_light_curves", f"gp_classnum_{class_num}.pickle")

    if os.path.exists(save_gp_filepath):
        with open(save_gp_filepath, "rb") as fp:  # Unpickling
            saved_gp_fits = pickle.load(fp)
    else:
        args_list = []
        for objid, lc in light_curves.items():
            args_list.append((lc, objid, passbands, plot))

        saved_gp_fits = {}
        if nprocesses == 1:
            for args in args_list:
                lc, objid, passbands, plot = args
                saved_gp_fits[objid] = fit_gaussian_process(args)
        else:
            pool = mp.Pool(nprocesses)
            results = pool.map_async(fit_gaussian_process, args_list)
            pool.close()
            pool.join()

            outputs = results.get()
            print('combining results...')
            for i, output in enumerate(outputs):
                print(i, len(outputs))
                saved_gp_fits[objid] = output

    return saved_gp_fits


def get_arrays(data_dir='data'):
    X = np.load(os.path.join(data_dir, "X.npy"), mmap_mode='r')
    y = np.load(os.path.join(data_dir, "y.npy"), mmap_mode='r')
    labels = np.load(os.path.join(data_dir, "labels.npy"), mmap_mode='r')
    timesX = np.load(os.path.join(data_dir, "tinterp.npy"), mmap_mode='r')
    objids_list = np.load(os.path.join(data_dir, "objid.npy"), mmap_mode='r')
    # with open(os.path.join(data_dir, "origlc.npy"), 'rb') as f:
    #     orig_lc = pickle.load(f)
    orig_lc = X

    # Correct shape for keras is (N_objects, N_timesteps, N_passbands) (where N_timesteps is lookback time)
    X = X.swapaxes(2, 1)

    # mask = []
    # X = X.copy()
    # for i in range(len(X)):
    #     for pbidx in range(2):
    #         minX = X[i, :, pbidx].min(axis=0)
    #         maxX = X[i, :, pbidx].max(axis=0)
    #         X[i, :, pbidx] = (X[i, :, pbidx] - minX) / (maxX - minX)
    #         # if (maxX - minX) != 0:
    #         #     mask.append(i)
    #         #     break
    finitemask = ~np.any(np.any(~np.isfinite(X), axis=1), axis=1)
    X = X[finitemask]
    y = y[finitemask]
    timesX = timesX[finitemask]
    objids_list = objids_list[finitemask]
    orig_lc = orig_lc[finitemask]
    labels = labels[finitemask]

    # Use only SNIa
    X = X[labels == 1]
    y = y[labels == 1]
    timesX = timesX[labels == 1]
    objids_list = objids_list[labels == 1]
    orig_lc = orig_lc[labels == 1]
    labels = labels[labels == 1]



    classes = sorted(list(set(labels)))
    sntypes_map = helpers.get_sntypes()
    class_names = [sntypes_map[class_num] for class_num in classes]

    # Count nobjects per class
    for c in classes:
        nobs = len(X[labels == c])
        print(c, nobs)

    # Use class numbers 1,2,3... instead of 1, 3, 13 etc.
    y_indexes = np.copy(y)
    for i, c in enumerate(classes):
        y_indexes[y == c] = i + 1
    y = y_indexes

    y = to_categorical(y)

    X_train, X_test, y_train, y_test, labels_train, labels_test, timesX_train, timesX_test, orig_lc_train, \
    orig_lc_test, objids_train, objids_test = \
        train_test_split(X, y, labels, timesX, orig_lc, objids_list, train_size=0.80, shuffle=False, random_state=42)

    y_train = X_train[:, 1:, :2]
    y_test = X_test[:, 1:, :2]
    X_train = X_train[:, :-1]
    X_test = X_test[:, :-1]
    timesX_train = timesX_train[:, :-1]
    timesX_test = timesX_test[:, :-1]

    return X_train, X_test, y_train, y_test, timesX_train, timesX_test


if __name__ == '__main__':
    nprocesses = 1
    class_num = 1
    light_curves = get_data(class_num, data_dir='/Users/danmuth/PycharmProjects/transomaly/data/ZTF_20190512/',
                            nprocesses=nprocesses)
    gp_lc = save_gps(light_curves, save_dir='/Users/danmuth/PycharmProjects/transomaly/data/', class_num=class_num,
                     passbands=('g', 'r'), plot=True, nprocesses=nprocesses)


