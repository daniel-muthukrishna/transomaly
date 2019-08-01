import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import celerite
from celerite import terms
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')

from transomaly.read_light_curves_from_snana_fits import read_light_curves_from_snana_fits_files


def get_data(class_num, data_dir='data/ZTF_20190512/', save_dir='data/saved_light_curves/', passbands=('g', 'r'),
             nprocesses=1, redo=False):
    save_lc_filepath = os.path.join(save_dir, f"lc_classnum_{class_num}.pickle")

    if os.path.exists(save_lc_filepath) and not redo:
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


def combined_neg_log_like(params, fluxes, gp_lcs, passbands):
    loglike = 0
    for pb in passbands:
        gp_lcs[pb].set_parameter_vector(params)
        y = fluxes[pb]
        loglike += gp_lcs[pb].log_likelihood(y)

    return -loglike


def fit_gaussian_process(args):
    lc, objid, passbands, plot = args

    gp_lc = {}
    if plot:
        plt.figure()

    kernel = terms.Matern32Term(log_sigma=5., log_rho=3.)
    times, fluxes, fluxerrs = {}, {}, {}
    for pbidx, pb in enumerate(passbands):
        times[pb] = lc[pb]['time'].dropna()
        fluxes[pb] = lc[pb]['flux'].dropna()
        fluxerrs[pb] = lc[pb]['fluxErr'].dropna()

        gp_lc[pb] = celerite.GP(kernel)
        gp_lc[pb].compute(times[pb], fluxerrs[pb])
        print("Initial log likelihood: {0}".format(gp_lc[pb].log_likelihood(fluxes[pb])))
        initial_params = gp_lc[pb].get_parameter_vector()  # This should be the same across passbands
        bounds = gp_lc[pb].get_parameter_bounds()

    # Optimise parameters
    try:
        r = minimize(combined_neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(fluxes, gp_lc, passbands))
        print(r)
    except Exception as e:
        print("Failed object", objid, e)
        return

    for pbidx, pb in enumerate(passbands):
        gp_lc[pb].set_parameter_vector(r.x)
        time = times[pb]
        flux = fluxes[pb]
        fluxerr = fluxerrs[pb]

        print("Final log likelihood: {0}".format(gp_lc[pb].log_likelihood(flux)))

        # Remove objects with bad fits
        x = np.linspace(min(time), max(time), 5000)
        pred_mean, pred_var = gp_lc[pb].predict(flux, x, return_var=True)
        if np.any(~np.isfinite(pred_mean)) or gp_lc[pb].log_likelihood(flux) < -380:
            print("Bad fit for object", objid)
            return

        # Plot GP fit
        if plot:
            # Predict with GP
            x = np.linspace(min(time), max(time), 5000)
            pred_mean, pred_var = gp_lc[pb].predict(flux, x, return_var=True)
            pred_std = np.sqrt(pred_var)

            color = {'g': 'tab:green', 'r': "tab:red"}
            # plt.plot(time, flux, "k", lw=1.5, alpha=0.3)
            plt.errorbar(time, flux, yerr=fluxerr, fmt=".", capsize=0, color=color[pb])
            plt.plot(x, pred_mean, color=color[pb])
            plt.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, color=color[pb], alpha=0.3,
                             edgecolor="none")

    # TODO: share hyperparameters across passband light curves

    if plot:
        plt.xlabel("Days since trigger")
        plt.ylabel("Flux")
        plt.savefig(f'/Users/danmuth/PycharmProjects/transomaly/plots/gp_fits/gp_{objid}.pdf')
        plt.close()

    return gp_lc, objid


def save_gps(light_curves, save_dir='data/saved_light_curves/', class_num=None, passbands=('g', 'r'), plot=False,
             nprocesses=1, redo=False):
    """ Save gaussian process fits.
    Don't plot in parallel
    """
    save_gp_filepath = os.path.join(save_dir, f"gp_classnum_{class_num}.pickle")

    if os.path.exists(save_gp_filepath) and not redo:
        with open(save_gp_filepath, "rb") as fp:  # Unpickling
            saved_gp_fits = pickle.load(fp)
    else:
        args_list = []
        for objid, lc in light_curves.items():
            args_list.append((lc, objid, passbands, plot))

        saved_gp_fits = {}
        if nprocesses == 1:
            for args in args_list:
                out = fit_gaussian_process(args)
                if out is not None:
                    gp_lc, objid = out
                    saved_gp_fits[objid] = gp_lc
        else:
            pool = mp.Pool(nprocesses)
            results = pool.map_async(fit_gaussian_process, args_list)
            pool.close()
            pool.join()

            outputs = results.get()
            print('combining results...')
            for i, output in enumerate(outputs):
                print(i, len(outputs))
                if output is not None:
                    gp_lc, objid = output
                    saved_gp_fits[objid] = gp_lc

        with open(save_gp_filepath, "wb") as fp:  # Pickling
            pickle.dump(saved_gp_fits, fp)

    return saved_gp_fits


def main():
    nprocesses = 1
    class_num = 1
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(SCRIPT_DIR, '..', 'data/ZTF_20190512')
    save_dir = os.path.join(SCRIPT_DIR, '..', 'data/saved_light_curves')

    light_curves = get_data(class_num, data_dir=data_dir, save_dir=save_dir, nprocesses=nprocesses)
    saved_gp_fits = save_gps(light_curves, save_dir=save_dir, class_num=class_num, passbands=('g', 'r'), plot=True,
                             nprocesses=nprocesses, redo=True)


if __name__ == '__main__':
    main()


