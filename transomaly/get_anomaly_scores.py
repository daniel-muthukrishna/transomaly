import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import load_model
from pkg_resources import resource_filename

from transomaly.prepare_input import PrepareInputArrays
from transomaly.loss_functions import mean_squared_error, chisquare_loss, mean_squared_error_over_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLPB = {'g': 'tab:green', 'r': 'tab:red'}
MARKPB = {'g': 'o', 'r': 's', 'z': 'd'}
ALPHAPB = {'g': 0.3, 'r': 1., 'z': 1}


class TransientRegressor(object):
    def __init__(self, nsamples=1, model_filepath='', passbands=('g', 'r')):

        self.nsamples = nsamples
        self.passbands = passbands
        self.contextual_info = ()
        self.npb = len(passbands)

        if model_filepath != '' and os.path.exists(model_filepath):
            self.model_filepath = model_filepath
            print("Invalid keras model. Using default model...")
        else:
            self.model_filepath = os.path.join(SCRIPT_DIR, "../plots/model_8020split_ci()_ns1_c(1,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5")

        self.model = load_model(self.model_filepath, custom_objects={'loss': mean_squared_error()})

    def process_light_curves(self, light_curves):
        """
        light_curve_list is a list of tuples with each tuple having entries:
        mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv

        """

        prepareinputarrays = PrepareInputArrays(self.passbands, self.contextual_info)
        X, Xerr, y, yerr, timesX, objids_list, trigger_mjds, lcs, gp_fits = prepareinputarrays.make_input_arrays(light_curves, self.nsamples)

        return X, Xerr, y, yerr, timesX, objids_list, trigger_mjds, lcs, gp_fits

    def get_regressor_predictions(self, light_curves, return_predictions_at_obstime=False):
        """ Return the RNN predictions as a function of time

        Parameters
        ----------
        light_curves : list
            Is a list of tuples. Each tuple contains the light curve information of a transient object in the form
            (mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv).
            Here, mjd, flux, fluxerr, passband, and photflag are arrays.
            And ra, dec, objid, redshift, and mwebv are floats.
                mjd: list of Modified Julian Dates of light curve

                flux: list of fluxes at each mjd

                fluxerr: list of flux errors

                passband: list of strings indicating the passband. 'r' or 'g' for r-band or g-band observations.

                photflag: list of flags identifying whether the observation is a detection (4096), non-detection (0),
                or the first detection (6144).

                ra: Right Ascension (float value).

                dec: Declination (float value).

                objid: Object Identifier (String).

                redshift: Cosmological redshift of object (float). Set to NoneType if redshift is unknown.

                mwebv: Milky way extinction.

        return_predictions_at_obstime: bool
            Return the predictions at the observation times instead of at the 50 interpolated timesteps.
        return_objids : bool, optional
            If True, also return the object IDs (objids) in the same order as the returned predictions.

        Returns
        -------
        y_predict: array
            Classification probability vector at each time step for each object.
            Array of shape (s, n, m) is returned.
            Where s is the number of obejcts that are classified,
            n is the number of times steps, and m is the number of classes.
        time_steps: array
            MJD time steps corresponding to the timesteps of the y_predict array.
        objids : array, optional
            The object ids (objids) that were input into light_curves are returned in the same order as y_predict.
            Only provided if return_objids is True.
        """

        # Do error checks
        assert isinstance(light_curves, (list, np.ndarray))
        for light_curve in light_curves:
            assert len(light_curve) == 10
            mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv = light_curve
            _lcshape = len(mjd)
            assert _lcshape == len(flux)
            assert _lcshape == len(fluxerr)
            assert _lcshape == len(passband)
            assert _lcshape == len(photflag)

        self.X, self.Xerr, self.y, self.yerr, self.timesX, self.objids, self.trigger_mjds, self.lcs, self.gp_fits = self.process_light_curves(light_curves)

        nobjects = len(self.objids)
        if nobjects == 0:
            print("No objects to classify. These may have been removed from the chosen selection cuts")
            return None, None, self.objids

        self.y_predict = self.model.predict(self.X)

        argmax = self.timesX.argmax(axis=1) + 1

        if return_predictions_at_obstime:
            (s, n, p) = self.y_predict.shape  # (s, n, m) = (num light curves, num timesteps, num passbands)
            y_predict = []
            time_steps = []
            for idx in range(s):
                obs_time = []
                for pb in self.passbands:
                    obs_time.append(self.lcs[self.objids[idx]][pb]['time'].values)
                obs_time = np.array(obs_time)
                obs_time = np.sort(obs_time[~np.isnan(obs_time)])
                y_predict_at_obstime = []
                for pbidx, pb in enumerate(self.passbands):
                    y_predict_at_obstime.append(np.interp(obs_time, self.timesX[idx][:-1][:argmax[idx]], self.y_predict[idx][:, pbidx][:argmax[idx]]))
                y_predict.append(np.array(y_predict_at_obstime).T)
                time_steps.append(obs_time + self.trigger_mjds[idx])
        else:
            y_predict = [self.y_predict[i][:argmax[i]] for i in range(nobjects)]
            time_steps = [self.timesX[i][:argmax[i]] + self.trigger_mjds[i] for i in range(nobjects)]

        y_predict = np.array(y_predict)

        return np.array(y_predict), time_steps, self.objids

    def get_anomaly_scores(self, y_predict):
        nobjects = len(self.objids)
        argmax = None

        anomaly_scores = []
        anomaly_scores_std = []
        for idx in range(nobjects):
            sidx = idx * self.nsamples
            chi2_samples = []
            for s in range(self.nsamples):
                chi2 = 0
                for pbidx, pb in enumerate(self.passbands):
                    m = self.y[sidx+s, :, pbidx][:argmax] != 0  # ignore zeros (where no data exists)
                    yt = y_predict[sidx+s, :, pbidx][:argmax][m]
                    yp = y_predict[sidx+s, :, pbidx][:argmax][m]
                    ye = self.yerr[sidx+s, :, pbidx][:argmax][m]
                    chi2 += ((yp - yt) / ye) ** 2
                chi2_samples.append(chi2 / self.npb)
            anomaly_score_samples = chi2_samples
            anomaly_scores = np.mean(anomaly_score_samples, axis=0)
            anomaly_scores_std = np.std(anomaly_score_samples, axis=0)

        return anomaly_scores, anomaly_scores_std, self.objids

    def plot_anomaly_scores(self, indexes_to_plot=None, fig_dir='.', light_curves=None, return_predictions_at_obstime=None):
        """
        Plot light curve (top panel) and classifications (bottom panel) vs time.

        Parameters
        ----------
        indexes_to_plot : tuple
            The indexes listed in the tuple will be plotted according to the order of the input light curves.
            E.g. (0, 1, 3, 5) will plot the zeroth, first, third and fifth light curves and classifications.
            If None or True, then all light curves will be plotted
        step : bool
            Plot step function along data points instead of interpolating classifications between data.
        use_interp_flux : bool
            Use all 50 timesteps when plotting classification probabilities rather than just at the timesteps with data.
        figdir : str
            Directory to save figure.
        plot_matrix_input : bool
            Plots the interpolated light curve passed into the neural network on top of the observations.
        light_curves : list
            This argument is only required if the get_predictions() method has not been run.
            Is a list of tuples. Each tuple contains the light curve information of a transient object in the form
            (mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv).
            Here, mjd, flux, fluxerr, passband, and photflag are arrays.
            ra, dec, objid, redshift, and mwebv are floats

        """

        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        font = {'family': 'normal',
                'size': 36}
        matplotlib.rc('font', **font)

        if indexes_to_plot is None or indexes_to_plot is True:
            indexes_to_plot = np.arange(len(self.y_predict))

        if not hasattr(self, 'y_predict'):
            self.get_regressor_predictions(light_curves, return_predictions_at_obstime)

        for idx in indexes_to_plot:
            sidx = idx * self.nsamples  # Assumes like samples are in order
            print("Plotting example vs time", idx, self.objids[sidx])
            argmax = None  # self.timesX[idx].argmax() + 1

            # Get raw light curve observations
            lc = self.lcs[self.objids[sidx]]
            gp_lc = self.gp_fits[self.objids[sidx]]

            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 15),
                                           num="classification_vs_time_{}".format(idx), sharex=True)
            # ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
            # ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)

            for pbidx, pb in enumerate(self.passbands):
                for s in range(self.nsamples):
                    lw = 3 if s == 0 else 0.5
                    alpha = 1 if s == 0 else 0.1
                    plotlabeltest = "ytest:{}".format(pb) if s == 0 else ''
                    plotlabelpred = "ypred:{}".format(pb) if s == 0 else ''
                    marker = None  # MARKPB[pb] if s == 0 else None
                    ax1.plot(self.timesX[sidx + s][1:][:argmax], self.y[sidx + s][:, pbidx][:argmax], c=COLPB[pb], lw=lw, label=plotlabeltest, marker=None, markersize=10, alpha=alpha, linestyle='-')
                    ax1.plot(self.timesX[sidx + s][1:][:argmax], self.y_predict[sidx + s][:, pbidx][:argmax], c=COLPB[pb], lw=lw, label=plotlabelpred, marker='*', markersize=10, alpha=alpha, linestyle=':')
                ax1.errorbar(lc[pb]['time'].dropna(), lc[pb]['flux'].dropna(), yerr=lc[pb]['fluxErr'].dropna(),
                             fmt=".", capsize=0, color=COLPB[pb], label='_nolegend_')

                gp_lc[pb].compute(lc[pb]['time'].dropna(), lc[pb]['fluxErr'].dropna())
                pred_mean, pred_var = gp_lc[pb].predict(lc[pb]['flux'].dropna(), self.timesX[sidx + s][:argmax],
                                                        return_var=True)
                pred_std = np.sqrt(pred_var)
                ax1.fill_between(self.timesX[sidx + s][:argmax], pred_mean + pred_std, pred_mean - pred_std,
                                 color=COLPB[pb], alpha=0.3,
                                 edgecolor="none")

            # Plot anomaly scores
            chi2_samples = []
            for s in range(self.nsamples):
                chi2 = 0
                for pbidx in range(self.npb):
                    m = self.y[sidx + s, :, pbidx][:argmax] != 0  # ignore zeros (where no data exists)
                    yt = self.y[sidx + s, :, pbidx][:argmax][m]
                    yp = self.y_predict[sidx + s, :, pbidx][:argmax][m]
                    ye = self.yerr[sidx + s, :, pbidx][:argmax][m]
                    try:
                        chi2 += ((yp - yt) / ye) ** 2
                    except ValueError as e:
                        pbidx -= 1
                        m = self.yerr[sidx + s, :, pbidx][:argmax] != 0
                        print(f"Failed chi2 object {self.objids[sidx + s]}", e)
                chi2_samples.append(chi2 / self.npb)
            anomaly_score_samples = chi2_samples
            anomaly_score_mean = np.mean(anomaly_score_samples, axis=0)
            anomaly_score_std = np.std(anomaly_score_samples, axis=0)
            ax2.text(0.05, 0.95, f"$\chi^2 = {round(np.sum(anomaly_score_mean) / len(yt), 3)}$",
                     horizontalalignment='left',
                     verticalalignment='center', transform=ax2.transAxes)

            ax2.plot(self.timesX[sidx][1:][:argmax][m], anomaly_score_mean, lw=3, marker='o')
            ax2.fill_between(self.timesX[sidx][1:][:argmax][m], anomaly_score_mean + anomaly_score_std,
                             anomaly_score_mean - anomaly_score_std, alpha=0.3, edgecolor="none")

            ax1.legend(frameon=True, fontsize=33)
            ax1.set_ylabel("Relative flux")
            ax2.set_ylabel("Anomaly score")
            ax2.set_xlabel("Time since trigger [days]")
            plt.tight_layout()
            fig.subplots_adjust(hspace=0)
            plt.savefig(os.path.join(fig_dir, f"lc_{self.objids[sidx]}_{idx}.pdf"))
            plt.close()

        return self.timesX, self.y_predict

