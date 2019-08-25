import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import OrderedDict
from keras.models import load_model
from pkg_resources import resource_filename

from transomaly.prepare_input import PrepareInputArrays
from transomaly.loss_functions import mean_squared_error, chisquare_loss, mean_squared_error_over_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COLPB = {'g': 'tab:green', 'r': 'tab:red'}
MARKPB = {'g': 'o', 'r': 's', 'z': 'd'}
ALPHAPB = {'g': 0.3, 'r': 1., 'z': 1}

CLASS_COLOR = {'SNIa-norm': 'tab:green', 'SNIbc': 'tab:orange', 'SNII': 'tab:blue', 'SNIIn': 'blue',
               'SNIa-91bg': 'tab:red', 'SNIa-x': 'bisque', 'point-Ia': 'tab:brown', 'Kilonova': '#aaffc3',
               'SLSN-I': 'tab:olive', 'PISN': 'tab:cyan', 'ILOT': '#FF1493', 'CART': 'navy', 'TDE': 'tab:pink',
               'AGN': 'tab:purple'}


class TransientRegressor(object):
    def __init__(self, nsamples=1, model_class='SNIa-norm', model_filepath='', passbands=('g', 'r')):

        self.nsamples = nsamples
        self.passbands = passbands
        self.contextual_info = ()
        self.npb = len(passbands)

        if model_filepath != '' and os.path.exists(model_filepath):
            self.model_filepath = model_filepath
            print("Invalid keras model. Using default model...")
        elif model_class == 'SNIa-norm':
            self.model_filepath = os.path.join(SCRIPT_DIR, "../plots/model_8020split_ci()_ns1_c(1,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5")
        elif model_class == 'SNII':
            self.model_filepath = os.path.join(SCRIPT_DIR, "../plots/model_8020split_ci()_ns1_c(2, 12, 14)/keras_model_epochs500_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs500_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5")
        elif model_class == 'SNIbc':
            self.model_filepath = os.path.join(SCRIPT_DIR, "../plots/model_8020split_ci()_ns1_c(3, 13)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5")
        elif model_class == 'SNIa-91bg':
            self.model_filepath = os.path.join(SCRIPT_DIR, "../plots/model_8020split_ci()_ns1_c(41,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5")
        elif model_class == 'SNIa-x':
            self.model_filepath = os.path.join(SCRIPT_DIR, "../plots/model_8020split_ci()_ns1_c(43,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5")
        elif model_class == 'SLSN-I':
            self.model_filepath = os.path.join(SCRIPT_DIR, "../plots/model_8020split_ci()_ns1_c(60,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5")
        elif model_class == 'AGN':
            self.model_filepath = os.path.join(SCRIPT_DIR, "../plots/model_8020split_ci()_ns1_c(70,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5")
        elif model_class == 'TDE':
            self.model_filepath = os.path.join(SCRIPT_DIR, "../plots/model_8020split_ci()_ns1_c(64,)/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons/keras_model_epochs1000_unnormalised_mse_predict_last49_timesteps_nodropout_100lstmneurons.hdf5")
        else:
            print("INVALID MODEL SPECIFIED. Using SNIa-norm model...")
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
                obs_time = self.get_obstime(idx)
                y_predict_at_obstime = []
                for pbidx, pb in enumerate(self.passbands):
                    y_predict_at_obstime.append(np.interp(obs_time, self.timesX[idx][1:][:argmax[idx]], self.y_predict[idx][:, pbidx][:argmax[idx]]))
                y_predict.append(np.array(y_predict_at_obstime).T)
                time_steps.append(obs_time + self.trigger_mjds[idx])
        else:
            y_predict = [self.y_predict[i][:argmax[i]] for i in range(nobjects)]
            time_steps = [self.timesX[i][1:][:argmax[i]] + self.trigger_mjds[i] for i in range(nobjects)]

        y_predict = np.array(y_predict)

        return np.array(y_predict), time_steps, self.objids

    def get_obstime(self, idx):
        obs_time = []
        for pb in self.passbands:
            obs_time.append(self.lcs[self.objids[idx]][pb]['time'].values)
        obs_time = np.array(obs_time)
        obs_time = np.sort(obs_time[~np.isnan(obs_time)])

        return obs_time

    def get_anomaly_scores(self, return_predictions_at_obs_time=False, light_curves=None):
        if not hasattr(self, 'y_predict'):
            self.get_regressor_predictions(light_curves)

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
                    yt = self.y[sidx + s, :, pbidx][:argmax][m]
                    yp = self.y_predict[sidx + s, :, pbidx][:argmax][m]
                    ye = self.yerr[sidx+s, :, pbidx][:argmax][m]
                    try:
                        chi2 += ((yp - yt) / ye) ** 2
                    except ValueError as e:
                        print(f"Failed chi2 object {self.objids[sidx + s]}, {pb}", e)
                chi2_reduced = chi2 / self.npb

                if return_predictions_at_obs_time:
                    obs_time = self.get_obstime(sidx+s)
                    chi2_reduced = np.interp(obs_time, self.timesX[sidx+s][1:][:argmax], chi2_reduced)

                chi2_samples.append(chi2_reduced)
            anomaly_score_samples = chi2_samples
            anomaly_scores.append(np.mean(anomaly_score_samples, axis=0))
            anomaly_scores_std.append(np.std(anomaly_score_samples, axis=0))

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

        if not hasattr(self, 'y_predict'):
            self.get_regressor_predictions(light_curves, return_predictions_at_obstime)

        font = {'family': 'normal',
                'size': 36}
        matplotlib.rc('font', **font)

        if indexes_to_plot is None or indexes_to_plot is True:
            indexes_to_plot = np.arange(int(len(self.y_predict)/self.nsamples))

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
                    ax1.plot(self.timesX[sidx + s][1:][:argmax], self.y_predict[sidx + s][:, pbidx][:argmax], c=COLPB[pb], lw=lw, label=plotlabelpred, marker=None, markersize=10, alpha=alpha, linestyle=':')
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


class GetAllTransientRegressors(object):
    def __init__(self, model_classes='all', nsamples=1, passbands=('g', 'r')):
        self.nsamples = nsamples
        self.passbands = passbands

        if model_classes == 'all':
            self.model_classes = ('SNIa-norm', 'SNII', 'SNIbc', 'SLSN-I', 'AGN', 'TDE')  # ('SNIa-norm', 'SNIa-x', 'SNII', 'SNIbc', 'SLSN-I', 'TDE', 'AGN') # 'SNIa-91bg'
        else:
            self.model_classes = model_classes

        self.transient_regressors = {}
        for model_class in self.model_classes:
            self.transient_regressors[model_class] = TransientRegressor(nsamples=nsamples, model_class=model_class,
                                                                        passbands=passbands)

    def get_regressor_predictions(self, light_curves, return_predictions_at_obstime=False):
        predictions = {}
        for model_class in self.model_classes:
            predictions[model_class], time_steps, objids = self.transient_regressors[model_class].get_regressor_predictions(light_curves, return_predictions_at_obstime)

        return predictions, time_steps, objids

    def get_processed_light_curves(self):
        lcs = {}
        gps = {}
        trigger_mjds = {}
        timesX = {}
        y = {}
        yerr = {}
        objids = {}
        for model_class in self.model_classes:
            lcs[model_class] = self.transient_regressors[model_class].lcs
            gps[model_class] = self.transient_regressors[model_class].gp_fits
            trigger_mjds = self.transient_regressors[model_class].trigger_mjds
            timesX = self.transient_regressors[model_class].timesX
            y[model_class] = self.transient_regressors[model_class].y
            yerr[model_class] = self.transient_regressors[model_class].yerr
            objids = self.transient_regressors[model_class].objids
        return lcs, gps, trigger_mjds, timesX, y, yerr, objids

    def get_anomaly_scores(self, return_predictions_at_obstime):
        anomaly_scores, anomaly_scores_std = {}, {}
        for model_class in self.model_classes:
            anomaly_scores[model_class], anomaly_scores_std[model_class], objids = self.transient_regressors[model_class].get_anomaly_scores(return_predictions_at_obstime)

        return anomaly_scores, anomaly_scores_std, objids

    def plot_anomaly_scores_all_models(self, anomaly_scores, time_steps, indexes_to_plot=None, fig_dir='.', plot_animation=True):
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        if indexes_to_plot is None or indexes_to_plot is True:
            indexes_to_plot = np.arange(len(time_steps))

        class_names = list(anomaly_scores.keys())

        font = {'family': 'normal',
                'size': 36}
        matplotlib.rc('font', **font)

        # Get raw light curve observations
        lcs, gps, trigger_mjds, timesX, y, yerr, objids = self.get_processed_light_curves()

        for idx in indexes_to_plot:
            sidx = idx * self.nsamples

            lc = lcs[self.model_classes[0]][objids[sidx]]

            fig = plt.figure(figsize=(26, 15))
            ax1 = plt.subplot(221)
            ax2 = plt.subplot(122)
            ax3 = plt.subplot(223, sharex=ax1)
            # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(26, 15), sharex='none')
            # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(13, 15))

            all_flux = np.concatenate([list(lc[pb]['flux'].dropna()) for pb in self.passbands])
            all_anomaly_scores = np.concatenate([anomaly_scores[c][idx] for c in class_names])
            time_steps[idx] = time_steps[idx] - trigger_mjds[idx]

            a_scores = {c: np.zeros(len(time_steps[idx])) for c in class_names}
            for c in class_names:
                len_t = 0
                chi2_cumsum = 0
                for t in range(len(time_steps[idx])):
                    len_t += 1
                    chi2_cumsum += anomaly_scores[c][idx][t]
                    a_scores[c][t] = chi2_cumsum / len_t
            a_scores = pd.DataFrame(a_scores)
            a_scores = np.exp(-a_scores/2)

            ax1.legend()
            ax1.set_ylabel('Relative flux')
            ax1.set_ylim(min(all_flux)-0.2*min(all_flux), 1.2*max(all_flux))
            ax1.set_xlim(min(time_steps[idx]), max(time_steps[idx]))
            ax2.set_ylabel('Likelihood')
            ax2.set_ylim(a_scores.values.min(), a_scores.values.max())
            ax3.set_ylabel('Anomaly score')
            ax3.set_xlabel('Time since trigger [Days]')
            ax3.set_xlim(min(time_steps[idx]), max(time_steps[idx]))
            ax3.set_ylim(0, 1.1*max(all_anomaly_scores))
            # ax3.legend()
            fig.subplots_adjust(hspace=0)
            plt.tight_layout()
            plt.gcf().subplots_adjust(bottom=0.2)

            if plot_animation:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=2, bitrate=1800)

            def animate(ani_i):
                for pbidx, pb in enumerate(self.passbands):
                    if ani_i + 1 >= len(time_steps[idx]):
                        break

                    dea = [lc[pb]['time'].dropna().values < time_steps[idx][int(ani_i + 1)]]

                    ax1.errorbar(lc[pb]['time'].dropna().values[dea], lc[pb]['flux'].dropna().values[dea], yerr=lc[pb]['fluxErr'].dropna().values[dea],
                                 fmt="o", color=COLPB[pb], label=pb, markersize='10', capsize=4, elinewidth=4)
                ax2.clear()
                ax2.set_ylim(a_scores.values.min(), a_scores.values.max())
                ax2.set_ylabel('Likelihood')
                barplt = ax2.bar(np.arange(len(class_names)), a_scores.iloc[int(ani_i)])
                ax2.set_xticks(np.arange(len(class_names)), class_names)
                ax2.set_xticklabels(np.insert(class_names, 0, 0), rotation=90)

                for i, c in enumerate(class_names):
                    barplt[i].set_color(CLASS_COLOR[c])
                    ax3.plot(time_steps[idx][:int(ani_i + 1)], anomaly_scores[c][idx][:int(ani_i + 1)], lw=4, color=CLASS_COLOR[c], label=c)

                # Don't repeat legend items
                ax1.legend()
                handles1, labels1 = ax1.get_legend_handles_labels()
                by_label1 = OrderedDict(zip(labels1, handles1))
                ax1.legend(by_label1.values(), by_label1.keys(), loc='upper right')

            if plot_animation:
                ani = animation.FuncAnimation(fig, animate, frames=50, repeat=True, interval=1000)
                savename = os.path.join(fig_dir, f"anomaly_plots_{objids[sidx]}.mp4")
                ani.save(savename, writer=writer)
            animate(49)
            plt.savefig(os.path.join(fig_dir, f"anomaly_plots_{objids[sidx]}.pdf"))


