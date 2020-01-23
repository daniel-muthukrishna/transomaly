import os
import numpy as np
from astrorapid.process_light_curves import read_multiple_light_curves

from transomaly.prepare_arrays import PrepareArrays
from transomaly.fit_gaussian_processes import fit_gaussian_process


class PrepareInputArrays(PrepareArrays):
    def __init__(self, passbands=('g', 'r'), contextual_info=(0,), extrapolate_gp=True):
        PrepareArrays.__init__(self, passbands, contextual_info)
        self.passbands = passbands
        self.contextual_info = contextual_info
        self.extrapolate_gp = extrapolate_gp
        self.npb = len(passbands)

    def get_light_curves(self, light_curves_list):
        """ Returns dictionary of light curve information with each object ID as a key. """
        processed_lightcurves = read_multiple_light_curves(light_curves_list, known_redshift=False,
                                                           training_set_parameters=None)
        return processed_lightcurves

    def get_gp_fits(self, light_curves):
        """ Returns dictionary of Gaussian Process fits with each object ID as a key. """
        gp_fits = {}
        for objid, lc in light_curves.items():
            out = fit_gaussian_process(lc, objid, self.passbands, plot=False,
                                       extrapolate=self.extrapolate_gp, bad_loglike_thresh=-np.inf)
            if out is not None:
                gp_lc, objid = out
                gp_fits[objid] = gp_lc
            else:
                print(f"Unable to fit gaussian process to object: {objid}")
                gp_fits[objid] = None

        return gp_fits

    def make_input_arrays(self, light_curves_list, nsamples=1):
        nobjects = len(light_curves_list)
        nrows = nobjects * nsamples

        X = np.zeros(shape=(nrows, self.nfeatures, self.nobs))
        Xerr = np.zeros(shape=(nrows, self.npb, self.nobs))
        timesX = np.zeros(shape=(nrows, self.nobs))
        objids = []
        trigger_mjds = []

        lcs = self.get_light_curves(light_curves_list)
        gp_fits = self.get_gp_fits(lcs)

        for i, (objid, gp_lc) in enumerate(gp_fits.items()):
            print(f"Preparing light curve {i} of {nobjects}")

            idx = i * nsamples
            lc = lcs[objid]
            otherinfo = lc['otherinfo'].values.flatten()
            redshift, b, mwebv, trigger_mjd = otherinfo[0:4]

            tinterp, len_t = self.get_t_interp(lc, extrapolate=self.extrapolate_gp)
            for ns in range(nsamples):
                timesX[idx + ns][0:len_t] = tinterp
                objids.append(objid)
                trigger_mjds.append(trigger_mjd)
            X, Xerr = self.update_X(X, Xerr, idx, gp_lc, lc, tinterp, len_t, objid, self.contextual_info, otherinfo, nsamples)

        # Correct shape for keras is (N_objects, N_timesteps, N_passbands) (where N_timesteps is lookback time)
        X = X.swapaxes(2, 1)
        Xerr = Xerr.swapaxes(2, 1)

        y = X[:, 1:, :self.npb]
        yerr = Xerr[:, 1:, :self.npb]
        X = X[:, :-1]
        Xerr = Xerr[:, :-1]

        return X, Xerr, y, yerr, timesX, objids, trigger_mjds, lcs, gp_fits
