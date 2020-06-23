import numpy as np
from scipy.interpolate import interp1d
from transomaly import helpers


class PrepareArrays(object):
    def __init__(self, passbands=('g', 'r'), contextual_info=('redshift',), use_gp_interp=False):
        self.passbands = passbands
        self.contextual_info = contextual_info
        self.nobs = 50
        self.npb = len(passbands)
        self.nfeatures = self.npb + len(self.contextual_info)
        self.timestep = 3.0
        self.mintime = -70
        self.maxtime = 80
        self.use_gp_interp = use_gp_interp

    def get_min_max_time(self, lc):
        # Get min and max times for tinterp
        mintimes = []
        maxtimes = []
        for j, pb in enumerate(self.passbands):
            pbmask = lc['passband'] == pb

            time = lc[pbmask]['time'][0:self.nobs].data
            mintimes.append(time.min())
            maxtimes.append(time.max())
        mintime = min(mintimes)
        maxtime = max(maxtimes) + self.timestep

        return mintime, maxtime

    def get_t_interp(self, lc, extrapolate=False):
        mintime, maxtime = self.get_min_max_time(lc)

        if extrapolate:
            tinterp = np.arange(self.mintime, self.maxtime, step=self.timestep)
            len_t = len(tinterp)
            return tinterp, len_t

        tinterp = np.arange(mintime, maxtime, step=self.timestep)
        len_t = len(tinterp)

        if len_t > self.nobs:
            tinterp = tinterp[(tinterp >= self.mintime)]
            len_t = len(tinterp)
            if len_t > self.nobs:
                tinterp = tinterp[:-(len_t - self.nobs)]
                len_t = len(tinterp)

        return tinterp, len_t

    def update_X(self, X, Xerr, idx, gp_lc, lc, tinterp, len_t, objid, contextual_info, meta_data, nsamples=10):

        # # Drop infinite values
        # lc.replace([np.inf, -np.inf], np.nan)

        for j, pb in enumerate(self.passbands):
            pbmask = lc['passband'] == pb

            # Get data
            time = lc[pbmask]['time'][0:self.nobs].data
            flux = lc[pbmask]['flux'][0:self.nobs].data
            fluxerr = lc[pbmask]['fluxErr'][0:self.nobs].data
            photflag = lc[pbmask]['photflag'][0:self.nobs].data

            # Mask out times outside of mintime and maxtime
            timemask = (time > self.mintime) & (time < self.maxtime)
            time = time[timemask]
            flux = flux[timemask]
            fluxerr = fluxerr[timemask]
            photflag = photflag[timemask]

            if time.size < 2:
                print(f"Not enough {pb}-band observations in range for object {objid}")
                continue

            if self.use_gp_interp:
                # Draw samples from GP
                try:
                    gp_lc[pb].compute(time, fluxerr)
                except Exception as e:
                    print(f"ERROR FOR OBJECT: {objid}", e)
                    continue
                pred_mean, pred_var = gp_lc[pb].predict(flux, tinterp, return_var=True)
                pred_std = np.sqrt(pred_var)
                if nsamples > 1:
                    samples = gp_lc[pb].sample_conditional(flux, t=tinterp, size=nsamples)
                elif nsamples == 1:
                    samples = [pred_mean]

                # store samples in X
                for ns in range(nsamples):
                    X[idx + ns][j][0:len_t] = samples[ns]
                    Xerr[idx + ns][j][0:len_t] = pred_std

            else:
                # USE LINEAR SPLINE INTERPOLATION INSTEAD
                # f = interp1d(time, flux, kind='linear', bounds_error=False, fill_value=0.) ##
                spl = helpers.ErrorPropagationSpline(time, flux, fluxerr, k=1, N=100, ext='zeros')  # #

                # fluxinterp = f(tinterp) ##
                fluxinterp, fluxerrinterp = spl(tinterp)  # #
                fluxinterp = np.nan_to_num(fluxinterp)
                # fluxinterp = fluxinterp.clip(min=0)
                fluxerrinterp = np.zeros(len_t)

                ##
                # for interp_idx, fluxinterp_val in enumerate(fluxinterp):
                #     if fluxinterp_val == 0.:
                #         fluxerrinterp[interp_idx] = 0
                #     else:
                #         nearest_idx = helpers.find_nearest(time, tinterp[interp_idx])  ##
                #         fluxerrinterp[interp_idx] = fluxerr[nearest_idx]  ##
                #         # fluxerrinterp[interp_idx] = fluxerrinterp[interp_idx]

                X[idx][j][0:len_t] = fluxinterp
                Xerr[idx][j][0:len_t] = fluxerrinterp

        if self.use_gp_interp:
            # Add contextual information
            for ns in range(nsamples):
                for jj, c_info in enumerate(contextual_info, 1):
                    if meta_data[c_info] == None:
                        meta_data[c_info] = 0
                    X[idx + ns][j + jj][0:len_t] = meta_data[c_info] * np.ones(len_t)

        else:
            # Add contextual information
            for jj, c_info in enumerate(contextual_info, 1):
                if meta_data[c_info] == None:
                    meta_data[c_info] = 0
                X[idx][j + jj][0:len_t] = meta_data[c_info] * np.ones(len_t)

        return X, Xerr
