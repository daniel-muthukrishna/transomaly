import numpy as np


class PrepareArrays(object):
    def __init__(self, passbands=('g', 'r'), contextual_info=(0,)):
        self.passbands = passbands
        self.contextual_info = contextual_info
        self.nobs = 50
        self.npassbands = len(passbands)
        self.nfeatures = self.npassbands + len(self.contextual_info)
        self.timestep = 3.0
        self.mintime = -70
        self.maxtime = 80

    def get_min_max_time(self, lc):
        # Get min and max times for tinterp
        mintimes = []
        maxtimes = []
        for j, pb in enumerate(self.passbands):
            if pb not in lc:
                continue
            time = lc[pb]['time'][0:self.nobs].dropna()
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

    def update_X(self, X, Xerr, idx, gp_lc, lc, tinterp, len_t, objid, contextual_info, otherinfo, nsamples=10):

        for j, pb in enumerate(self.passbands):
            if pb not in lc:
                print("No", pb, "in objid:", objid)
                continue

            # Drop infinite values
            lc.replace([np.inf, -np.inf], np.nan)

            # Get data
            time = lc[pb]['time'][0:self.nobs].dropna().values
            flux = lc[pb]['flux'][0:self.nobs].dropna().values
            fluxerr = lc[pb]['fluxErr'][0:self.nobs].dropna().values
            photflag = lc[pb]['photflag'][0:self.nobs].dropna().values

            # Mask out times outside of mintime and maxtime
            timemask = (time > self.mintime) & (time < self.maxtime)
            time = time[timemask]
            flux = flux[timemask]
            fluxerr = fluxerr[timemask]
            photflag = photflag[timemask]

            if time.size < 1:
                print(f"No {pb}-band observations in range for object {objid}")
                continue

            # Draw samples from GP
            gp_lc[pb].compute(time, fluxerr)
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

        # Add contextual information
        for ns in range(nsamples):
            for jj, c_idx in enumerate(contextual_info, 1):
                X[idx + ns][j + jj][0:len_t] = otherinfo[c_idx] * np.ones(len_t)

        return X

    def make_arrays(self, light_curves, saved_gp_fits, nsamples, extrapolate=True):
        nobjects = len(light_curves)
        nrows = nobjects * nsamples

        labels = np.zeros(shape=nrows, dtype=np.uint16)
        # X = np.memmap(os.path.join(self.training_set_dir, 'X_lc_data.dat'), dtype=np.float32, mode='w+',
        #               shape=(nrows, self.nfeatures, self.nobs))
        # Xerr = np.memmap(os.path.join(self.training_set_dir, 'Xerr_lc_data.dat'), dtype=np.float32, mode='w+',
        #               shape=(nrows, self.nfeatures, self.nobs))
        X = np.zeros(shape=(nrows, self.nfeatures, self.nobs))
        Xerr = np.zeros(shape=(nrows, self.nfeatures, self.nobs))
        timesX = np.zeros(shape=(nrows, self.nobs))
        objids = []

        for i, (objid, gp_lc) in enumerate(saved_gp_fits.items()):
            idx = i * nsamples
            if i % 100 == 0:
                print(i, nobjects)
            lc = light_curves[objid]

            otherinfo = lc['otherinfo'].values.flatten()
            # redshift, b, mwebv, trigger_mjd, t0, peakmjd = otherinfo[0:6]

            # TODO: make cuts

            tinterp, len_t = self.get_t_interp(lc, extrapolate=extrapolate)
            for ns in range(nsamples):
                timesX[idx + ns][0:len_t] = tinterp
                objids.append(objid)
                labels[idx + ns] = int(objid.split('_')[0])
            X = self.update_X(X, Xerr, idx, gp_lc, lc, tinterp, len_t, objid, self.contextual_info, otherinfo, nsamples)

        # Count nobjects per class
        classes = sorted(list(set(labels)))
        for c in classes:
            nobs = len(X[labels == c])
            print(f"class {c}: {nobs}")

        return X, Xerr, timesX, labels, objids
