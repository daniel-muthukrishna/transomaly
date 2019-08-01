import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from transomaly.fit_gaussian_processes import get_data, save_gps


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

    def get_t_interp(self, lc):
        mintime, maxtime = self.get_min_max_time(lc)

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

    def make_arrays(self, light_curves, saved_gp_fits, nsamples):
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
            print(i, nobjects)
            lc = light_curves[objid]

            otherinfo = lc['otherinfo'].values.flatten()
            # redshift, b, mwebv, trigger_mjd, t0, peakmjd = otherinfo[0:6]

            # TODO: make cuts

            tinterp, len_t = self.get_t_interp(lc)
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


class PrepareTrainingSetArrays(PrepareArrays):
    def __init__(self, passbands=('g', 'r'), contextual_info=(0,), data_dir='data/ZTF_20190512/',
                 save_dir='data/saved_light_curves/', training_set_dir='data/training_set_files/', redo=False):
        PrepareArrays.__init__(self, passbands, contextual_info)
        self.passbands = passbands
        self.contextual_info = contextual_info
        self.nobs = 50
        self.npassbands = len(passbands)
        self.nfeatures = self.npassbands + len(self.contextual_info)
        self.timestep = 3.0
        self.mintime = -70
        self.maxtime = 80
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.training_set_dir = training_set_dir
        self.redo = redo
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.training_set_dir):
            os.makedirs(self.training_set_dir)

    def get_light_curves(self, class_num, nprocesses=1):
        light_curves = get_data(class_num, self.data_dir, self.save_dir, self.passbands, nprocesses, self.redo)

        return light_curves

    def get_gaussian_process_fits(self, light_curves, class_num, plot=False, nprocesses=1):
        saved_gp_fits = save_gps(light_curves, self.save_dir, class_num, self.passbands, plot=plot,
                                 nprocesses=nprocesses, redo=self.redo)

        return saved_gp_fits

    def make_training_set(self, class_nums=(1,), nsamples=10, otherchange='', nprocesses=1):
        savepath = os.path.join(self.training_set_dir, "X_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums))

        if self.redo is True or not os.path.isfile(savepath):
            light_curves = {}
            saved_gp_fits = {}
            for class_num in class_nums:
                light_curves.update(self.get_light_curves(class_num, nprocesses))
                saved_gp_fits.update(self.get_gaussian_process_fits(light_curves, class_num, plot=False, nprocesses=nprocesses))

            # Find intersection of dictionaries
            objids = list(set(light_curves.keys()) & set(saved_gp_fits.keys()))

            # Train test split for light_ curves and GPs
            objids_train, objids_test = train_test_split(objids, train_size=0.80, shuffle=True, random_state=42)
            lcs_train = {k: light_curves[k] for k in objids_train}
            gps_train = {k: saved_gp_fits[k] for k in objids_train}
            lcs_test = {k: light_curves[k] for k in objids_test}
            gps_test = {k: saved_gp_fits[k] for k in objids_test}

            X_train, Xerr_train, timesX_train, labels_train, objids_train = self.make_arrays(lcs_train, gps_train, nsamples)
            X_test, Xerr_test, timesX_test, labels_test, objids_test = self.make_arrays(lcs_test, gps_test, nsamples)

            # Shuffle training set but leave testing set in order or gaussian process samples
            X_train, Xerr_train, timesX_train, labels_train, objids_train = shuffle(X_train, Xerr_train, timesX_train, labels_train, objids_train, random_state=42)

            np.save(os.path.join(self.training_set_dir, "X_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), X_train)
            np.save(os.path.join(self.training_set_dir, "Xerr_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), Xerr_train)
            np.save(os.path.join(self.training_set_dir, "timesX_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), timesX_train)
            np.save(os.path.join(self.training_set_dir, "labels_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), labels_train)
            np.save(os.path.join(self.training_set_dir, "objids_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), objids_train)
            np.save(os.path.join(self.training_set_dir, "X_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), X_test)
            np.save(os.path.join(self.training_set_dir, "Xerr_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), Xerr_test)
            np.save(os.path.join(self.training_set_dir, "timesX_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), timesX_test)
            np.save(os.path.join(self.training_set_dir, "labels_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), labels_test)
            np.save(os.path.join(self.training_set_dir, "objids_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), objids_test)
        else:
            X_train = np.load(os.path.join(self.training_set_dir, "X_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), mmap_mode='r')
            Xerr_train = np.load(os.path.join(self.training_set_dir, "Xerr_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), mmap_mode='r')
            timesX_train = np.load(os.path.join(self.training_set_dir, "timesX_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)))
            labels_train = np.load(os.path.join(self.training_set_dir, "labels_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)))
            objids_train = np.load(os.path.join(self.training_set_dir, "objids_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)))
            X_test = np.load(os.path.join(self.training_set_dir, "X_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), mmap_mode='r')
            Xerr_test = np.load(os.path.join(self.training_set_dir, "Xerr_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), mmap_mode='r')
            timesX_test = np.load(os.path.join(self.training_set_dir, "timesX_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)))
            labels_test = np.load(os.path.join(self.training_set_dir, "labels_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)))
            objids_test = np.load(os.path.join(self.training_set_dir, "objids_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)))

        X_train = X_train.swapaxes(2, 1)  # Correct shape for keras is (N_objects, N_timesteps, N_passbands)
        Xerr_train = Xerr_train.swapaxes(2, 1)
        y_train = X_train[:, 1:, :2]
        yerr_train = Xerr_train[:, 1:, :2]
        X_train = X_train[:, :-1]
        Xerr_train = Xerr_train[:, :-1]
        timesX_train = timesX_train[:, :-1]
        X_test = X_test.swapaxes(2, 1)  # Correct shape for keras is (N_objects, N_timesteps, N_passbands)
        Xerr_test = Xerr_test.swapaxes(2, 1)
        y_test = X_test[:, 1:, :2]
        yerr_test = Xerr_test[:, 1:, :2]
        X_test = X_test[:, :-1]
        Xerr_test = Xerr_test[:, :-1]
        timesX_test = timesX_test[:, :-1]

        # Add errors as extra column to y
        ye_train = np.copy(yerr_train)
        ye_test = np.copy(yerr_test)
        ye_train[yerr_train == 0] = np.ones(yerr_train[yerr_train == 0].shape)
        ye_test[yerr_test == 0] = np.ones(yerr_test[yerr_test == 0].shape)
        y_train = np.dstack((y_train, ye_train))
        y_test = np.dstack((y_test, ye_test))


        return X_train, X_test, y_train, y_test, Xerr_train, Xerr_test, yerr_train, yerr_test, \
               timesX_train, timesX_test, labels_train, labels_test, objids_train, objids_test


def main():
    passbands = ('g', 'r')
    contextual_info = (0,)
    data_dir = '/Users/danmuth/PycharmProjects/transomaly/data/ZTF_20190512/'
    save_dir = '/Users/danmuth/PycharmProjects/transomaly/data/saved_light_curves'
    training_set_dir = '/Users/danmuth/PycharmProjects/transomaly/data/training_set_files/'
    nprocesses = 1
    class_nums = (1,)
    otherchange = ''
    nsamples = 100

    prepare_training_set = PrepareTrainingSetArrays(passbands, contextual_info, data_dir, save_dir, training_set_dir)
    prepare_training_set.make_training_set(class_nums, nsamples, otherchange, nprocesses)


if __name__ == '__main__':
    main()







