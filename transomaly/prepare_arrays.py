import os
import numpy as np
from sklearn.model_selection import train_test_split

from transomaly.prepare_light_curves import get_data, save_gps


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

    def update_X(self, X, i, gp_lc, lc, tinterp, len_t, objid, contextual_info, otherinfo, nsamples=10):

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

            # Draw samples from GP
            gp_lc[pb].compute(time, fluxerr)
            samples = gp_lc[pb].sample_conditional(flux, t=tinterp, size=10)

            # store samples in X
            for ns in range(nsamples):
                X[i + ns][j][0:len_t] = samples[ns]

                # Add contextual information
                for jj, c_idx in enumerate(contextual_info, 1):
                    X[i + ns][j + jj][0:len_t] = otherinfo[c_idx] * np.ones(len_t)

        return X

    def make_arrays(self, light_curves, saved_gp_fits):
        nobjects = len(light_curves)

        labels = np.zeros(shape=nobjects, dtype=np.uint16)
        X = np.memmap(os.path.join(self.training_set_dir, 'X_lc_data.dat'), dtype=np.float32, mode='w+',
                      shape=(nobjects, self.nfeatures, self.nobs))
        X[:] = np.zeros(shape=(nobjects, self.nfeatures, self.nobs))
        timesX = np.zeros(shape=(nobjects, self.nobs))
        objids = []

        for i, (objid, gp_lc) in enumerate(saved_gp_fits.items()):
            lc = light_curves[objid]

            otherinfo = lc['otherinfo'].values.flatten()
            # redshift, b, mwebv, trigger_mjd, t0, peakmjd = otherinfo[0:6]

            # TODO: make cuts

            tinterp, len_t = self.get_t_interp(lc)
            timesX[i][0:len_t] = tinterp
            objids.append(objid)
            labels[i] = int(objid.split('_')[0])
            X = self.update_X(X, i, gp_lc, lc, tinterp, len_t, objid, self.contextual_info, otherinfo)

        X = X.swapaxes(2, 1)  # Correct shape for keras is (N_objects, N_timesteps, N_passbands)
        y = X[:, 1:, :2]
        X = X[:, :-1]
        timesX = timesX[:, :-1]

        # Count nobjects per class
        classes = sorted(list(set(labels)))
        for c in classes:
            nobs = len(X[labels == c])
            print(c, nobs)

        return X, y, timesX, labels, objids


class PrepareTrainingSetArrays(PrepareArrays):
    def __init__(self, passbands=('g', 'r'), contextual_info=(0,), data_dir='data/ZTF_20190512/',
                 save_dir='data/saved_light_curves/', training_set_dir='data/saved_light_curves/', redo=False):
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

    def make_training_set(self, class_nums=(1,), otherchange='', nprocesses=1):
        savepath = os.path.join(self.training_set_dir, "X_train_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums))

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

            X_train, y_train, timesX_train, labels_train, objids_train = self.make_arrays(lcs_train, gps_train)
            X_test, y_test, timesX_test, labels_test, objids_test = self.make_arrays(lcs_test, gps_test)

            np.save(os.path.join(self.training_set_dir, "X_train_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums), X_train))
            np.save(os.path.join(self.training_set_dir, "y_train_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info), y_train))
            np.save(os.path.join(self.training_set_dir, "timesX_train_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums), timesX_train))
            np.save(os.path.join(self.training_set_dir, "labels_train_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums), labels_train))
            np.save(os.path.join(self.training_set_dir, "objids_train_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums), objids_train))
            np.save(os.path.join(self.training_set_dir, "X_test_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums), X_test))
            np.save(os.path.join(self.training_set_dir, "y_test_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums), y_test))
            np.save(os.path.join(self.training_set_dir, "timesX_test_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums), timesX_test))
            np.save(os.path.join(self.training_set_dir, "labels_test_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums), labels_test))
            np.save(os.path.join(self.training_set_dir, "objids_test_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums), objids_test))
        else:
            X_train = np.load(os.path.join(self.training_set_dir, "X_train_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums)), mmap_mode='r')
            y_train = np.load(os.path.join(self.training_set_dir, "y_train_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums)))
            timesX_train = np.load(os.path.join(self.training_set_dir, "timesX_train_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums)))
            labels_train = np.load(os.path.join(self.training_set_dir, "labels_train_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums)))
            objids_train = np.load(os.path.join(self.training_set_dir, "objids_train_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums)))
            X_test = np.load(os.path.join(self.training_set_dir, "X_test_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums)), mmap_mode='r')
            y_test = np.load(os.path.join(self.training_set_dir, "y_test_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums)))
            timesX_test = np.load(os.path.join(self.training_set_dir, "timesX_test_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums)))
            labels_test = np.load(os.path.join(self.training_set_dir, "labels_test_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums)))
            objids_test = np.load(os.path.join(self.training_set_dir, "objids_test_{}_ci{}_c{}.npy".format(otherchange, self.contextual_info, class_nums)))

        return X_train, X_test, y_train, y_test, timesX_train, timesX_test, \
               labels_train, labels_test, objids_train, objids_test


def main():
    passbands = ('g', 'r')
    contextual_info = (0,)
    data_dir = '/Users/danmuth/PycharmProjects/transomaly/data/ZTF_20190512/'
    save_dir = '/Users/danmuth/PycharmProjects/transomaly/data/saved_light_curves'
    training_set_dir = '/Users/danmuth/PycharmProjects/transomaly/data/saved_light_curves/'
    nprocesses = 1
    class_nums = (1,)
    otherchange = ''

    prepare_training_set = PrepareTrainingSetArrays(passbands, contextual_info, data_dir, save_dir, training_set_dir)
    prepare_training_set.make_training_set(class_nums, otherchange, nprocesses)


if __name__ == '__main__':
    main()







