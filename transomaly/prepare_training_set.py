import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from astrorapid.get_training_data import get_data
from transomaly.fit_gaussian_processes import save_gps
from transomaly.prepare_arrays import PrepareArrays
from transomaly.helpers import delete_indexes_from_arrays


class PrepareTrainingSetArrays(PrepareArrays):
    def __init__(self, passbands=('g', 'r'), contextual_info=('redshift',), data_dir='data/ZTF_20190512/',
                 save_dir='data/saved_light_curves/', training_set_dir='data/training_set_files/', redo=False,
                 get_data_func=None, use_gp_interp=False):
        PrepareArrays.__init__(self, passbands, contextual_info, use_gp_interp)
        self.passbands = passbands
        self.contextual_info = contextual_info
        self.nobs = 50
        self.npb = len(passbands)
        self.nfeatures = self.npb + len(self.contextual_info)
        self.timestep = 3.0
        self.mintime = -70
        self.maxtime = 80
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.training_set_dir = training_set_dir
        self.redo = redo
        self.get_data_func = get_data_func

        if 'redshift' in contextual_info:
            self.known_redshift = True
        else:
            self.known_redshift = False

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.training_set_dir):
            os.makedirs(self.training_set_dir)

    def get_light_curves(self, class_num, nprocesses=1):
        light_curves = get_data(get_data_func=self.get_data_func, class_num=class_num, data_dir=self.data_dir,
                                save_dir=self.save_dir, passbands=self.passbands, known_redshift=self.known_redshift,
                                nprocesses=nprocesses, redo=self.redo, calculate_t0=False)

        return light_curves

    def get_gaussian_process_fits(self, light_curves, class_num, plot=False, nprocesses=1, extrapolate=True):
        saved_gp_fits = save_gps(light_curves, self.save_dir, class_num, self.passbands, plot=plot,
                                 nprocesses=nprocesses, redo=self.redo, extrapolate=extrapolate)

        return saved_gp_fits

    def make_arrays(self, light_curves, saved_gp_fits, nsamples, extrapolate=True):
        nobjects = len(light_curves)
        nrows = nobjects * nsamples

        labels = np.empty(shape=nrows, dtype=object)
        # X = np.memmap(os.path.join(self.training_set_dir, 'X_lc_data.dat'), dtype=np.float32, mode='w+',
        #               shape=(nrows, self.nfeatures, self.nobs))
        # Xerr = np.memmap(os.path.join(self.training_set_dir, 'Xerr_lc_data.dat'), dtype=np.float32, mode='w+',
        #               shape=(nrows, self.nfeatures, self.nobs))
        X = np.zeros(shape=(nrows, self.nfeatures, self.nobs))
        Xerr = np.zeros(shape=(nrows, self.npb, self.nobs))
        timesX = np.zeros(shape=(nrows, self.nobs))
        objids = []

        for i, (objid, lc) in enumerate(light_curves.items()):
            idx = i * nsamples
            if i % 100 == 0:
                print(i, nobjects)
            if self.use_gp_interp:
                gp_lc = saved_gp_fits[objid]
            else:
                gp_lc = None

            redshift = lc.meta['redshift']
            b = lc.meta['b']
            mwebv = lc.meta['mwebv']
            trigger_mjd = lc.meta['trigger_mjd']

            # TODO: make cuts

            tinterp, len_t = self.get_t_interp(lc, extrapolate=extrapolate)
            for ns in range(nsamples):
                timesX[idx + ns][0:len_t] = tinterp
                objids.append(objid)
                labels[idx + ns] = lc.meta['class_num']
            X, Xerr = self.update_X(X, Xerr, idx, gp_lc, lc, tinterp, len_t, objid, self.contextual_info, lc.meta, nsamples)

        # Count nobjects per class
        classes = sorted(list(set(labels)))
        for c in classes:
            nobs = len(X[labels == c])
            print(f"class {c}: {nobs}")

        return X, Xerr, timesX, labels, objids

    def make_training_set(self, class_nums=(1,), nsamples=10, otherchange='', nprocesses=1, extrapolate_gp=True, reframe=False, npred=49, normalise=False, use_uncertainties=False, ignore_objids=(), only_use_objids=None, train_size=0.8):
        savepath = os.path.join(self.training_set_dir, "X_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums))

        if self.redo is True or not os.path.isfile(savepath):
            light_curves = {}
            saved_gp_fits = {}
            for class_num in class_nums:
                lcs = self.get_light_curves(class_num, nprocesses)
                light_curves.update(lcs)
                if self.use_gp_interp:
                    gps = self.get_gaussian_process_fits(lcs, class_num, plot=False, nprocesses=nprocesses, extrapolate=extrapolate_gp)
                    saved_gp_fits.update(gps)
                    # Find intersection of dictionaries
                    objids = list(set(light_curves.keys()) & set(saved_gp_fits.keys()))
                else:
                    objids = list(set(light_curves.keys()))

            # Only use objids in only_use_objids unless not specified
            if only_use_objids is not None and len(only_use_objids) >= 1:
                print("Removing objects that were not in only_use_objids:", set(objids) - set(only_use_objids))
                objids = list(set(only_use_objids) & set(objids))

            # Remove objids in ignore_objids
            for objid in ignore_objids:
                objids.remove(objid)
                print(f"Ignoring object: {objid}")

            # Train test split for light_ curves and GPs
            objids_train, objids_test = train_test_split(objids, train_size=train_size, shuffle=True, random_state=42)
            lcs_train = {k: light_curves[k] for k in objids_train}
            lcs_test = {k: light_curves[k] for k in objids_test}
            if self.use_gp_interp:
                gps_train = {k: saved_gp_fits[k] for k in objids_train}
                gps_test = {k: saved_gp_fits[k] for k in objids_test}
            else:
                gps_train = None
                gps_test = None

            X_train, Xerr_train, timesX_train, labels_train, objids_train = self.make_arrays(lcs_train, gps_train, nsamples, extrapolate_gp)
            X_test, Xerr_test, timesX_test, labels_test, objids_test = self.make_arrays(lcs_test, gps_test, nsamples, extrapolate_gp)

            # Shuffle training set but leave testing set in order or gaussian process samples
            X_train, Xerr_train, timesX_train, labels_train, objids_train = shuffle(X_train, Xerr_train, timesX_train, labels_train, objids_train, random_state=42)

            np.save(os.path.join(self.training_set_dir, "X_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), X_train)
            np.save(os.path.join(self.training_set_dir, "Xerr_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), Xerr_train)
            np.save(os.path.join(self.training_set_dir, "timesX_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), timesX_train)
            np.save(os.path.join(self.training_set_dir, "labels_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), labels_train, allow_pickle=True)
            np.save(os.path.join(self.training_set_dir, "objids_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), objids_train, allow_pickle=True)
            np.save(os.path.join(self.training_set_dir, "X_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), X_test)
            np.save(os.path.join(self.training_set_dir, "Xerr_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), Xerr_test)
            np.save(os.path.join(self.training_set_dir, "timesX_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), timesX_test)
            np.save(os.path.join(self.training_set_dir, "labels_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), labels_test, allow_pickle=True)
            np.save(os.path.join(self.training_set_dir, "objids_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), objids_test, allow_pickle=True)
        else:
            X_train = np.load(os.path.join(self.training_set_dir, "X_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), mmap_mode='r')
            Xerr_train = np.load(os.path.join(self.training_set_dir, "Xerr_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), mmap_mode='r')
            timesX_train = np.load(os.path.join(self.training_set_dir, "timesX_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)))
            labels_train = np.load(os.path.join(self.training_set_dir, "labels_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), allow_pickle=True)
            objids_train = np.load(os.path.join(self.training_set_dir, "objids_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), allow_pickle=True)
            X_test = np.load(os.path.join(self.training_set_dir, "X_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), mmap_mode='r')
            Xerr_test = np.load(os.path.join(self.training_set_dir, "Xerr_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), mmap_mode='r')
            timesX_test = np.load(os.path.join(self.training_set_dir, "timesX_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)))
            labels_test = np.load(os.path.join(self.training_set_dir, "labels_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), allow_pickle=True)
            objids_test = np.load(os.path.join(self.training_set_dir, "objids_test_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums)), allow_pickle=True)


        X_train = X_train.swapaxes(2, 1)  # Correct shape for keras is (N_objects, N_timesteps, N_passbands)
        Xerr_train = Xerr_train.swapaxes(2, 1)
        X_test = X_test.swapaxes(2, 1)  # Correct shape for keras is (N_objects, N_timesteps, N_passbands)
        Xerr_test = Xerr_test.swapaxes(2, 1)

        #
        # Normalise light curves
        if normalise:
            nobjects, ntimesteps, nfeatures = X_train.shape
            npassbands = len(self.passbands)
            X_train_normalised = np.zeros(X_train.shape)
            Xerr_train_normalised = np.zeros(Xerr_train.shape)
            for i in range(nobjects):
                for pbidx in range(npassbands):
                    flux = X_train[i, :, pbidx]
                    fluxerr = Xerr_train[i, :, pbidx]
                    minflux = min(flux)
                    maxflux = max(flux)
                    if maxflux - minflux == 0:
                        norm = 1
                    else:
                        norm = maxflux - minflux
                    mask_zeros = (flux == 0)
                    fluxnorm = (flux - minflux)/norm
                    fluxnorm[mask_zeros] = 0
                    X_train_normalised[i, :, pbidx] = fluxnorm
                    Xerr_train_normalised[i, :, pbidx] = fluxerr / norm
            nobjects, ntimesteps, nfeatures = X_test.shape
            X_test_normalised = np.zeros(X_test.shape)
            Xerr_test_normalised = np.zeros(Xerr_test.shape)
            for i in range(nobjects):
                for pbidx in range(npassbands):
                    flux = X_test[i, :, pbidx]
                    fluxerr = Xerr_test[i, :, pbidx]
                    minflux = min(flux)
                    maxflux = max(flux)
                    if maxflux - minflux == 0:
                        norm = 1
                    else:
                        norm = maxflux - minflux
                    X_test_normalised[i, :, pbidx] = (flux - minflux)/norm
                    Xerr_test_normalised[i, :, pbidx] = fluxerr / norm
            X_train = X_train_normalised
            X_test = X_test_normalised
            Xerr_train = Xerr_train_normalised
            Xerr_test = Xerr_test_normalised
        #

        if reframe is True:
            # Reframe X and y
            newX_train = []
            newy_train = []
            newXerr_train = []
            newyerr_train = []
            nobjects, ntimesteps, npassbands = X_train.shape
            for i in range(nobjects):
                for j in range(1, npred+1):#ntimesteps):
                    newX_entry = np.zeros(X_train[i,:,:].shape)
                    newX_entry[:-j,:] = X_train[i,:-j,:]
                    newXerr_entry = np.zeros(Xerr_train[i, :, :].shape)
                    newXerr_entry[:-j, :] = Xerr_train[i, :-j, :]
                    newy_entry = X_train[i, -j, :]
                    newyerr_entry = Xerr_train[i, -j, :]
                    newX_train.append(newX_entry)
                    newXerr_train.append(newXerr_entry)
                    newy_train.append(newy_entry)
                    newyerr_train.append(newyerr_entry)
            newX_test = []
            newy_test = []
            newXerr_test = []
            newyerr_test = []
            nobjects, ntimesteps, npassbands = X_test.shape
            for i in range(nobjects):
                for j in range(1, npred+1):#ntimesteps):
                    newX_entry = np.zeros(X_test[i,:,:].shape)
                    newX_entry[:-j,:] = X_test[i,:-j,:]
                    newXerr_entry = np.zeros(Xerr_test[i, :, :].shape)
                    newXerr_entry[:-j, :] = Xerr_test[i, :-j, :]
                    newy_entry = X_test[i, -j, :]
                    newyerr_entry = Xerr_test[i, -j, :]
                    newX_test.append(newX_entry)
                    newy_test.append(newy_entry)
                    newXerr_test.append(newXerr_entry)
                    newyerr_test.append(newyerr_entry)
            X_train = np.array(newX_train)
            X_test = np.array(newX_test)
            y_train = np.array(newy_train)
            y_test = np.array(newy_test)
            Xerr_train = np.array(newXerr_train)
            Xerr_test = np.array(newXerr_test)
            yerr_train = np.array(newyerr_train)
            yerr_test = np.array(newyerr_test)
            # timesX_train = timesX_train[:, :-1]
            # timesX_test = timesX_test[:, :-1]
        else:
            n_pred = npred
            # nobjects, ntimesteps, npassbands = X_train.shape
            # y_train = np.zeros((nobjects, ntimesteps-n_pred, npassbands*n_pred)) # np.tile(np.hstack((X_train[:, 1:, 0], X_train[:, 1:, 1])), (49,1,1)).swapaxes(0,1)  # X_train[:, 1:, :2]
            # yerr_train = np.zeros((nobjects, ntimesteps-n_pred, npassbands*n_pred))
            # for t in range(ntimesteps-n_pred):
            #     y_train[:,t,:] = X_train[:,(t+1):(t+1+n_pred),:2].reshape((nobjects,npassbands*n_pred))
            #     yerr_train[:,t,:] = Xerr_train[:,(t+1):(t+1+n_pred),:2].reshape((nobjects,npassbands*n_pred))
            y_train = X_train[:, n_pred:, :2]
            yerr_train = Xerr_train[:, n_pred:, :2]
            X_train = X_train[:, :-n_pred]
            Xerr_train = Xerr_train[:, :-n_pred]
            # timesX_train = timesX_train[:, :-1]

            # nobjects, ntimesteps, npassbands = X_test.shape
            # y_test = np.zeros((nobjects, ntimesteps-n_pred, npassbands*n_pred)) # np.tile(np.hstack((X_train[:, 1:, 0], X_train[:, 1:, 1])), (49,1,1)).swapaxes(0,1)  # X_train[:, 1:, :2]
            # yerr_test = np.zeros((nobjects, ntimesteps-n_pred, npassbands*n_pred))
            # for t in range(ntimesteps-n_pred):
            #     y_test[:,t,:] = X_test[:,(t+1):(t+1+n_pred),:2].reshape((nobjects,npassbands*n_pred))
            #     yerr_test[:,t,:] = Xerr_test[:, (t + 1):(t + 1 + n_pred), :2].reshape((nobjects, npassbands * n_pred))
            y_test = X_test[:, n_pred:, :self.npb]
            yerr_test = Xerr_test[:, n_pred:, :self.npb]
            X_test = X_test[:, :-n_pred]
            Xerr_test = Xerr_test[:, :-n_pred]
            # timesX_test = timesX_test[:, :-1]

            # # Delete indexes where any errors are zero
            # delete_indexes = np.unique(np.where(Xerr_train == 0)[0])
            # print("Deleting indexes where any uncertainties are zero for objids", objids_train[delete_indexes])
            # X_train, y_train, Xerr_train, yerr_train, timesX_train, labels_train, objids_train = delete_indexes_from_arrays(delete_indexes, 0, X_train, y_train, Xerr_train, yerr_train, timesX_train, labels_train, objids_train)
            # delete_indexes = np.unique(np.where(Xerr_test == 0)[0])
            # print("Deleting indexes where any uncertainties are zero for objids", objids_test[delete_indexes])
            # X_test, y_test, Xerr_test, yerr_test, timesX_test, labels_test, objids_test = delete_indexes_from_arrays(delete_indexes, 0, X_test, y_test, Xerr_test, yerr_test, timesX_test, labels_test, objids_test)

            if use_uncertainties:
                # Add errors as extra column to y and X
                ye_train = np.copy(yerr_train)
                ye_test = np.copy(yerr_test)
                y_train = np.dstack((y_train, ye_train))
                y_test = np.dstack((y_test, ye_test))

                Xe_train = np.copy(Xerr_train)
                Xe_test = np.copy(Xerr_test)
                X_train = np.dstack((X_train, Xe_train))
                X_test = np.dstack((X_test, Xe_test))

        return X_train, X_test, y_train, y_test, Xerr_train, Xerr_test, yerr_train, yerr_test, \
               timesX_train, timesX_test, labels_train, labels_test, objids_train, objids_test


def main():
    passbands = ('g', 'r')
    contextual_info = ('redshift',)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(SCRIPT_DIR, '..', 'data/ZTF_20190512')
    save_dir = os.path.join(SCRIPT_DIR, '..', 'data/saved_light_curves')
    training_set_dir = os.path.join(SCRIPT_DIR, '..', 'data/training_set_files')
    nprocesses = 1
    class_nums = (1,)
    otherchange = ''
    nsamples = 100
    extrapolate_gp = True

    prepare_training_set = PrepareTrainingSetArrays(passbands, contextual_info, data_dir, save_dir, training_set_dir)
    prepare_training_set.make_training_set(class_nums, nsamples, otherchange, nprocesses, extrapolate_gp)


if __name__ == '__main__':
    main()
