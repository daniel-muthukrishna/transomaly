import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from transomaly.get_training_data import get_data
from transomaly.fit_gaussian_processes import save_gps
from transomaly.prepare_arrays import PrepareArrays


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

    def get_gaussian_process_fits(self, light_curves, class_num, plot=False, nprocesses=1, extrapolate=True):
        saved_gp_fits = save_gps(light_curves, self.save_dir, class_num, self.passbands, plot=plot,
                                 nprocesses=nprocesses, redo=self.redo, extrapolate=extrapolate)

        return saved_gp_fits

    def make_training_set(self, class_nums=(1,), nsamples=10, otherchange='', nprocesses=1, extrapolate_gp=True, reframe=False, npred=49):
        savepath = os.path.join(self.training_set_dir, "X_train_{}_ci{}_ns{}_c{}.npy".format(otherchange, self.contextual_info, nsamples, class_nums))

        if self.redo is True or not os.path.isfile(savepath):
            light_curves = {}
            saved_gp_fits = {}
            for class_num in class_nums:
                lcs = self.get_light_curves(class_num, nprocesses)
                gps = self.get_gaussian_process_fits(lcs, class_num, plot=False, nprocesses=nprocesses, extrapolate=extrapolate_gp)
                light_curves.update(lcs)
                saved_gp_fits.update(gps)

            # Find intersection of dictionaries
            objids = list(set(light_curves.keys()) & set(saved_gp_fits.keys()))
            # objids = ['1_50075859', '1_50075859'] # '1_99285690', '1_99285690']  ####

            # Train test split for light_ curves and GPs
            objids_train, objids_test = train_test_split(objids, train_size=0.80, shuffle=True, random_state=42)
            lcs_train = {k: light_curves[k] for k in objids_train}
            gps_train = {k: saved_gp_fits[k] for k in objids_train}
            lcs_test = {k: light_curves[k] for k in objids_test}
            gps_test = {k: saved_gp_fits[k] for k in objids_test}

            X_train, Xerr_train, timesX_train, labels_train, objids_train = self.make_arrays(lcs_train, gps_train, nsamples, extrapolate_gp)
            X_test, Xerr_test, timesX_test, labels_test, objids_test = self.make_arrays(lcs_test, gps_test, nsamples, extrapolate_gp)

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
        X_test = X_test.swapaxes(2, 1)  # Correct shape for keras is (N_objects, N_timesteps, N_passbands)
        Xerr_test = Xerr_test.swapaxes(2, 1)

        # # Normalise light curves
        # nobjects, ntimesteps, npassbands = X_train.shape
        # X_train_normalised = np.zeros(X_train.shape)
        # Xerr_train_normalised = np.zeros(Xerr_train.shape)
        # for i in range(nobjects):
        #     for pbidx in range(npassbands):
        #         flux = X_train[i, :, pbidx]
        #         fluxerr = Xerr_train[i, :, pbidx]
        #         minflux = min(flux)
        #         maxflux = max(flux)
        #         if maxflux - minflux == 0:
        #             norm = 1
        #         else:
        #             norm = maxflux - minflux
        #         X_train_normalised[i, :, pbidx] = (flux - minflux)/norm
        #         Xerr_train_normalised[i, :, pbidx] = fluxerr / norm
        # nobjects, ntimesteps, npassbands = X_test.shape
        # X_test_normalised = np.zeros(X_test.shape)
        # Xerr_test_normalised = np.zeros(Xerr_test.shape)
        # for i in range(nobjects):
        #     for pbidx in range(npassbands):
        #         flux = X_test[i, :, pbidx]
        #         fluxerr = Xerr_test[i, :, pbidx]
        #         minflux = min(flux)
        #         maxflux = max(flux)
        #         if maxflux - minflux == 0:
        #             norm = 1
        #         else:
        #             norm = maxflux - minflux
        #         X_test_normalised[i, :, pbidx] = (flux - minflux)/norm
        #         Xerr_test_normalised[i, :, pbidx] = fluxerr / norm
        # X_train = X_train_normalised
        # X_test = X_test_normalised
        # Xerr_train = Xerr_train_normalised
        # Xerr_test = Xerr_test_normalised

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
            y_train = X_train[:, 1:, :2]
            yerr_train = Xerr_train[:, 1:, :2]
            X_train = X_train[:, :-1]
            Xerr_train = Xerr_train[:, :-1]
            # timesX_train = timesX_train[:, :-1]

            y_test = X_test[:, 1:, :2]
            yerr_test = Xerr_test[:, 1:, :2]
            X_test = X_test[:, :-1]
            Xerr_test = Xerr_test[:, :-1]
            # timesX_test = timesX_test[:, :-1]

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


def delete_indexes(deleteindexes, *args):
    newarrs = []
    for arr in args:
        newarr = np.delete(arr, deleteindexes)
        newarrs.append(newarr)

    return newarrs

if __name__ == '__main__':
    main()
