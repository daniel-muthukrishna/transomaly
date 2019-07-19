import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from transomaly import helpers


def get_data(data_dir='data'):
    X = np.load(os.path.join(data_dir, "X.npy"), mmap_mode='r')
    y = np.load(os.path.join(data_dir, "y.npy"), mmap_mode='r')
    labels = np.load(os.path.join(data_dir, "labels.npy"), mmap_mode='r')
    timesX = np.load(os.path.join(data_dir, "tinterp.npy"), mmap_mode='r')
    objids_list = np.load(os.path.join(data_dir, "objid.npy"), mmap_mode='r')
    # with open(os.path.join(data_dir, "origlc.npy"), 'rb') as f:
    #     orig_lc = pickle.load(f)
    orig_lc = X

    # Correct shape for keras is (N_objects, N_timesteps, N_passbands) (where N_timesteps is lookback time)
    X = X.swapaxes(2, 1)

    X = X.copy()
    for i in range(len(X)):
        for pbidx in range(2):
            minX = X[i, :, pbidx].min(axis=0)
            maxX = X[i, :, pbidx].max(axis=0)
            X[i, :, pbidx] = (X[i, :, pbidx] - minX) / (maxX - minX)
    finitemask = ~np.any(np.any(~np.isfinite(X), axis=1), axis=1)
    X = X[finitemask]
    y = y[finitemask]
    timesX = timesX[finitemask]
    objids_list = objids_list[finitemask]
    orig_lc = orig_lc[finitemask]
    labels = labels[finitemask]

    # Use only SNIa
    X = X[labels == 1]
    y = y[labels == 1]
    timesX = timesX[labels == 1]
    objids_list = objids_list[labels == 1]
    orig_lc = orig_lc[labels == 1]
    labels = labels[labels == 1]



    classes = sorted(list(set(labels)))
    sntypes_map = helpers.get_sntypes()
    class_names = [sntypes_map[class_num] for class_num in classes]

    # Count nobjects per class
    for c in classes:
        nobs = len(X[labels == c])
        print(c, nobs)

    # Use class numbers 1,2,3... instead of 1, 3, 13 etc.
    y_indexes = np.copy(y)
    for i, c in enumerate(classes):
        y_indexes[y == c] = i + 1
    y = y_indexes

    y = to_categorical(y)

    X_train, X_test, y_train, y_test, labels_train, labels_test, timesX_train, timesX_test, orig_lc_train, \
    orig_lc_test, objids_train, objids_test = \
        train_test_split(X, y, labels, timesX, orig_lc, objids_list, train_size=0.80, shuffle=False, random_state=42)

    y_train = X_train[:, 1:, :2]
    y_test = X_test[:, 1:, :2]
    X_train = X_train[:, :-1]
    X_test = X_test[:, :-1]
    timesX_train = timesX_train[:, :-1]
    timesX_test = timesX_test[:, :-1]

    return X_train, X_test, y_train, y_test, timesX_train, timesX_test
