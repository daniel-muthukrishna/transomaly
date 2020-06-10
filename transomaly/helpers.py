import numpy as np


def get_sntypes():
    sntypes_map = {1: 'SNIa-norm',
                   11: 'SNIa-norm',
                   2: 'SNII',
                   12: 'SNII',
                   14: 'SNIIn',
                   3: 'SNIbc',
                   13: 'SNIbc',
                   5: 'SNIbc',
                   6: 'SNII',
                   41: 'SNIa-91bg',
                   43: 'SNIa-x',
                   45: 'point-Ia',
                   50: 'Kilonova-GW170817',
                   51: 'Kilonova',
                   60: 'SLSN-I',
                   61: 'PISN',
                   62: 'ILOT',
                   63: 'CART',
                   64: 'TDE',
                   70: 'AGN',
                   80: 'RRLyrae',
                   81: 'Mdwarf',
                   83: 'Eclip. Bin.',
                   84: 'Mira',
                   90: 'uLens-BSR',
                   91: 'uLens-1STAR',
                   92: 'uLens-String',
                   93: 'uLens - Point',
                   99: 'Rare'}
    return sntypes_map


def delete_indexes_from_arrays(delete_indexes, axis=None, *args):
    newarrs = []
    for arr in args:
        newarr = np.delete(arr, delete_indexes, axis=axis)
        newarrs.append(newarr)
        assert len(arr.shape) == len(newarr.shape)

    return newarrs


def find_nearest(array, value):
    """
    Find the index nearest to a given value.
    Adapted from: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return idx