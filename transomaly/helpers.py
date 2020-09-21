import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline



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


class ErrorPropagationSpline(object):
    """
    Does a spline fit, but returns both the spline value and associated uncertainty.
    https://gist.github.com/thriveth/4680e3d3cd2cfe561a57
    """
    def __init__(self, x, y, yerr, N=1000, *args, **kwargs):
        """
        See docstring for InterpolatedUnivariateSpline
        """
        try:
            yy = np.vstack([y + np.random.normal(loc=0, scale=yerr) for i in range(N)]).T
            self._splines = [spline(x, yy[:, i], *args, **kwargs) for i in range(N)]
        except ValueError:
            # Error because x must be strictly increasing. Removing consecutive duplicates
            repeated_indexes = np.where(np.diff(x) == 0)[0]
            y = np.delete(y, repeated_indexes)
            yerr = np.delete(yerr, repeated_indexes)
            x = np.delete(x, repeated_indexes)

            yy = np.vstack([y + np.random.normal(loc=0, scale=yerr) for i in range(N)]).T
            self._splines = [spline(x, yy[:, i], *args, **kwargs) for i in range(N)]

    def __call__(self, x, *args, **kwargs):
        """
        Get the spline value and uncertainty at point(s) x. args and kwargs are passed to spline.__call__
        :param x:
        :return: a tuple with the mean value at x and the standard deviation
        """
        x = np.atleast_1d(x)
        s = np.vstack([curve(x, *args, **kwargs) for curve in self._splines])
        return (np.mean(s, axis=0), np.std(s, axis=0))
