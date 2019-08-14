import numpy as np
from collections import OrderedDict
import json
from six.moves.urllib.request import urlopen
from astroquery.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u

from transomaly.get_anomaly_scores import TransientRegressor


def read_json(url):
    response = urlopen(url)
    return json.loads(response.read(), object_pairs_hook=OrderedDict)


def delete_indexes(deleteindexes, *args):
    newarrs = []
    for arr in args:
        newarr = np.delete(arr, deleteindexes)
        newarrs.append(newarr)

    return newarrs


def convert_lists_to_arrays(*args):
    output = []
    for arg in args:
        out_array = np.asarray(arg)
        output.append(out_array)

    return output


def read_lasair_json(object_name='ZTF18acsovsw'):
    """
    Read light curve from lasair website API based on object name.

    Parameters
    ----------
    object_name : str
        The LASAIR object name. E.g. object_name='ZTF18acsovsw'

    """
    print(object_name)
    if isinstance(object_name, tuple):
        object_name, z_in = object_name
    else:
        z_in = None

    url = 'https://lasair.roe.ac.uk/object/{}/json/'.format(object_name)

    data = read_json(url)

    objid = data['objectId']
    ra = data['objectData']['ramean']
    dec = data['objectData']['decmean']
    # lasair_classification = data['objectData']['classification']
    tns_info = data['objectData']['annotation']
    photoZ = None
    for cross_match in data['crossmatches']:
        # print(cross_match)
        photoZ = cross_match['photoZ']
        separation_arcsec = cross_match['separationArcsec']
        catalogue_object_type = cross_match['catalogue_object_type']
    if photoZ is None:  # TODO: Get correct redshift
        try:
            if "z=" in tns_info:
                photoZ = tns_info.split('z=')[1]
                redshift = float(photoZ.replace(')', '').split()[0])
            elif "Z=" in tns_info:
                photoZ = tns_info.split('Z=')[1]
                redshift = float(photoZ.split()[0])
            else:
                redshift = None
        except Exception as e:
            redshift = None
            print(e)
    else:
        redshift = photoZ
    if z_in is not None:
        redshift = z_in
    print("Redshift is {}".format(redshift))
    objid += "_z={}".format(round(redshift, 2))

    # Get extinction  TODO: Maybe add this to RAPID code
    coo = coord.SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
    dust = IrsaDust.get_query_table(coo, section='ebv')
    mwebv = dust['ext SandF mean'][0]
    print("MWEBV")
    print(mwebv)

    mjd = []
    passband = []
    mag = []
    magerr = []
    photflag = []
    zeropoint = []
    for cand in data['candidates']:
        mjd.append(cand['mjd'])
        passband.append(cand['fid'])
        mag.append(cand['magpsf'])
        if 'sigmapsf' in cand:
            magerr.append(cand['sigmapsf'])
            photflag.append(4096)
            if cand['magzpsci'] == 0:
                zeropoint.append(26.2)  # TODO: Tell LASAIR their zeropoints are wrong
            else:
                zeropoint.append(cand['magzpsci'])
        else:
            magerr.append(0.1 * cand['magpsf'])
            photflag.append(0)
            zeropoint.append(26.2)

    mjd, passband, mag, magerr, photflag, zeropoint = convert_lists_to_arrays(mjd, passband, mag, magerr, photflag, zeropoint)

    deleteindexes = np.where(magerr == None)
    mjd, passband, mag, magerr, photflag, zeropoint = delete_indexes(deleteindexes, mjd, passband, mag, magerr, photflag, zeropoint)

    return mjd, passband, mag, magerr, photflag, zeropoint, ra, dec, objid, redshift, mwebv


def classify_lasair_light_curves(object_names=('ZTF18acsovsw',), plot=True, figdir='.'):
    light_curve_list = []
    peakfluxes_g, peakfluxes_r, redshifts = [], [], []
    mjds, passbands, mags, magerrs,zeropoints, photflags = [], [], [], [], [], []
    obj_names = []
    ras = []
    decs = []
    peakmags_g, peakmags_r = [], []
    for object_name in object_names:
        try:
            mjd, passband, mag, magerr, photflag, zeropoint, ra, dec, objid, redshift, mwebv = read_lasair_json(object_name)
            sortidx = np.argsort(mjd)
            mjds.append(mjd[sortidx])
            passbands.append(passband[sortidx])
            mags.append(mag[sortidx])
            magerrs.append(magerr[sortidx])
            zeropoints.append(zeropoint[sortidx])
            photflags.append(photflag[sortidx])
            obj_names.append(object_name)
            ras.append(ra)
            decs.append(dec)
            peakmags_g.append(min(mag[passband==1]))
            peakmags_r.append(min(mag[passband==2]))

        except Exception as e:
            print(e)
            continue

        flux = 10. ** (-0.4 * (mag - zeropoint))
        fluxerr = np.abs(flux * magerr * (np.log(10.) / 2.5))

        passband = np.where((passband == 1) | (passband == '1'), 'g', passband)
        passband = np.where((passband == 2) | (passband == '2'), 'r', passband)

        mjd_first_detection = min(mjd[photflag == 4096])
        photflag[np.where(mjd == mjd_first_detection)] = 6144

        deleteindexes = np.where(((passband == 3) | (passband == '3')) | (mjd > mjd_first_detection) & (photflag == 0))
        if deleteindexes[0].size > 0:
            print("Deleting indexes {} at mjd {} and passband {}".format(deleteindexes, mjd[deleteindexes], passband[deleteindexes]))
        mjd, passband, flux, fluxerr, zeropoint, photflag = delete_indexes(deleteindexes, mjd, passband, flux, fluxerr, zeropoint, photflag)

        light_curve_list += [(mjd, flux, fluxerr, passband, photflag, ra, dec, objid, redshift, mwebv)]

        try:
            dummy = max(flux[passband == 'g'])
            dummy = max(flux[passband == 'r'])
        except Exception as e:
            print(e)
            continue

        peakfluxes_g.append(max(flux[passband == 'g']))
        peakfluxes_r.append(max(flux[passband == 'r']))
        redshifts.append(redshift)

    regressor = TransientRegressor()
    predictions, time_steps, objids = regressor.get_regressor_predictions(light_curve_list, return_predictions_at_obstime=False)
    anomaly_scores, anomaly_scores_std, objids = regressor.get_anomaly_scores(predictions)
    print(predictions)
    regressor.plot_anomaly_scores(fig_dir=figdir)

    return regressor.lcs, regressor.timesX, regressor.gp_fits, regressor.y_predict, regressor.y


if __name__ == '__main__':
    classify_lasair_light_curves(object_names=['ZTF19aazcxwk', 'ZTF19abauzma',
                                               'ZTF18abxftqm',  # TDE
                                               'ZTF19aadnmgf',  # SNIa
                                               'ZTF18acmzpbf',  # SNIa
                                               'ZTF19aakzwao',  # SNIa
                                               ], figdir='transomaly')
