import os
import numpy as np
from collections import OrderedDict
import json
from six.moves.urllib.request import urlopen
from astroquery.irsa_dust import IrsaDust
import astropy.coordinates as coord
import astropy.units as u

from transomaly.get_anomaly_scores import TransientRegressor, GetAllTransientRegressors


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


def classify_lasair_light_curves(object_names=('ZTF18acsovsw',), plot=True, figdir='.', plot_animation=False):
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
    anomaly_scores, anomaly_scores_std, objids = regressor.get_anomaly_scores()
    # np.save(os.path.join(figdir, 'saved_anomaly_scores.npy'), np.array(anomaly_scores))
    # np.save(os.path.join(figdir, 'objids.npy'), objids)
    print(predictions)
    regressor.plot_anomaly_scores(fig_dir=figdir)

    for light_curve in light_curve_list:
        try:
            many_regressors = GetAllTransientRegressors(model_classes='all', nsamples=1)
            predictions, time_steps, objids = many_regressors.get_regressor_predictions([light_curve], return_predictions_at_obstime=True)
            anomaly_scores, anomaly_scores_std, objids = many_regressors.get_anomaly_scores(return_predictions_at_obstime=True)
            many_regressors.plot_anomaly_scores_all_models(anomaly_scores, time_steps, indexes_to_plot=None, fig_dir=figdir, plot_animation=plot_animation)
        except Exception as e:
            print(e)
            continue



    return regressor.lcs, regressor.timesX, regressor.gp_fits, regressor.y_predict, regressor.y


if __name__ == '__main__':
    # classify_lasair_light_curves(object_names=['ZTF19aazcxwk', 'ZTF19abauzma',
    #                                            'ZTF18abxftqm',  # TDE
    #                                            'ZTF19aadnmgf',  # SNIa
    #                                            'ZTF18acmzpbf',  # SNIa
    #                                            'ZTF19aakzwao',  # SNIa
    #                                            'ZTF19abplzzk'
    #                                            ], figdir='real_ZTF_objects')

    # # Type Ia SNe in ZTF from OSC
    # snia_names = []
    # import json
    # with open('/Users/danmuth/PycharmProjects/transomaly/ZTF_SNIa.json') as json_file:
    #     data = json.load(json_file)[:200]
    # for i, sn in enumerate(data):
    #     try:
    #         snia_name = [name for name in data[i]['Name'].split() if 'ZTF' in name][0]
    #         snia_z = float(data[i]['z'].split()[0])
    #         snia_names.append((snia_name, snia_z))
    #         print(i, snia_name, snia_z)
    #     except IndexError as e:
    #         print(f"failed on {i} {data[i]['Name']}")

    classify_lasair_light_curves([('ZTF18abcrxoj', 0.0309), ('ZTF19abzrhgq', 0.0151), ('ZTF19abuhlxk', 0.02), ('ZTF19abylxyt', 0.0533)], plot_animation=True)
    # classify_lasair_light_curves(object_names=['ZTF18aavrmcg','ZTF18acahuph','ZTF18acenqto', 'ZTF18aapgrxo', 'ZTF18abmasep', ], figdir='real_ZTF_objects/real_snia_from_osc')

    #
    # anomaly_scores = np.load(os.path.join('real_ZTF_objects/real_snia_from_osc', 'saved_anomaly_scores.npy'))
    # objids = np.load(os.path.join('real_ZTF_objects/real_snia_from_osc', 'objids.npy'))

    # nuclear_transients_080819 = np.unique(('ZTF19abfdupx', 'ZTF19abfjjwc', 'ZTF19abfjjwc', 'ZTF19abflufo', 'ZTF19abfdupx', 'ZTF19abfqmqz', 'ZTF19abfrydu', 'ZTF19abfvhlx', 'ZTF19abgbdcp', 'ZTF19abgbdcp', 'ZTF19abfdupx', 'ZTF19abflufo', 'ZTF19abfdupx', 'ZTF19abfqmqz', 'ZTF19abfqmqz', 'ZTF19abezcns', 'ZTF19abggmrt', 'ZTF19abgbtla', 'ZTF19abgcbey', 'ZTF19abfvhlx', 'ZTF19abgjlef', 'ZTF19abfvhlx', 'ZTF19abfwfiq', 'ZTF19abfwhja', 'ZTF19abglmpf', 'ZTF19abgbdcp', 'ZTF19abglmpf', 'ZTF19abfjjwc', 'ZTF19abgncfz', 'ZTF19abgffaj', 'ZTF19abipkyb', 'ZTF19abfrydu', 'ZTF19abgcbey', 'ZTF19abfrydu', 'ZTF19abgpnge', 'ZTF19abgncfz', 'ZTF19abgbtla', 'ZTF19abgcbey', 'ZTF19abglmpf', 'ZTF19abfvhlx', 'ZTF19abgjlef', 'ZTF18abtlqoh', 'ZTF19abfvfxp', 'ZTF19abgrhzk', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abgrhzk', 'ZTF19abfwfiq', 'ZTF19abgdcyx', 'ZTF19abglmpf', 'ZTF19abgrcxs', 'ZTF19abgmkef', 'ZTF19abfzzxy', 'ZTF19abgbdcp', 'ZTF19abfjjwc', 'ZTF19abgncfz', 'ZTF19abgbtla', 'ZTF19abfdupx', 'ZTF19abglmpf', 'ZTF19aambfxc', 'ZTF19abfdupx', 'ZTF19abeeqoj', 'ZTF19aambfxc', 'ZTF19abfdupx', 'ZTF19abgctni', 'ZTF19abgjfoj', 'ZTF19abhbkdd', 'ZTF19abgpnge', 'ZTF19abgcbey', 'ZTF19abgppki', 'ZTF19abgpnge', 'ZTF19abfrydu', 'ZTF19abgppki', 'ZTF19abgpjed', 'ZTF19abgbtla', 'ZTF19abglmpf', 'ZTF19abflufo', 'ZTF19abghldi', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abgqksj', 'ZTF19abgpjed', 'ZTF19abheryh', 'ZTF19abgbtla', 'ZTF19abfdupx', 'ZTF19abghldi', 'ZTF19abglmpf', 'ZTF19abgdcyx', 'ZTF19abghldi', 'ZTF19abglmpf', 'ZTF19abgmkef', 'ZTF19abgrcxs', 'ZTF19abfwhja', 'ZTF19abgbdcp', 'ZTF19abfjjwc', 'ZTF19abgffaj', 'ZTF19abgjfoj', 'ZTF19abfqmqz', 'ZTF19abfvhlx', 'ZTF19abhzdjp', 'ZTF19abiaxpi', 'ZTF19abgvfst', 'ZTF19abglmpf', 'ZTF18aarwxum', 'ZTF19abiiqcu', 'ZTF18aarwxum', 'ZTF19abgjlef', 'ZTF19abicwzc', 'ZTF19abgjlef', 'ZTF19abgqksj', 'ZTF19abhjhes', 'ZTF19abfjjwc', 'ZTF19abhceez', 'ZTF19abheryh', 'ZTF19abgrhzk', 'ZTF19abhenjb', 'ZTF19abgdcyx', 'ZTF19abgrhzk', 'ZTF19abhenjb', 'ZTF19abgbdcp', 'ZTF18aamasph', 'ZTF19abgncfz', 'ZTF19abgdcyx', 'ZTF19abglmpf', 'ZTF19abgpydp', 'ZTF19abglmpf', 'ZTF19abghldi', 'ZTF19abgpydp', 'ZTF19abgctni', 'ZTF19abidfag', 'ZTF19abipnwp', 'ZTF19abfvhlx', 'ZTF19abipktm', 'ZTF19abfqmqz', 'ZTF19abgfnmt', 'ZTF19abipnwp', 'ZTF19abfzwpe', 'ZTF19abiptub', 'ZTF19abfvhlx', 'ZTF19abgfnmt', 'ZTF19abiqpux', 'ZTF19abiszzn', 'ZTF19abgpjed', 'ZTF19abhzdjp', 'ZTF19abgdcyx', 'ZTF19aambfxc', 'ZTF19abgcbey', 'ZTF19abghldi', 'ZTF19abgvfst', 'ZTF19abhzdjp', 'ZTF19abfrydu', 'ZTF19abgncfz', 'ZTF19abiszzn', 'ZTF19abgpjed', 'ZTF19abggmrt', 'ZTF19abgppki', 'ZTF19abiietd', 'ZTF19abgcbey', 'ZTF19abglmpf', 'ZTF19abgqksj', 'ZTF19abgjlef', 'ZTF19abgjoth', 'ZTF19abidbya', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abgrcxs', 'ZTF19abgqksj', 'ZTF19abgqksj', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abgjlef', 'ZTF19abgrcxs', 'ZTF19abfwhja', 'ZTF19abgncfz', 'ZTF19abfjjwc', 'ZTF19abgvfst', 'ZTF19abglmpf', 'ZTF19abgvfst', 'ZTF19abfdupx', 'ZTF19abjgbgc', 'ZTF19abgmjtu', 'ZTF19abjgdko', 'ZTF19abicvxs', 'ZTF19aambfxc', 'ZTF18aarwxum', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abiijov', 'ZTF18aarwxum', 'ZTF19abfdupx', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abgpydp', 'ZTF19abgctni', 'ZTF19abipkyb', 'ZTF19abgjfoj', 'ZTF19abgffaj', 'ZTF19abiptub', 'ZTF19abfzwpe', 'ZTF19abjnzsi', 'ZTF19abiovhj', 'ZTF19abhdxcs', 'ZTF19abgjfoj', 'ZTF19abidfag', 'ZTF19abgmmfu', 'ZTF19abipmfl', 'ZTF19abisbgx', 'ZTF19abgrcxs', 'ZTF19aaqfrrl', 'ZTF19abfzzxy', 'ZTF19abhdlxp', 'ZTF19abhdvme', 'ZTF19abhusrq', 'ZTF19abisbgx', 'ZTF19abhdlxp', 'ZTF19abjravi', 'ZTF19abiietd', 'ZTF19abhenjb', 'ZTF19abgbdcp', 'ZTF19abglmpf', 'ZTF19abfdupx', 'ZTF18aarwxum', 'ZTF19abgppki', 'ZTF19abgpnge', 'ZTF19abiietd', 'ZTF19abgpjed', 'ZTF19abgbdcp', 'ZTF19abgncfz', 'ZTF19abhzdjp', 'ZTF19abjibet', 'ZTF19abiszzn', 'ZTF19abgppki', 'ZTF19abhoyxd', 'ZTF19abiaxpi', 'ZTF19abfvhlx', 'ZTF19abidbya', 'ZTF19abiptrq', 'ZTF19abgjlef', 'ZTF19abgqksj', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abidbya', 'ZTF19abfvhlx', 'ZTF19abiovhj', 'ZTF19abipktm', 'ZTF19abkcbri', 'ZTF19abgjlef', 'ZTF19abkfmjp', 'ZTF19abkfxfb', 'ZTF19abhdxcs', 'ZTF19abgrcxs', 'ZTF19abgafkt', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abkfxfb', 'ZTF19abjgdko', 'ZTF19abhdxcs', 'ZTF18aarwxum', 'ZTF18aarwxum', 'ZTF19abkcbri', 'ZTF19ablesob', 'ZTF19abidbya', 'ZTF19abgmmfu', 'ZTF19abipmfl', 'ZTF19abjnzsi', 'ZTF19abiqqve', 'ZTF19abgmjtu', 'ZTF19abicvxs', 'ZTF19abjgbwx', 'ZTF19abjgazt', 'ZTF19abjgdxx', 'ZTF19abjgdko', 'ZTF19ablesob', 'ZTF19abgdcyx', 'ZTF19abjibet', 'ZTF19abgvfst', 'ZTF19abghldi', 'ZTF18aarwxum', 'ZTF19abjibet', 'ZTF19abfdupx', 'ZTF19abghldi', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abgctni', 'ZTF19abgpydp', 'ZTF19abgmmfu', 'ZTF19abipmfl', 'ZTF19abidfag', 'ZTF19abjnzsi', 'ZTF19abiovhj', 'ZTF19abiovhj', 'ZTF19abiopky', 'ZTF19abgmmfu', 'ZTF19abgpydp', 'ZTF19abiptub', 'ZTF19abhusrq', 'ZTF19abgrcxs', 'ZTF19abisbgx', 'ZTF19abhdvme', 'ZTF19abhdvme', 'ZTF19abjravi', 'ZTF19abhdlxp', 'ZTF19abhdvme', 'ZTF19abhusrq', 'ZTF19abirbnk', 'ZTF19abiqpux', 'ZTF19ablovot', 'ZTF19abjpick', 'ZTF19aaqfrrl', 'ZTF19abgppki', 'ZTF19abhbtlo', 'ZTF18aawfquu', 'ZTF19abhzdjp', 'ZTF19abjioie', 'ZTF19abgcbey', 'ZTF19abhzdjp', 'ZTF19abiszzn', 'ZTF19abgppki', 'ZTF19abiietd', 'ZTF19abfdupx', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF19abfdupx', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abfvhlx', 'ZTF19abipktm', 'ZTF19abfvhlx', 'ZTF19abgqksj', 'ZTF19abiyyun', 'ZTF19abidbya', 'ZTF19abgjlef', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abiztbh', 'ZTF19abgrcxs', 'ZTF19ablesob', 'ZTF19abgjlef', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abkdeae', 'ZTF19ablesob', 'ZTF19abgrcxs', 'ZTF19abhjhes', 'ZTF19abjpimw', 'ZTF19ablesob', 'ZTF19abjgaye', 'ZTF19abjgbgc', 'ZTF19abjgcad', 'ZTF19abgmjtu', 'ZTF19abjgdko', 'ZTF19abkfxfb', 'ZTF19abkfmjp', 'ZTF19abgmjtu', 'ZTF19abkfxfb', 'ZTF19abjgdko', 'ZTF19abgncfz', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abgdcyx', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abjmnlw', 'ZTF19abirbnk', 'ZTF19abhusrq', 'ZTF19abhdxcs', 'ZTF19abhusrq', 'ZTF19abjpord', 'ZTF19abiqpux', 'ZTF19ablovot', 'ZTF19abjqytt', 'ZTF19abisbgx', 'ZTF19ablovot', 'ZTF19abgrcxs', 'ZTF19aaqfrrl', 'ZTF19abhoyxd', 'ZTF19abgppki', 'ZTF19abiszzn', 'ZTF19abfrydu', 'ZTF19abhzdjp', 'ZTF19abjioie', 'ZTF19abgcbey', 'ZTF19abjioie', 'ZTF19abhzdjp', 'ZTF19abgncfz', 'ZTF19abgdcyx', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF19abljkea', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abljinr', 'ZTF19abfvhlx', 'ZTF19abipktm', 'ZTF19ablesob', 'ZTF19abgjlef', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abkdlkl', 'ZTF19abgjlef', 'ZTF19abgmkef', 'ZTF19abhjhes', 'ZTF19abkdeae', 'ZTF19ablesob', 'ZTF19abhjhes', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abgdcyx', 'ZTF19abjibet', 'ZTF19abglmpf', 'ZTF19abixauz', 'ZTF19abipnwp', 'ZTF19abkevcb', 'ZTF19abipkyb', 'ZTF19abixauz', 'ZTF19abkfmjp', 'ZTF19abiqifg', 'ZTF19abkdeae', 'ZTF19ablesob', 'ZTF19ablybjv', 'ZTF19abgdcyx', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF19abljkea', 'ZTF19abglmpf', 'ZTF19abgdcyx', 'ZTF19abjibet', 'ZTF18aarwxum', 'ZTF19abglmpf', 'ZTF19abljinr', 'ZTF19abgctni', 'ZTF19abgcbey', 'ZTF19abjibet', 'ZTF19abgvfst', 'ZTF18aarwxum', 'ZTF19abghldi', 'ZTF19abglmpf'))
    # nuclear_transients_210819 = np.unique(("ZTF18abccpuw", "ZTF19abglmpf", "ZTF19ablovot", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF18acvqugw", "ZTF18acrttfl", "ZTF19abjmnlw", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aabfjyo", "ZTF18aanxnrz", "ZTF18aanxnrz", "ZTF18aaojsvk", "ZTF18abvbcln", "ZTF19ablovot", "ZTF19ablovot", "ZTF19ablovot", "ZTF19ablovot", "ZTF19ablovot", "ZTF18abmogca"))
    # # classify_lasair_light_curves(object_names=nuclear_transients_210819, figdir='real_ZTF_objects/nuclear_transients', plot_animation=False)
    #
    tde_candidates = [('ZTF19aapreis', 0.0512),
                      'ZTF18actaqdw',
                      ('ZTF18aahqkbt', 0.051),
                      ('ZTF18abxftqm', 0.09),
                      ('ZTF19aabbnzo', 0.08),
                      'ZTF18acpdvos',
                      'ZTF18aabtxvd',
                      ('ZTF19aarioci', 0.12)]
    classify_lasair_light_curves(object_names=tde_candidates, figdir='real_ZTF_objects')