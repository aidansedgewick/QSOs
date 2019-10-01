import numpy as np
import astropy.io.fits as fits
from matplotlib import pyplot as plt
import dla_cnn
from dla_cnn.data_model.Sightline import Sightline
parks_dir = '/'.join(dla_cnn.__file__.split('/')[:-1])+'/models/'
model_checkpoint = parks_dir + 'model_gensample_v7.1'
REST_RANGE = [900, 1346, 1748]


def print_stats(props):
    print("Confidence = {0:.2f}".format(props['dla_confidence']))
    print("N(H I) = {0:.3f} +/- {1:.3f}".format(props['column_density'], props['std_column_density']))
    print("z_abs = {0:.6f}".format(props['z_dla']))

def find_dlas(flux, loglam, z_qso, plot=False, print_summary=False):
    idnum = 0
    sl = Sightline(idnum, dlas=None, flux=flux, loglam=loglam, z_qso=z_qso)
    sl.process(model_checkpoint)
    if plot:
        lam = 10.0 ** loglam
        lam_rest = lam / (1.0 + z_qso)
        ix_dla_range = np.logical_and(lam_rest >= REST_RANGE[0], lam_rest <= REST_RANGE[1])
        y_plot_range = np.mean(flux[np.logical_not(np.isnan(flux))]) + 10
        plt.plot(10.0**loglam, flux, 'k-', drawstyle='steps')
        for ii in range(len(sl.dlas)):
            plt.axvline(sl.dlas[ii]['spectrum'], ymin=0.0, ymax=y_plot_range, color='r', linestyle='-')
        for ii in range(len(sl.subdlas)):
            plt.axvline(sl.subdlas[ii]['spectrum'], ymin=0.0, ymax=y_plot_range, color='b', linestyle='-')
        plt.show()
    if print_summary:
        print("--------------------------------")
        print("Summary")
        print("-------")
        if len(sl.dlas) == 0:
            print("No DLAs")
        else:
            for ii in range(len(sl.dlas)):
                print("DLA #{0:d}".format(ii+1))
                print_stats(sl.dlas[ii])
                print("-------")
        print("--------------------------------")
        if len(sl.subdlas) == 0:
            print("No sub DLAs")
        else:
            for ii in range(len(sl.subdlas)):
                print("sub DLA #{0:d}".format(ii+1))
                print_stats(sl.subdlas[ii])
                print("-------")
    print("--------------------------------")
    return sl

if __name__ == '__main__':
    #09:03:33.55 +26:28:36.3
    #fname = "spec-5778-56328-0546.fits"
    fname = "spec-4784-55677-0264.fits"
    fil = fits.open(fname)
    flux, loglam = fil[1].data['flux'], fil[1].data['loglam']
    z_qso = 3.219
    find_dlas(flux, loglam, z_qso, plot=True, print_summary=True)
