from specdb.specdb import IgmSpec
from pyigm.surveys.dlasurvey import DLASurvey
import dla_cnn.training_set as tset
from matplotlib import pyplot as plt
import astropy.units as u
import numpy as np
import pdb
from QSOtools import get_continuum, get_continuum_alt, get_continuum_alt2


def generate_fakespec(nspec, seed=1234):
    """ Generate nspec fake spectra.
    The seed needs to be changed if you want a different
    set of spectra. In other words, use the same seed to
    get reproducible results."""

    # Sightlines
    sdss = DLASurvey.load_SDSS_DR5(sample='all')
    slines, sdict = tset.grab_sightlines(sdss, flg_bal=0)

    # Run
    final_spec, full_dict = tset.make_set(nspec, slines, outroot=None, seed=seed, slls=False)
    return final_spec, full_dict


def plot_spectra(all_spec, full_dict):
    """ A simple plot of a few spectra to see the DLA."""
    igmsp = IgmSpec()
    sdss = DLASurvey.load_SDSS_DR5(sample='all')
    slines, sdict = tset.grab_sightlines(sdss, flg_bal=0)
    for qq in range(all_spec.nspec):
        plt.subplot(all_spec.nspec, 1, qq+1)
        isl = full_dict[qq]['sl']
        specl, meta = igmsp.spectra_from_coord((slines['RA'][isl], slines['DEC'][isl]),
                                               groups=['SDSS_DR7'], tol=1.0*u.arcsec, verbose=False)
        plt.plot(specl.wavelength, specl.flux, 'r-', drawstyle='steps')
        plt.plot(all_spec[qq].wavelength, all_spec[qq].flux, 'k-', drawstyle='steps')
        plt.xlim(all_spec.wvmin.value, all_spec.wvmax.value)
        plt.ylim(0.0, np.max(specl.flux).value)
    plt.show()


def plot_continuum_fits(all_spec, full_dict):
    """ A simple plot of a few spectra to see the DLA."""
    sdss = DLASurvey.load_SDSS_DR5(sample='all')
    slines, sdict = tset.grab_sightlines(sdss, flg_bal=0)
    for qq in range(all_spec.nspec):
        #plt.subplot(all_spec.nspec, 1, qq+1)
        isl = full_dict[qq]['sl']
        zem = slines['ZEM'][isl]
        print("zem = ", zem)
        cont = get_continuum_alt2(zem, all_spec[qq].wavelength.value, all_spec[qq].flux.value)
        plt.subplot(211)
        plt.plot(all_spec[qq].wavelength, all_spec[qq].flux, 'k-', drawstyle='steps')
        plt.plot(all_spec[qq].wavelength, cont, 'r-', drawstyle='steps')
        plt.xlim(all_spec.wvmin.value, all_spec.wvmax.value)
        plt.subplot(212)
        plt.plot(all_spec[qq].wavelength, all_spec[qq].flux/cont, 'k-', drawstyle='steps')
        plt.xlim(all_spec.wvmin.value, all_spec.wvmax.value)
        plt.show()
        #pdb.set_trace()


if __name__ == "__main__":
    all_spec, full_dict = generate_fakespec(20)
    #plot_spectra(all_spec, full_dict)
    plot_continuum_fits(all_spec, full_dict)
