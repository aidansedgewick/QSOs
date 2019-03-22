from specdb.specdb import IgmSpec
from pyigm.surveys.dlasurvey import DLASurvey
import dla_cnn.training_set as tset
from matplotlib import pyplot as plt
import astropy.units as u

igmsp = IgmSpec()

# Sightlines
sdss = DLASurvey.load_SDSS_DR5(sample='all')
slines, sdict = tset.grab_sightlines(sdss, flg_bal=0)

# Run
nspec = 5
seed = 1234
final_spec, full_dict = tset.make_set(nspec, slines, outroot=None, seed=seed, slls=False)

for qq in range(nspec):
    plt.subplot(nspec, 1, qq+1)
    isl = full_dict[qq]['sl']
    specl, meta = igmsp.spectra_from_coord((slines['RA'][isl], slines['DEC'][isl]),
                                           groups=['SDSS_DR7'], tol=1.0*u.arcsec, verbose=False)
    plt.plot(specl.wavelength, specl.flux, 'k-', drawstyle='steps')
    plt.plot(final_spec[qq].wavelength, final_spec[qq].flux, 'r-', drawstyle='steps')

plt.show()
