# QSOs
some scripts

in QSOtools.py, can change "workdir" (approx line 18) to your work dir - else is './'

## Contains:

datfiles/ contains: 
- points for nHI column density distribution, to fit spline 
- composite QSO spectrum (Harris et al. 16) 
- A list of all available SDSS spectra with z > 2.1.

linetools/:
- an old version of linetools.
- rebinning (to different resolutions) in new versions of LT incompatible for some reason?

envolope.py: from RJC, estimates continuum - adapted into "get_continuum" in QSOtools, acts on spec-4784-55677-0264.fits.

QSOtools.py: contains all useful functions.

QSO.py: uses these to generate fake spectra, and look at threshold. Needs a better name!

voigt_profile.py: also in QSOtools.py. Could remove?

## Use:

- Clone
- can edit the number of QSOs in QSOs.py, approx line 72. (default 10000, around 10 mins runtime)
- run "python3 QSOs.py"
- saves a python pkl file 'data.pkl', produces some plots.
- if data.pkl exists (ie from previous run), QSOs.py just reads this and makes the plots.



