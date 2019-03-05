## QSOs
some scripts

in QSOtools.py, can change "workdir" (approx line 18) to your work dir - else is './'

QSOtools.py contains all functions.
QSO.py uses these to generate fake spectra.

./datfiles/ contains: 
- points for nHI column density distribution to fit spline 
- composite QSO spectrum (Harris et al. 16) 
- A list of all available SDSS spectra with z > 2.1.

envolope.py - from RJC, estimates continuum - adapted into "get_continuum" in QSOtools, acts on spec-...-0264.fits.

