import numpy as np
import astropy.io.fits as fits
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter1d

filename = 'spec-4784-55677-0264.fits'
fil = fits.open(filename)
flux = fil[1].data['flux']
wave = 10.0**fil[1].data['loglam']
ivar = fil[1].data['ivar']
flue = 1.0/np.sqrt(ivar)
mask = np.zeros(wave.size)
zem = 2.8867#2.85636

ww = np.where(wave < 1215.6701*(1.0+zem))
wave = wave[ww]
flux = flux[ww]
flue = flue[ww]
ivar = ivar[ww]
mask = mask[ww]

hspc = 10
nmax = wave.size//(2*hspc)

# Setupt he arrays and their endpoints
xarr, yarr = np.zeros(nmax), np.zeros(nmax)
xarr[0], xarr[1] = wave[0], wave[-1]
yarr[0], yarr[1] = np.max(flux[:hspc]), np.max(flux[-hspc:])
mask[:hspc] = 1
mask[-hspc:] = 1
# Mask all significantly zero points
mask[np.where(flux<3.0*flue)] = 1
# Now solves for all of the midpoints
for ii in range(2, nmax):
    ww = np.where(mask==0)[0]
    if ww.size == 0:
        xarr = xarr[:ii]
        yarr = yarr[:ii]
        break
    amax = np.argmax(flux[ww])
    xarr[ii] = wave[ww[amax]]
    yarr[ii] = flux[ww[amax]]
    # Set the bounds
    xmn = ww[amax]-hspc
    xmx = ww[amax]+hspc+1
    if xmn < 0: xmn = 0
    if xmx > wave.size: xmx = wave.size
    mask[xmn:xmx] = 1


print(flue/flux)


asrt = np.argsort(xarr)
f = interpolate.interp1d(xarr[asrt], yarr[asrt], kind='linear')
cont = f(wave)
contsm = gaussian_filter1d(cont, 2*hspc)
plt.plot(wave, flux, 'k-', drawstyle='steps')
plt.plot(wave, cont, 'b-')
plt.plot(wave, contsm, 'r-')
plt.show()

plt.plot(wave, flux/contsm, 'k-', drawstyle='steps')
plt.plot(wave, contsm/contsm, 'r-')
plt.show()