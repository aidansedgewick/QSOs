import numpy as np
import astropy.io.fits as fits
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter1d, maximum_filter, minimum_filter
from scipy.signal import savgol_filter

filename = 'spec-4784-55677-0264.fits'
fil = fits.open(filename)
flux = fil[1].data['flux']
wave = 10.0**fil[1].data['loglam']
ivar = fil[1].data['ivar']
flue = 1.0/np.sqrt(ivar)
mask = np.zeros(wave.size)
zem = 2.8867#2.85636

#wave, flux = np.loadtxt("test_spec.dat", unpack=True)
#zem = 3.6384499073028564

idxnum = 4
#ww = np.where(wave < 1215.6701 * (1.0 + zem))
wsky = np.where(((wave > 5570) & (wave < 5580)) |
                ((wave > 6295) & (wave < 6305)) |
                ((wave > 6360) & (wave < 6370)))

wfl = np.round(idxnum * flux / np.median(flux))

# Perform a min/max filter
minflux = minimum_filter(flux, size=20)
maxflux = maximum_filter(flux, size=20)/(0.2+wfl/idxnum)
frac = 0.3
meanval = frac * minflux + (1 - frac) * maxflux
# meanval = savgol_filter(flux, 51, 3)
#fluxcut = maximum_filter(flux, size=10)/(0.5+0.5*wfl/idxnum)
wfl[np.where((flux < meanval) & (wave < 1215.6701 * (1.0 + zem)))] = 0  # Mask low flux forest
wfl[wsky] = 0  # Mask the sky lines
wfl = np.clip(wfl, 0.0, 2*idxnum).astype(np.int)
idxarr = []
for ii in range(flux.size):
    idxarr += wfl[ii] * [ii]

wvsm = wave[idxarr]
fxsm = flux[idxarr]
asrt = np.argsort(wvsm)

cont = gaussian_filter1d(fxsm[asrt], 60)
_, idn, _ = np.intersect1d(np.array(idxarr), np.arange(flux.size), return_indices=True)
f = interpolate.interp1d(wvsm[asrt][idn], cont[idn], kind='linear', bounds_error=False)
cont = f(wave)
#cont = gaussian_filter1d(cont, 20)

plt.subplot(211)
plt.plot(wave, flux, 'k-', drawstyle='steps')
#plt.plot(wave, fluxcut, 'g-', drawstyle='steps')
plt.plot(wave, cont, 'r-')
plt.subplot(212)
plt.plot(wave, flux/cont, 'k-', drawstyle='steps')
plt.show()
assert(False)

np.gradient(flux, edge_order=2)

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