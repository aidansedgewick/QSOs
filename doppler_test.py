import numpy as np
import matplotlib.pyplot as plt
from linetools.spectra.xspectrum1d import XSpectrum1D as xspec

import QSOtools as QT




cold = 20.3     # H I Column density of absorber
zabs = 0.0     # Redshift of absorber
bval = 10.0     # Doppler parameter (km/s)
Lya = 1215.6701   # Rest frame Lya wavelength
fval = 0.4164       # Oscillator Strength
gamma = 6.265E8     # Einstein A coefficient


wv_vals = np.arange(1200,1250,0.01)
b_vals = np.arange(10,200,10)


fig,ax = plt.subplots()
for i,b in enumerate(b_vals):
    parameters = [cold, zabs, b, Lya, fval, gamma]
    
    absorb = QT.voigt(wv_vals,parameters)
    ax.plot(wv_vals,absorb)

plt.show()



