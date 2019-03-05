import numpy as np
from scipy.special import wofz
from matplotlib import pyplot as plt

def voigt(wave, par):
	"""
	Define the model here
	"""
	cold = 10.0**par[0]
	zp1=par[1]+1.0
	wv=par[3]*1.0e-8
	bl=par[2]*wv/2.99792458E5
	a=par[5]*wv*wv/(3.76730313461770655E11*bl)
	cns=wv*wv*par[4]/(bl*2.002134602291006E12)
	cne=cold*cns
	ww=(wave*1.0e-8)/zp1
	v=wv*ww*((1.0/ww)-(1.0/wv))/bl
	tau = cne*wofz(v + 1j * a).real
	return np.exp(-1.0*tau)

# Plot a voigt profile

if __name__ == "__main__":


    wave = np.linspace(3000.0, 4000.0, 1000.0)
    cold = 22.0     # H I Column density of absorber
    zabs = 1.88     # Redshift of absorber
    bval = 10.0     # Doppler parameter (km/s)
    wave0 = 1215.6701   # Rest frame Lya wavelength
    fval = 0.4164       # Oscillator Strength
    gamma = 6.265E8     # Einstein A coefficient
    par = [cold, zabs, bval, wave0, fval, gamma]
    model = voigt(wave, par)

    plt.plot(wave, model, 'k-')
    plt.show()