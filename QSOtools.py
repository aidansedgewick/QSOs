import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.integrate as integrate
#from scipy.interpolate import interp1d
from scipy import interpolate
from scipy.special import wofz
import astropy.io.fits as fits
import astropy.units as u

from linetools.spectra.xspectrum1d import XSpectrum1D as xspec
from scipy.ndimage.filters import gaussian_filter1d

import warnings
import pickle
import time
import sys
import os


workdir = './' #'/home/aidan/PostGrad/QSOs2/' ## **DO NOT** use "~" for $HOME, pyfits doesn't like it!!

###--- SOME HANDY REFERENCES ---###
#1 PROCHASKA ApJ 675:1002  - gives details of SDSS spec limits
#2 PROCHASKA14 MNRAS 438:476 - the points for the ColDensity dist. func. spline
#3 RUDIE13     ApJ 769:146   - the formula for calculating m, DX.
#4 HARRIS16    AJ 151:155    - the composite QSO spectrum   



###------ SOME CONSTANTS -------###

c = 3.0E5  # km/s
pi = np.pi



b_min, b_max = 10.0,20.0 # Doppler parameters.
cold = 20.3     # H I Column density of absorber
zabs = 1.88     # Redshift of absorber
bval = 10.0     # Doppler parameter (km/s)
Lya = 1215.6701   # Rest frame Lya wavelength
fval = 0.4164       # Oscillator Strength
gamma = 6.265E8     # Einstein A coefficient

restframe_window = 3.5 #Angstrom.

SDSS_min, SDSS_max = 3800.0, 9200.0 # Ang., limits of SDSS spectrometer.
SDSS_wv_resolution = 2.0
SDSS_wave = np.arange(SDSS_min, SDSS_max, SDSS_wv_resolution)
SDSS_smear = [2.0]


SDSS_min, SDSS_max = 3800.0, 9200.0 # Ang., limits of SDSS spectrometer.
SDSS_resolution = 2.0
SDSS_wave = np.arange(SDSS_min, SDSS_max, SDSS_wv_resolution)

SDSS_z_min = SDSS_min/Lya-1

min_col_density = 17.0


###------------------ LOAD THE COMPOSITE --------------------###

## Get the composite.
composite_loc = workdir + 'datfiles/harris15_composite.fits'
composite = fits.open(composite_loc)[1].data

# Get rid of the zeros...
rf_composite_wv = composite['WAVE'][ composite['FLUX'] > 0.0 ]
rf_composite_fl = composite['FLUX'][ composite['FLUX'] > 0.0 ]

# REBIN the composite to v. high res, so that we can stretch and absorb it...
high_wave_resolution = 0.05
high_resolution_wv = np.arange(rf_composite_wv.min(), rf_composite_wv.max(), high_wave_resolution)

# Create xspec obj. for rebinning into high res.
composite_xspec = xspec.from_tuple( (rf_composite_wv*u.Angstrom,rf_composite_fl) )
HR_composite_xspec = composite_xspec.rebin(high_resolution_wv*u.Angstrom)
high_resolution_fl = HR_composite_xspec.flux.value

###------------ LOAD SOME COLUMN DENSITY INFO ---------------###

## Get a function for the column density distribution.
prochaska_loc = workdir + 'datfiles/prochaska_data.txt'
nHI_data = pd.read_csv(prochaska_loc,delim_whitespace=True) # Column density points, to fit spline to.
# print(nHI_data.columns)

linear_fit = interpolate.interp1d(nHI_data['logN'], nHI_data['logf']) # A linear spline
cubic_fit  = interpolate.interp1d(nHI_data['logN'], nHI_data['logf'], kind='cubic') # A cubic spline



N_points = 200 # Evaluate DN on a LOG GRID, equal in log space.
nHI_vals = np.logspace(nHI_data['logN'].min(), nHI_data['logN'].max(), N_points)
nHI_intervals   = nHI_vals[1:]-nHI_vals[:-1]

nHI_mids = 0.5*(nHI_vals[:-1] + nHI_vals[1:])
nHI_evaluations = 10**cubic_fit( np.log10(nHI_mids) )

#plt.plot(nHI_mids,nHI_evaluations)
#plt.semilogy()
#plt.show()

###--------------- LOAD THE SDSS QSOs DATA -----------------###

SDSS_QSOs_loc = workdir + 'datfiles/AllQSOspec_zgt2p1.csv'
SDSS_data = pd.read_csv(SDSS_QSOs_loc)
# print(SDSS_data.columns)


###------- THE FUNCTIONS ----------###
# in order --
# write: carriage-return print
# midpoints: calc midpoints of an array
# calc_DX: the absorption pathlength
# voigt: the absorption profile
# call_CPU: the instrument convolution
# evaluate_Q: the log-likelihood of each score
#


def write(*args):
    '''Effectively 'print' with carriage return '''
    string = '> '+ ' '.join('%s' %arg for arg in args)
    sys.stdout.write('\r')     # Return carriage BEFORE the string
    sys.stdout.flush()         # Does this fix 'blinking'?
    sys.stdout.write(string)
    sys.stdout.flush()
    return None

def midpoints(arr):
    ''' Calculate the midpoints of an array '''
    return 0.5*(arr[1:] + arr[:-1])


def calc_DX(z_max, z_min, z_points=1000, om_l=0.7, om_m=0.3):
    '''RUDIE13  ApJ 769:146   - the formula for calculating m, DX.
        This is the 'pathlength?' '''
    z_vals = np.linspace(z_max, z_min, z_points)
    zp1 = z_vals + 1.0
    y_vals = zp1*zp1/np.sqrt(om_l + om_m*zp1**3)
    delta_X = integrate.trapz(y_vals, x=z_vals)
    return delta_X



def voigt(wave, par):
    '''RJC. get a voigt profile to do some absorption.'''
    # N_HI   z_DLA   Dop_val   Lya   Osc_stren   A_21(Einstein)
    #   0      1        2        3        4               5
    #ck = 2.99792458E5
    #J = 3.76730313461770655E11 # WHAT ARE THESE...?
    #K = 2.002134602291006E12   # VALUES...?

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


def call_CPU(x, y, p):
    """
    From RJC's ALIS. --Is the name 'call_CPU' relevant?
    *think* is used for instrument convolution...
    Modified from original, extra "int()" around "df" calculation...
    

    Define the functional form of the model
    --------------------------------------------------------
    x  : array of wavelengths
    y  : model flux array
    p  : array of parameters for this model
    --------------------------------------------------------
    """
    sigd = p[0] / ( 2.0*np.sqrt(2.0*np.log(2.0)) )
    if np.size(sigd) == 1: cond = sigd > 0
    else: cond = np.size(np.where(sigd > 0.0)) >= 1
    if cond:
        ysize=y.size
        fsigd=6.0*sigd
        dwav = 0.5*(x[2:]-x[:-2])
        dwav = np.append(np.append(dwav[0],dwav), dwav[-1])
        if np.size(sigd) == 1:
            df=int(np.min([np.int(np.ceil(fsigd/dwav).max()), ysize/2 - 1]))
                ### MODIFIED FROM ORIGINAL!! extra 'int'
            yval = np.zeros(2*df+1)
            yval[df:2*df+1] = (x[df:2*df+1] - x[df])/sigd
            yval[:df] = (x[:df] - x[df])/sigd
            gaus = np.exp(-0.5*yval*yval)
            size = ysize + gaus.size - 1
            fsize = 2 ** np.int(np.ceil(np.log2(size))) # Use this size for a more efficient computation
            conv = np.fft.fft(y, fsize)
            conv *= np.fft.fft(gaus/gaus.sum(), fsize)
            ret = np.fft.ifft(conv).real.copy()
            del conv
            return ret[df:df+ysize]
        elif np.size(sigd) == szflx:
            yb = y.copy()
            df=np.min([np.int(np.ceil(fsigd/dwav).max()), ysize/2 - 1])
                ### MODIFIED FROM ORIGINAL!!! extra 'int'
            for i in range(szflx):
                if sigd[i] == 0.0:
                    yb[i] = y[i]
                    continue
                yval = np.zeros(2*df+1)
                yval[df:2*df+1] = (x[df:2*df+1] - x[df])/sigd[i]
                yval[:df] = (x[:df] - x[df])/sigd[i]
                gaus = np.exp(-0.5*yval*yval)
                size = ysize + gaus.size - 1
                fsize = 2 ** np.int(np.ceil(np.log2(size))) # Use this size for a more efficient computation
                conv  = np.fft.fft(y, fsize)
                conv *= np.fft.fft(gaus/gaus.sum(), fsize)
                ret   = np.fft.ifft(conv).real.copy()
                yb[i] = ret[df:df+ysize][i]
            del conv
            return yb
        else:
            msgs.error("Afwhm and flux arrays have different sizes.")
    else: return y


def generate_spectrum(z_QSO,sig1=0.05,sig2=0.05,wv_min=SDSS_min,wv_max=SDSS_max,wv_res=SDSS_resolution,
                        v_min=-9000.0,v_max=+3000.0,min_col_density=min_col_density,MW=400.0):
    '''Generate a fake spectrum. Requires z_QSO (Lya emission redshift). 
    Must return the observed wavelength, flux, error, and the absorbers' column densities and redshifts.
    
    Optional: sig1 & sig2 (noise paramters), wv_min & wv_max (limits of returned wv array),
    wv_res (spectral res of returned wv arr), v_min & v_max (min/max absorber recession vel from QSO in km/s), 
    min_col_density (the lowest col density to return), MW (the maximum expected width of any single absorber).
    All wv params in Angstroms.
    '''

    ##------- INITIALISE SOME STUFF.

    #v_min = -9000.0 # kms^-1, min recession from QSO.
    #v_max = 30.0 #3000.0  # kms^-1, max recession from QSO.

    observed_wv = np.arange(wv_min,wv_max,wv_res)

    composite_wv = HR_composite_xspec.wavelength.value.copy()*(1.0+z_QSO)
    composite_fl = HR_composite_xspec.flux.value.copy()
    
    # Only really care about absorption bluewards of Lya peak...? but NOT bluewards of minimum.
    absorber_z_min = max(z_QSO + v_min/c*(1.0+z_QSO), wv_min/Lya-1.0 ) 
    absorber_z_max = max(z_QSO + v_max/c*(1.0+z_QSO), wv_min/Lya-1.0 + 0.01) 

    absorber_wv_min = (1.0+absorber_z_min)*Lya
    absorber_wv_max = (1.0+absorber_z_max)*Lya

    DX = calc_DX(absorber_z_min, absorber_z_max)
    m_vals = nHI_evaluations * nHI_intervals * DX

    # Lists to store the absorbers.
    absorber_CDs = []
    absorber_zs  = []
    #absorber_bs  = []


    ##------ DO SOME ABSORPTION.

    for j,m in enumerate(m_vals): # Get 'm', and an index 'j'
        prob = m % 1 # The 'decimal' of 'm' == m - int(m)
 		#0<rand<1 to compare to decimal part of 'm'. eg 4.267: ~75% of time, m=4, else m=5. Think about it...   
        rand_val = np.random.random(1)[0]        
        absorber_count = int(m) + int( (rand_val <= prob) ) # Returns 'm' if RV<=prob TRUE, else 'm+1' 

        col_density = np.log10(nHI_mids[j])
        z_vals = np.random.uniform(absorber_z_min, absorber_z_max, absorber_count) # Redshift values within DX
        b_vals = np.random.uniform(b_min, b_max, absorber_count) # Doppler vals  

        #Store the values if above some interesting minimum.
        if col_density > min_col_density:
            absorber_CDs.extend( [col_density]*absorber_count )
            absorber_zs.extend(z_vals)
            ## absorber_bs.extend(b_vals)   

        # Which "wavelength pixels" do the absorbers correspond to?
        wv_pix   = np.digitize( (1.0+z_vals)*Lya, composite_wv)-1
        min_pix  = np.digitize( (1.0+z_vals)*Lya-MW/2, composite_wv)-1
        max_pix  = np.digitize( (1.0+z_vals)*Lya+MW/2, composite_wv)-1

        for k,(z_k,b_k) in enumerate(zip(z_vals,b_vals)):
            parameters = [col_density, z_k, b_k, Lya, fval, gamma]

            # Be smart about which parts of the spectrum we want to calculate absorption for...
            # Define a slice object to look only at the interesting parts.
            slicer = slice(max(0,min_pix[k]), min(max_pix[k],len(composite_wv)-1))

            absorption = voigt(composite_wv[slicer], parameters) # 
            composite_fl[slicer] = composite_fl[slicer]*absorption



    ##------ MAKE IT LOOK LIKE A REAL SPECTRUM.

    # Convolve with an instrument profile.
    convolved_fl = call_CPU(composite_wv,composite_fl,[wv_res])
    
    # Create a xspec obj. to rebin onto SDSS wavelengths.
    absorbed_composite_xspec = xspec.from_tuple( (composite_wv*u.Angstrom,convolved_fl) )
    convolved_xspec = absorbed_composite_xspec.rebin(observed_wv*u.Angstrom)

    convolved_fl = convolved_xspec.flux.copy()

    sigma = sig1 + sig2*convolved_fl
    observed_flux = np.random.normal(convolved_fl,sigma)

    absorber_CDs = np.array(absorber_CDs) # > 
    absorber_zs  = np.array(absorber_zs)  # > Arrays are nicer to deal with...
    #absorber_bs  = np.array(absorber_bs)  # >

    # REVERSE sort the absorbers by Column Density.
    sorted_absorber_indices = np.argsort(absorber_CDs)    
    absorber_CDs = absorber_CDs[ sorted_absorber_indices[::-1] ]
    absorber_zs  = absorber_zs[  sorted_absorber_indices[::-1] ]
    ## absorber_bs  = absorber_bs[  sorted_absorber_indices[::-1] ]

    return observed_wv,observed_flux,sigma,absorber_CDs,absorber_zs#,absorber_bs


def get_continuum(zem,wave,flux,flue,hspc=None,kind='smooth'):
    ''' From RJC envolopy.py
    
    Give z_em, wavelength, flux, flux_error.
    also, 'kind' as one of smooth, linear, max (default=smooth)
    '''
    mask = np.zeros(wave.size)
    ww = np.where(wave < 100.0*1215.6701*(1.0+zem))
    wave = wave[ww]
    flux = flux[ww]
    flue = flue[ww]
    #ivar = ivar[ww]
    mask = mask[ww]


    if hspc is None:
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

    asrt = np.argsort(xarr)
    f = interpolate.interp1d(xarr[asrt], yarr[asrt], kind='linear')
    cont = f(wave)
    contsm = gaussian_filter1d(cont, 2*hspc)

    if kind == 'smooth':
        return contsm
    elif kind == 'linear':
        return cont
    elif kind == 'max':
        return np.maximum(cont,contsm)
    else:
        warn_msg = '''\033[33m \n Choose 'kind' from: smooth, linear, max. Using default: smooth.\033[0m'''
        warnings.warn(warn_msg)
        


def evaluate_Q(fl,sig):
    '''See notes.
    Use **NORMALIZED** flux. Derive with: likelhood of Npix draw from zero (f=0),
    divided by Npix. likelihood of draw from continuum (ie,f=1), take logs.'''
    return (1-2*fl)/(2*sig*sig)



def evaluate_scores(wv,fl,sig,Npix):
    '''See notes.
    Use **NORMALIZED** flux. Derive with: likelhood of Npix draw from zero (f=0),
    divided by Npix. likelihood of draw from continuum (ie,f=1), take logs.

    what is the fastest/best to do this...? Surely a less ugly way than loops. Unsure.'''

    Npix = int(Npix)

    Q = evaluate_Q(fl,sig)    

    score_wave,score_vals = np.zeros(len(Q)),np.zeros(len(Q))
    ## Surely there is a less ugly way than a for loop...?!
    for i in range(len(Q)):
        sl_min,sl_max = max(0,i-Npix//2),min(len(Q),i+Npix//2)
        slicer = slice( sl_min,sl_max )
        score_vals[i] = np.sum(Q[slicer])/float(len(Q[slicer]))
        score_wave[i] = wv[i]
    
    return score_wave,score_vals



def consecutive(data, stepsize=1):
    '''see: stackoverflow.com/questions/7352684, Answer 2.
       >>> a = np.array([0, 47, 48, 49, 50, 97, 98, 99])
       >>> consecutive(a)
       >>> [array([0]), array([47, 48, 49, 50]), array([97, 98, 99])] '''

    if len(data) == 0:
        return np.array([])
    else:
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def extract_features(z_QSO,scores_wv,score_vals,threshold,min_width=3.5,scaled=True):
    '''Find consecutive pixels with scores above the threshold.
        Returns their widths, estimated redshifts. '''

    (inds,) = np.where(score_vals > threshold) # np.where gives a len-1 tuple. unpack it to split...  
    feature_masks = consecutive( inds ) # List of arrays. Each arr is **indices** of consec pix above the thresh.
    
    feature_masks = [fm for fm in feature_masks if len(fm) >= min_width ]

    score_zs = scores_wv/Lya - 1.0
    
    # SCALE the widths to redshift zero...
    feature_widths = np.array([ len(fm) for fm in feature_masks ])
    feature_zs = np.array([ np.average(score_zs[fm]) for fm in feature_masks ])
    #feature_bs = np.array([ 15.0 for fm in feature_masks ])

    if scaled:
        feature_widths = feature_widths/(1.0+z_QSO)

    sorted_feature_indices = np.argsort(feature_widths)
    feature_widths = feature_widths[ sorted_feature_indices[::-1] ]
    feature_zs = feature_zs[ sorted_feature_indices[::-1] ]
    #feature_bs = feature_bs[ sorted_feature_indices[::-1] ]

    return feature_widths,feature_zs #,feature_bs


def match_features(z_QSO,absorber_CDs,absorber_zs, #absorber_bs,
                    feature_widths,feature_zs,#feature_bs,
                    blank_value=-1.0):
    
    Ntuples = max( len(feature_widths),len(absorber_CDs) )

    this_QSO = []

    for k in range(Ntuples):

        if k >= len(absorber_CDs): # If we have more features than DLAs to associate...
            this_feature = (z_QSO,   blank_value   ,  blank_value , # blank_value  ,
                                    feature_widths[k], feature_zs[k]) #, feature_bs[k])

        elif k >= len(feature_widths): # If there are undetected DLAs...
            this_feature = (z_QSO,absorber_CDs[k], absorber_zs[k], #absorber_bs[k],
                                       blank_value ,  blank_value  ) #,  blank_value  )
        
        else: # The 'normal' case where a feature matches a DLA.                
            this_feature = (z_QSO,absorber_CDs[k],absorber_zs[k], #absorber_bs[k],
                                  feature_widths[k], feature_zs[k]) #, feature_bs[k])

        this_QSO.append(this_feature) 

    return this_QSO


