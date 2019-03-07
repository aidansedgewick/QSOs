import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.special import wofz
import astropy.io.fits as fits
import astropy.units as u

from linetools.spectra.xspectrum1d import XSpectrum1D as xspec
import QSOtools as QT

import pickle
import time
import sys
import os

workdir = QT.workdir 
pkl_file_loc = workdir + 'data.pkl' 

### SOME HANDY REFERENCES ###
#1 PROCHASKA ApJ 675:1002  - gives details of SDSS spec limits
#2 PROCHASKA14 MNRAS 438:476 - the points for the ColDensity dist. func. spline
#3 RUDIE13     ApJ 769:146   - the formula for calculating m, DX.
#4 HARRIS16    AJ 151:155    - the composite QSO spectrum   



### CONSTANTS AND PARAMETERS ###

c = 3.0E5  # km/s
pi = np.pi

v_min = -25000.0 # kms^-1, min recession from QSO.
v_max = 3000.0 #3000.0  # kms^-1, max recession from QSO.

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

SDSS_z_min = SDSS_min/Lya-1.0



###------- LOAD IN SOME DATA --------###

SDSS_QSOs_loc = workdir + 'datfiles/AllQSOspec_zgt2p1.csv'
SDSS_data = pd.read_csv(SDSS_QSOs_loc)
# print(SDSS_data.columns)



#### ------- MAKE THE SPECTRA! ------- ###

DLA_def = 20.3
min_col_density = 15.0
min_feature_width = 5
# No *single* absorber is likely to be wider than 400 Ang...!?
MW = 400.0 #max_width
num_QSOs = 10000

threshold_vals = np.arange(-10,200,5) # A list of threshold values

# Initialize the horrible data strucure:
data_structure = [ [] for i in threshold_vals ]

# If we've done it before...
pickle_exists = os.path.exists(pkl_file_loc)


t_start = time.time()

for i in range(num_QSOs):
    if pickle_exists:
        break
        

    random_index = np.random.randint(0,len(SDSS_data),1)
    z_QSO = SDSS_data['z'].iloc[random_index].values[0] # [0] selects the 1st (and only!) value in the array.
    wv_em = (1.0+z_QSO)*Lya

    
    ## ------GENERATE A SPECTRUM
    spec_wv,spec_fl,spec_sig,absorber_CDs,absorber_zs = QT.generate_spectrum(z_QSO,v_min=v_min,v_max=v_max)

    prox_wv_min = wv_em*(1.0+v_min/c)
    prox_wv_max = wv_em*(1.0+v_max/c)

    # Keep only the interesting bits?
    proximate_mask = (prox_wv_min < spec_wv) & (spec_wv < prox_wv_max+100)

    prox_wv  = spec_wv[ proximate_mask ]
    prox_fl  = spec_fl[ proximate_mask ]
    prox_sig = spec_sig[ proximate_mask ]


    ## ------NORMALISE IT
    prox_cont = QT.get_continuum(z_QSO,prox_wv,prox_fl,prox_sig,hspc=10,kind='max')
    norm_fl  = prox_fl/prox_cont
    norm_sig = prox_sig/prox_cont


    ## ------DO THE SCORES
    z_window = (1.0+z_QSO)*restframe_window
    Npix = int(z_window/SDSS_wv_resolution)

    score_wv,score_vals = QT.evaluate_scores(prox_wv,norm_fl,norm_sig,Npix)
    score_zs = score_wv/Lya - 1.0

    ## ------EVALUATE FOR VARYING 
    for j,th in enumerate(threshold_vals):
        # Look at the width of the 
        feature_widths,feature_zs = QT.extract_features(z_QSO,score_wv,score_vals,threshold=th)
        
        # Match the widest feature to the largest column density 
        QSO_matches = QT.match_features(z_QSO,absorber_CDs,absorber_zs, feature_widths,feature_zs)
        data_structure[j].extend(QSO_matches)

    QT.write(i+1,num_QSOs)
    
    '''
    ### Some plotting stuff...
    if len(absorber_CDs[ absorber_CDs > 20.3 ]) > 0: # np.max(abs_CDs) > 20.3 doesn't work for len=0.
        fig,gs=plt.figure(),plt.GridSpec(4,3)
        ax1,ax2,ax3 = plt.subplot(gs[:2,:]),plt.subplot(gs[2:3,:]),plt.subplot(gs[3:,:])

        ax1.plot(prox_wv,prox_fl,'k',drawstyle='steps')
        ax1.plot(prox_wv,prox_cont,'k--')

        ax2.plot(prox_wv,norm_fl,color='k',drawstyle='steps')
        ax2.plot(prox_wv,norm_fl/norm_fl,'k--')

        ax3.plot(score_wv,score_vals)
        
        for q,ax in enumerate([ax1,ax2,ax3]):
            ax.set_xlim(prox_wv_min,prox_wv_max)
            if q!=2: ax.set_xticks([])
            for ab_z,ab_CD in zip(absorber_zs[ absorber_CDs > 20.3 ], absorber_CDs[ absorber_CDs > 20.3 ]):
                ax.axvline((ab_z+1.0)*Lya,color='g',ls='--')
                if q==0: ax.set_ylim(-1.5,12),ax.text((ab_z+1.0)*Lya+5.0,11,'%.1f'%ab_CD,horizontalalignment='left',verticalalignment='center')
                
        fig.subplots_adjust(hspace=0)
        plt.show()
        '''



if pickle_exists is False:
    t_end = time.time()
    print('\n ~%i per second.' %round(num_QSOs/(t_end-t_start),0) )
    with open(pkl_file_loc, 'wb') as f:
        pickle.dump(data_structure,f)
else:
    with open(pkl_file_loc, 'rb') as f:
        data_structure = pickle.load(f)
    

cols = 'z_QSO absorber_CDs absorber_zs feature_widths feature_zs'.split()

#completeness  = np.zeros(len(threshold_vals))
#contamination = np.zeros(len(threshold_vals))

min_widths = np.arange(1.5,6.5,0.5)

N_pos = np.zeros( (len(min_widths),len(threshold_vals)) )
N_neg = np.zeros( (len(min_widths),len(threshold_vals)) )
N_det = np.zeros( (len(min_widths),len(threshold_vals)) )


for j,th in enumerate(threshold_vals):
    df = pd.DataFrame(data_structure[j],columns=cols)

    true_DLA_mask = (df['absorber_CDs'] > DLA_def)

    N_true = len( df[ true_DLA_mask ] )

    for k,w in enumerate(min_widths):
        # For detection above this minumum feature width, 
        # how many true positives, false positives, total positives?

        detected_mask = (df['feature_widths'] > w )#*(1.0+df['z_QSO'])
        

        N_pos[k,j] = len( df[  true_DLA_mask & detected_mask ]) 
        N_neg[k,j] = len( df[ ~true_DLA_mask & detected_mask ])
        N_det[k,j] = len( df[ detected_mask ] ) # = N_pos+N_neg??


# Interested in...
recall  = N_pos/N_true
contamination = N_neg/(N_neg + N_pos)

# An array to save the values of area under the ROC curve.
AUC = np.zeros(len(min_widths))

# Make some axes to do the plots...
fig1 = plt.figure(figsize=(8,5))
ax1a = fig1.add_subplot(211)
ax1b = fig1.add_subplot(212)

fig2,ax2 = plt.subplots(figsize=(8,5))


for k,w in enumerate(min_widths):
    #plt.plot(threshold_vals,completeness[k,:],color='C%i'%k)
    
    # Recall & contam for min_width[k], the slice is for 'neatness'
    Rk = recall[k,:] 
    Ck = contamination[k,:]

    ax1a.plot(threshold_vals,Rk*(1.0-Ck),color='C%i'%k,label=w)
    ax1b.plot(threshold_vals,1.0-Ck,color='C%i'%k)
    ax1b.plot(threshold_vals,Rk,ls='--',color='C%i'%k)
    
    if k==0: # Only want one solid & one dash line in the legend.
        ax1b.plot((0,0),(0,0),'k--',label='Contamination')
        ax1b.plot((0,0),(0,0),'k',label='Recall')
    ax2.plot(Ck,Rk,color='C%i'%k)
    
    Ck_mask = np.isfinite(Ck) # Can't do the AUC integration --
    Rk_mask = np.isfinite(Rk) # -- if we have NaN values!

    try:
        AUC[k] = integrate.trapz(Rk[ Ck_mask & Rk_mask ][::-1], x=Ck[ Ck_mask & Rk_mask ][::-1])
    except:
        print("%.1f: can't integrate")
    
fig1.subplots_adjust(hspace=0)
ax1a.legend(ncol=2)
ax1b.legend()
ax1a.set_ylabel(r'$R\times\left(1-C\right)$',fontsize=18)
ax1b.set_ylabel(r'$R,C$',fontsize=18)
ax1b.set_xlabel(r'Threshold score',fontsize=18)

ax2.set_xlabel(r'Contamination $C$',fontsize=18)
ax2.set_ylabel(r'Recall $R$',fontsize=18)

ax2_sub = fig2.add_axes([0.45,0.18,0.43,0.4])#,transform=ax2.transAxes)
ax2_sub.plot(min_widths,AUC,'k')
ax2_sub.set_ylabel(r'AUC',fontsize=18)


# Choose a few example thresholds to look at...
th_vals = [10.0,20.0,50.0]
th_indices = np.digitize(th_vals, threshold_vals)-1




### Some more plots...

width_bins = np.arange(0,15,0.2)
width_mids = QT.midpoints(width_bins)

nHI_bins = np.log10(QT.nHI_vals)
nHI_mids = QT.midpoints(nHI_bins)

fig_scatter,ax_scatter = plt.subplots()
ax_scatter.set_xlabel(r'$\log_{10}\left(n_{\mathrm{HI}}\right)$ Column density')
ax_scatter.set_ylabel('Feature width [pix]')

fig_pcolor,ax_pcolor = plt.subplots()
ax_pcolor.set_xlabel(r'$\log_{10}\left(n_{\mathrm{HI}}\right)$ Column density')
ax_pcolor.set_ylabel(r'$\frac{\mathrm{Feature\;width\;[pix]}}{1+z_{QSO}}$',fontsize=20)

# Contour plot is a failed experiment.
fig_contour,ax_contour = plt.subplots()
ax_contour.set_xlabel(r'$\log_{10}\left(n_{\mathrm{HI}}\right)$ Column density')
ax_contour.set_ylabel(r'$\frac{\mathrm{Feature\;width\;[pix]}}{1+z_{QSO}}$',fontsize=20)

fig_hist,ax_hist = plt.subplots()
ax_hist.set_xlabel(r'$z_{est}-z_{abs}$',fontsize=20)

fig_zerr = plt.figure()
gs = plt.GridSpec(3,3)
ax_zerr1 = fig_zerr.add_subplot(gs[:2,:])
ax_zerr2 = fig_zerr.add_subplot(gs[2:,:])
ax_zerr2.set_xlabel(r'$z_{QSO}-z_{abs}$',fontsize=20)
ax_zerr1.set_ylabel(r'$z_{est}-z_{abs}$',fontsize=20)

for i,th in enumerate(th_vals):
    df = pd.DataFrame(data_structure[th_indices[i]],columns=cols)

    true_absorber_mask = (df['absorber_CDs'] > 0 )
    detected_mask = (df['feature_widths'] > 3 )

    # Look only at the true stuff that we've detected...
    df = df[ true_absorber_mask & detected_mask ]

    ax_scatter.scatter(df['absorber_CDs'],df['feature_widths'],color='C%i'%i,s=2)


    hist2d,_,_ = np.histogram2d(df['absorber_CDs'],df['feature_widths'] ,bins=[nHI_bins,width_bins])
    hist1d,_ = np.histogram(df['absorber_CDs'], bins=nHI_bins)
    hist2d = hist2d/hist1d[:,None]
    hist2d[ hist2d == 0 ] = np.nan
    if i ==0:
        cp = ax_pcolor.pcolormesh(nHI_bins,width_bins,hist2d.T,vmin=0)
        plt.colorbar(cp,ax=ax_pcolor)

    xx,yy = np.meshgrid(nHI_mids,width_mids)
    #ax_contour.contour(xx,yy,hist2d.T,levels=[0.01,0.05,0.1],color='C%i'%i)


    ax_hist.hist(df['absorber_zs']-df['feature_zs'],bins=np.linspace(-0.025,0.025,50),color='C%i'%i,histtype='step',density=True)

    ax_zerr1.scatter(df['absorber_zs']-df['z_QSO'],df['feature_zs']-df['absorber_zs'],s=2)
    

plt.show()

























