import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pandas as pd
import astropy.units as u
import time
import os
from scipy.interpolate import interp1d
from linetools.spectra.xspectrum1d import XSpectrum1D as xspec
from scipy.special import wofz
import astropy.io.fits as fits

import pickle

import QSOtools as QT

workdir = QT.workdir 
pkl_file = workdir + 'data.pkl' 

### SOME HANDY REFERENCES ###
#1 PROCHASKA ApJ 675:1002  - gives details of SDSS spec limits
#2 PROCHASKA14 MNRAS 438:476 - the points for the ColDensity dist. func. spline
#3 RUDIE13     ApJ 769:146   - the formula for calculating m, DX.
#4 HARRIS16    AJ 151:155    - the composite QSO spectrum   



### CONSTANTS AND PARAMETERS ###

c = 3.0E5  # km/s
pi = np.pi

v_min = -25000.0 # kms^-1, min recession from QSO.
v_max = 30.0 #3000.0  # kms^-1, max recession from QSO.

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

SDSS_z_min = SDSS_min/Lya-1



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
pickle_exists = os.path.exists(pkl_file)


t_start = time.time()

for i in range(num_QSOs):
    if pickle_exists:
        break
        

    random_index = np.random.randint(0,len(SDSS_data),1)
    z_QSO = SDSS_data['z'].iloc[random_index].values[0] # [0] selects the 1st (and only!) value in the array.

    spec_wv,spec_fl,spec_err,absorber_CDs,absorber_zs = generate_spectrum(z_QSO)

    prox_wv_min = 

    proximate_mask = (wave_min-100 < observed_wv) & (observed_wv < wave_max+100)

    prox_wave  = SDSS_wave[ proximate_mask ]
    prox_flux  = spectrum_flux[ proximate_mask ]
    prox_sigma = sigma[ proximate_mask ]

    try:
        prox_cont = QT.get_continuum(z_QSO,prox_wave,prox_flux,prox_sigma,hspc=7,kind='max')
        # continuum norm falls over when nmax ("Nmidpoints"?) is zero -- happens
        # for QSOs where only a few wv's are redward of the QSO because of SDSS limit 3800A.
    except:
        prox_cont = np.ones(len(prox_wave))

    norm_flux  = prox_flux/prox_cont
    norm_sigma = prox_sigma/prox_cont

    z_window = (1.0+z_QSO)*restframe_window
    Npix = int(z_window/SDSS_wv_resolution)

    score_wv,score_vals = evaluate_scores(prox_wave,norm_flux,norm_sigma,Npix)
    score_zs = score_wave/Lya - 1.0

    for j,th in enumerate(threshold_vals):
        
        

    



    
    '''
    ### Some plotting stuff...
    if len(absorber_CDs[ absorber_CDs > 20.3]) > 0:      
        gs = plt.GridSpec(4,3)
        fig=plt.figure()
        ax1,ax2,ax3 = plt.subplot(gs[:2,:]),plt.subplot(gs[2:3,:]),plt.subplot(gs[3:,:])
        ax1.plot(composite_wave,composite_flux,'k',alpha=0.5,label='true')
        ax1.plot(SDSS_wave,SDSS_flux,'b',label='convolved',alpha=0.5)
        ax1.plot(prox_wave,prox_flux,'k',drawstyle='steps',label='observed')
        ax1.plot(prox_wave,prox_cont,'r',label='est. continuum')
        ax1.plot(composite_wave,HR_composite_xspec.flux,'k',alpha=0.5)
        ax1.legend()
        ax1.text(SDSS_min-5.0,6.5,'SDSS limit',rotation=90,color='r',
                verticalalignment='center',horizontalalignment='center')
        ax1.set_ylim(-3.0,12.0)

        for wv,CD in zip(absorber_wvs,absorber_CDs):
            if CD > 17:
                ax1.plot([wv,wv],[-1.0,0.0],color='g')
                ax1.text(wv,-2.0,'%.1f'%CD,horizontalalignment='center',verticalalignment='center')

        ax2.plot(prox_wave,norm_flux,'k',drawstyle='steps')
        ax2.plot(prox_wave,norm_flux/norm_flux,'r',drawstyle='steps')

        ax3.plot(score_wave,score_vals,'k')
        #for i,fm in enumerate(feature_masks):
        #    ax3.plot(score_wave[fm], score_vals[fm],'g')
        #    ax3.axhline(th,color='g',ls='--')
        #    f = this_QSO[i]
        #    ax3.text((f[3]+1)*Lya,th-50,'%.1f'%f[2],horizontalalignment='center',verticalalignment='top')

        ax1.set_xticks([])
        ax2.set_xticks([])

        for ax in [ax1,ax2,ax3]:
            ax.set_xlim(wave_min-20,wave_max+20)   
            ax.axvline(SDSS_min,color='r',ls='--')

        ax1.set_ylabel(r'Flux')
        ax2.set_ylabel(r'Normed flux')
        ax3.set_ylabel(r'Score')
        ax3.set_xlabel(r'Wavelength $(\AA)$')

        fig.subplots_adjust(hspace=0)
        plt.show()
    '''



if pickle_exists is False:
    t_end = time.time()
    print('\n ~%i per second.' %round(num_QSOs/(t_end-t_start),0) )

    with open(pkl_file, 'wb') as f:
        pickle.dump(data_structure,f)
else:
    with open(pkl_file, 'rb') as f:
        data_structure = pickle.load(f)
    

cols = 'i z_QSO absorber_CDs absorber_zs absorber_bs feature_widths feature_zs absorber_bs'.split()

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
        
        detected_mask = (df['feature_widths'] > w )#*(1.0+df['z_QSO'])
        

        N_pos[k,j] = len( df[  true_DLA_mask & detected_mask ]) 
        N_neg[k,j] = len( df[ ~true_DLA_mask & detected_mask ])
        N_det[k,j] = len( df[ detected_mask ] )


# Interested in...
recall  = N_pos/N_true
contamination = N_neg/(N_neg + N_pos)
AUC = np.zeros(len(min_widths))


fig1 = plt.figure(figsize=(8,5))
ax1a = fig1.add_subplot(211)
ax1b = fig1.add_subplot(212)

fig2,ax2 = plt.subplots(figsize=(8,5))


for k,w in enumerate(min_widths):
    #plt.plot(threshold_vals,completeness[k,:],color='C%i'%k)
    
    Rk = recall[k,:]
    Ck = contamination[k,:]


    ax1a.plot(threshold_vals,Rk*(1.0-Ck),color='C%i'%k,label=w)
    ax1b.plot(threshold_vals,1.0-Ck,color='C%i'%k)
    ax1b.plot(threshold_vals,Rk,ls='--',color='C%i'%k)
    
    if k==0:
        ax1b.plot((0,0),(0,0),'k--',label='Contamination')
        ax1b.plot((0,0),(0,0),'k',label='Recall')
    ax2.plot(Ck,Rk,color='C%i'%k)
    
    Ck_mask = np.isfinite(Ck)
    Rk_mask = np.isfinite(Rk)

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


print(min_widths,AUC)
ax2_sub = fig2.add_axes([0.45,0.18,0.43,0.4])#,transform=ax2.transAxes)
ax2_sub.plot(min_widths,AUC,'k')
ax2_sub.set_ylabel(r'AUC',fontsize=18)




# Choose a few example thresholds to look at...
th_vals = [10.0,20.0,50.0]
th_indices = np.digitize(th_vals, threshold_vals)-1




### DO SOME EXTRA PLOTS...

width_bins = np.arange(0,15,0.2)
width_mids = QT.midpoints(width_bins)

nHI_bins = np.log10(nHI_vals)
nHI_mids = QT.midpoints(nHI_bins)

fig_scatter,ax_scatter = plt.subplots()
ax_scatter.set_xlabel(r'$\log_{10}\left(n_{\mathrm{HI}}\right)$ Column density')
ax_scatter.set_ylabel('Feature width [pix]')

fig_pcolor,ax_pcolor = plt.subplots()
ax_pcolor.set_xlabel(r'$\log_{10}\left(n_{\mathrm{HI}}\right)$ Column density')
ax_pcolor.set_ylabel(r'$\frac{\mathrm{Feature\;width\;[pix]}}{1+z_{QSO}}$',fontsize=20)

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

























