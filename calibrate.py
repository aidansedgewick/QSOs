import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.special import wofz
import astropy.io.fits as fits
import astropy.units as u

from specdb.specdb import IgmSpec
from pyigm.surveys.dlasurvey import DLASurvey
import dla_cnn.training_set as tset
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# from linetools.spectra.xspectrum1d import XSpectrum1D as xspec
import QSOtools as QT
import generate_fakespec as FS
import warnings

import pickle
import time
import sys
import os

### Some constants



###========Physics======###
c = 3.0E5  # km/s
pi = np.pi
b_min, b_max = 10.0,20.0 # Doppler parameters.
cold = 20.3     # H I Column density of absorber
zabs = 1.88     # Redshift of absorber
bval = 10.0     # Doppler parameter (km/s)
Lya = 1215.6701   # Rest frame Lya wavelength
Lylim = 911.3 # Restframe lyman limit.
fval = 0.4164       # Oscillator Strength
gamma = 6.265E8     # Einstein A coefficient

###========Instrument=======###
spec_min = 3800.0
spec_max = 9200.0
spec_res = 1.0 # IS IT?! CHECK!!!

###========Parameters=======###
DLA_def = 19.5
min_col_density = 19.5
v_min = -25000.0 # kms^-1, min recession from QSO.
v_max = 3000.0 #3000.0  # kms^-1, max recession from QSO.
MW = 400.0 #max_width - No *single* absorber likely to be wider than 400A...!?
rf_window = 3.5 # The width of a 20.3 absorber at rf?


### Some preferences
warnings.filterwarnings('ignore') # linetools is very noisy!

nspec = 5000
spec_pkl_loc = './calibrate_spec%i.pkl'%nspec

spec_pkl_exists = os.path.exists(spec_pkl_loc)

if spec_pkl_exists is False:
    all_spec, all_absorbers = FS.generate_fakespec(nspec,mix=True)
    with open(spec_pkl_loc, 'wb') as f:
        pickle.dump((all_spec,all_absorbers),f)
else:
    print('Pickle exists! Read %s'%spec_pkl_loc)
    with open(spec_pkl_loc, 'rb') as f:
        (all_spec, all_absorbers) = pickle.load(f)


# Still need the sightlines.
sdss = DLASurvey.load_SDSS_DR5(sample='all')
slines, sdict = tset.grab_sightlines(sdss, flg_bal=0)


threshold_vals = np.arange(-0.5,1.5,0.05) # A list of threshold values

# Initialize the horrible data strucure:
data_structure = [ [] for i in threshold_vals ]
cols = 'zQSO absorber_CDs absorber_zs feature_widths feature_zs k i specID snr'.split()

med_snr = []
z_vals = []

for i,spec in enumerate(all_spec):
    if i>5000:
        break

    absorbers = all_absorbers[i]
    isl = absorbers['sl'] # Find the i'th sightline
    nDLA = all_absorbers[i]['nDLA']

    #print(absorbers)

    #print(nDLA)

    zQSO = slines['ZEM'][isl]
    z_vals.append(zQSO)

    npix = (1.0+zQSO)*rf_window//spec_res

    wv = spec.wavelength.value
    fl = spec.flux.value
    er = 1.0/np.sqrt(spec.ivar.value)

    wvmin = Lylim*(1.0+zQSO)##spec.wvmin.value/Lya
    em_wv = Lya*(1.0+zQSO)

    absorber_CDs = np.array([absorbers[ab]['NHI'] for ab in range(nDLA)])
    absorber_zs = np.array([absorbers[ab]['zabs'] for ab in range(nDLA)])

    sorted_absorber_indices = np.argsort(absorber_CDs)[::-1]
    absorber_CDs = absorber_CDs[ sorted_absorber_indices ]
    absorber_zs = absorber_zs[ sorted_absorber_indices ]

    #prox_zmin = max(zQSO + v_min/c*(1.0+zQSO), wvmin-1.0 )
    #prox_zmax = max(zQSO + v_max/c*(1.0+zQSO), wvmin-1.0 + 0.01)

    #prox_wvmin,prox_wvmax = (1.0+prox_zmin)*Lya, (1.0+prox_zmax)*Lya

    prox_mask = (wvmin < wv) & (wv < em_wv) #(prox_wvmin < wv) & (wv < prox_wvmax)

    prox_wv = wv[ prox_mask ]
    prox_fl = fl[ prox_mask ]
    prox_er = er[ prox_mask ]

    #prox_co = co[ prox_mask ]

    try:
        prox_co = QT.get_continuum_alt2(zQSO,prox_wv,prox_fl)
    except:
        print("\033[31m  %i: Couldn't get continuum.\033[0m"%i)
        prox_co = np.ones(len(prox_wv))

    norm_fl = prox_fl/prox_co
    norm_er = prox_er/prox_co

    snr =  1.0/norm_er # As norm_fl = 1.0 everywhere! (suggested by RJC!)
    med_snr_val = np.nanmedian(snr)
    med_snr.append( med_snr_val )

    score_wv,score_vals = QT.evaluate_scores(prox_wv,norm_fl,norm_er,npix)

    for j, th in enumerate(threshold_vals):
        # Look at the width of the
        feature_widths, feature_zs = QT.extract_features(zQSO, score_wv, score_vals, threshold=th)

        # Match the widest feature to the largest column density
        QSO_data = (zQSO, absorber_CDs, absorber_zs, feature_widths, feature_zs)

        QSO_matches = QT.match_features(*QSO_data,extras=[i,med_snr_val])
        data_structure[j].extend(QSO_matches)

        Nmatches = min(len(feature_widths), len(absorber_CDs))

        delta_z = abs(feature_zs[:Nmatches] - absorber_zs[:Nmatches])
        '''if any(delta_z > 0.05):
            if nDLA < 2:
                continue

            print('\nthresh=%.2f'%th )
            print('True CD:', absorber_CDs)
            print('widths:', feature_widths[:10])

            print('true z:', absorber_zs)
            print('est_z:', feature_zs[:10])'''

    QT.write(i + 1, nspec)
    plotting = False

    if (plotting): # & (med_snr_val > 2.0):
        fig = plt.figure()
        gs = plt.GridSpec(4, 4)

        ax1 = fig.add_subplot(gs[:2, :])
        ax2 = fig.add_subplot(gs[2:3, :])
        ax3 = fig.add_subplot(gs[3:, :])
        fig.suptitle('z=%.2f' %zQSO)

        for ax in [ax1,ax2]:
            ax.set_xticks([])
            for j in range(nDLA):
                wvDLA = (1.0+absorbers[j]['zabs'])*Lya
                DLA_NHI = absorbers[j]['NHI']
                ax.axvline(wvDLA,color='r')
                if ax is ax2:
                    ax.text(wvDLA - 10.0, 2, '%.1f' % DLA_NHI, horizontalalignment='right')

        for ax in [ax1,ax2,ax3]:
            ax.set_xlim(np.min(prox_wv),np.max(prox_wv))



        ax1.fill_between(prox_wv,prox_fl-prox_er,prox_fl+prox_er,alpha=0.5)
        ax1.plot(prox_wv,prox_fl,drawstyle='steps')
        ax1.plot(prox_wv,prox_co)
        #ax1.scatter(knot_wv,knot_fl,marker='x',s=40,color='r',zorder=5)

        ax2.plot(prox_wv,norm_fl,drawstyle='steps')
        ax2.plot(prox_wv,prox_co/prox_co,'--')
        ax2.fill_between(prox_wv, norm_fl-norm_er,norm_fl+norm_er,alpha=0.5)
        #ax2.scatter(knot_wv,knot_fl/prox_co,marker='x',s=40,color='r',zorder=5)

        ax2.set_ylim(-1,2.5)

        test_th = 0.05

        feature_widths, feature_zs = QT.extract_features(zQSO, score_wv, score_vals, threshold=test_th)
        features = pd.DataFrame(QT.match_features(*QSO_data, extras=[i, med_snr_val]),columns=cols)


        print(features['feature_widths'],features['feature_zs'])
        for feature in features.itertuples():
            if feature.feature_widths > 2.0:
                ax3.axvline((feature.feature_zs+1)*Lya,color='g')

        if test_th:
            ax3.axhline(test_th,ls='--')


        ax3.plot(score_wv,score_vals,drawstyle='steps')
        ax3.set_ylim(-2,5)

        plt.subplots_adjust(hspace=0)

        plt.show()
print('')


bins = np.arange(0,7,0.1)

###================Some basic histograms==============###
fig,ax = plt.subplots(figsize=(6,6))
ax.hist(med_snr,bins=bins)
ax.set_xlabel('Median spec SNR')
#plt.show()

fig,ax = plt.subplots(figsize=(6,6))
ax.hist(z_vals,bins=bins)
ax.set_xlabel('Redshift z')
#plt.show()

fig = plt.figure(figsize=(6,6))
gs = plt.GridSpec(3,3)
axm = fig.add_subplot(gs[:2,1:])
axx,axy = fig.add_subplot(gs[-1:,1:]),fig.add_subplot(gs[:-1,:1])
hist2d,_,_ = np.histogram2d(z_vals,med_snr,bins=[bins,bins])
hist2d[ hist2d == 0] = np.nan
axm.pcolormesh(bins,bins,hist2d.T)
axx.hist(z_vals,bins=bins)
axy.hist(med_snr,bins=bins,orientation='horizontal')
fig.subplots_adjust(hspace=0,wspace=0)
axm.set_xlim(bins[0],bins[-1]),axm.set_ylim(bins[0],bins[-1])
axx.set_xlim(bins[0],bins[-1]),axy.set_ylim(bins[0],bins[-1])
#plt.show()

### Rearrange the data into a nicer format.

min_widths = np.arange(3.0,8.0,0.5)

N_pos = np.zeros( (len(min_widths),len(threshold_vals)) )
N_neg = np.zeros( (len(min_widths),len(threshold_vals)) )
N_det = np.zeros( (len(min_widths),len(threshold_vals)) )

# REMINDER:
# cols = 'zQSO absorber_CDs absorber_zs feature_widths feature_zs specID snr'.split()
for j, th in enumerate(threshold_vals):
    df = pd.DataFrame(data_structure[j], columns=cols)

    print('Threshold %.3f: %i features' %(th,len(df)))

    true_DLA_mask = (df['absorber_CDs'] > DLA_def)
    N_true = float(len(df[true_DLA_mask]))  # This should be the same every loop...!

    for k, w in enumerate(min_widths):
        # For detection above this minumum feature width,
        # how many true positives, false positives, total positives?

        detected_mask = (df['feature_widths'] > w)  # *(1.0+df['z_QSO'])

        N_pos[k, j] = float(len(df[true_DLA_mask & detected_mask]))
        N_neg[k, j] = float(len(df[~true_DLA_mask & detected_mask]))
        N_det[k, j] = float(len(df[detected_mask]))  # = N_pos+N_neg??

# Interested in...
recall = N_pos / N_true
contamination = N_neg / (N_neg + N_pos)

fig1,ax1 = plt.subplots(figsize=(8,4))
fig2,ax2 = plt.subplots(figsize=(8,4))
fig3,ax3 = plt.subplots(figsize=(6,6))

for k,w in enumerate(min_widths):
    rk = recall[k,:]
    ck = contamination[k,:]

    ax1.plot(threshold_vals,rk,'C%i'%k,label='%.1f'%w)
    ax1.plot(threshold_vals,ck,'C%i--'%k)

    ax2.plot(threshold_vals,rk*(1-ck),'C%i'%k,label='%.1f'%w)

    ax3.plot(ck,rk,'C%i'%k,label='%.1f'%w)

ax1.legend(ncol=2)
ax2.legend(ncol=2)
ax3.legend(ncol=2)
#plt.show()

snr_ranges=[0,0.5,1.0,2.0,9.9]
figs = [plt.figure(figsize=(12,6)) for i in range(3)]

for i,(low,high) in enumerate(zip(snr_ranges[:-1],snr_ranges[1:])):
    ax1,ax2,ax3 = (fig.add_subplot(2,2,i+1) for fig in figs)

    N_pos_i = np.zeros((len(min_widths), len(threshold_vals)))
    N_neg_i = np.zeros((len(min_widths), len(threshold_vals)))
    N_det_i = np.zeros((len(min_widths), len(threshold_vals)))

    for j, th in enumerate(threshold_vals):
        df = pd.DataFrame(data_structure[j], columns=cols)

        snr_mask = (low < df['snr']) & (df['snr'] < high)
        df = df[ snr_mask ]

        true_DLA_mask = (df['absorber_CDs'] > DLA_def)
        N_true = float(len(df[true_DLA_mask]))  # This should be the same every loop...!

        for k, w in enumerate(min_widths):
            # For detection above this minumum feature width,
            # how many true positives, false positives, total positives?

            detected_mask = (df['feature_widths'] > w)  # *(1.0+df['z_QSO'])

            N_pos_i[k, j] = float(len(df[true_DLA_mask & detected_mask]))
            N_neg_i[k, j] = float(len(df[~true_DLA_mask & detected_mask]))
            N_det_i[k, j] = float(len(df[detected_mask]))  # = N_pos+N_neg??

    # Interested in...
    recall_i = N_pos_i / N_true
    contamination_i = N_neg_i / (N_neg_i + N_pos)

    for k, w in enumerate(min_widths):
        rk_i = recall_i[k, :]
        ck_i = contamination_i[k, :]

        ax1.plot(threshold_vals, rk_i, 'C%i' % k, label='%.1f' % w)
        ax1.plot(threshold_vals, ck_i, 'C%i--' % k)

        ax2.plot(threshold_vals, rk_i * (1 - ck_i), 'C%i' % k, label='%.1f' % w)

        ax3.plot(ck_i, rk_i, 'C%i' % k, label='%.1f' % w)

        for ax in [ax1,ax2,ax3]:
            ax.set_title('%.1f<SNR<%.1f'%(low,high))

#plt.show()


z_ranges=[2,2.5,3.0,4.0,9.9]
figs = [plt.figure(figsize=(12,6)) for i in range(3)]

for i,(low,high) in enumerate(zip(z_ranges[:-1],z_ranges[1:])):
    ax1,ax2,ax3 = (fig.add_subplot(2,2,i+1) for fig in figs)

    N_pos_i = np.zeros((len(min_widths), len(threshold_vals)))
    N_neg_i = np.zeros((len(min_widths), len(threshold_vals)))
    N_det_i = np.zeros((len(min_widths), len(threshold_vals)))

    for j, th in enumerate(threshold_vals):
        df = pd.DataFrame(data_structure[j], columns=cols)

        z_mask = (low < df['zQSO']) & (df['zQSO'] < high) # df changes for each 'j'...
        df = df[ z_mask ]

        true_DLA_mask = (df['absorber_CDs'] > DLA_def)
        zN_true = float(len(df[true_DLA_mask]))  # This should be the same every loop...!

        for k, w in enumerate(min_widths):
            # For detection above this minumum feature width,
            # how many true positives, false positives, total positives?

            detected_mask = (df['feature_widths'] > w)  # *(1.0+df['z_QSO'])

            N_pos_i[k, j] = float(len(df[true_DLA_mask & detected_mask]))
            N_neg_i[k, j] = float(len(df[~true_DLA_mask & detected_mask]))
            N_det_i[k, j] = float(len(df[detected_mask]))  # = N_pos+N_neg??

    # Interested in...
    recall_i = N_pos_i / zN_true
    contamination_i = N_neg_i / (N_neg_i + N_pos_i)

    for k, w in enumerate(min_widths):
        rk_i = recall_i[k, :]
        ck_i = contamination_i[k, :]

        ax1.plot(threshold_vals, rk_i, 'C%i' % k, label='%.1f' % w)
        ax1.plot(threshold_vals, ck_i, 'C%i--' % k)

        ax2.plot(threshold_vals, rk_i * (1 - ck_i), 'C%i' % k, label='%.1f' % w)

        ax3.plot(ck_i, rk_i, 'C%i' % k, label='%.1f' % w)

        for ax in [ax1,ax2,ax3]:
            ax.set_title('%.1f<z<%.1f'%(low,high))

#plt.show()

dw = 2
width_bins = np.arange(5,35+dw,dw)
width_mids = QT.midpoints(width_bins)

plotting_th_vals = [0.05]  # threshold_vals.copy()

def logfunc(x,A,B,C):
    return A*np.log10(x+B) + C

reject = True
Nmad = 3 # Reject outliers at N*MAD from median


for k,w in enumerate(plotting_th_vals):
    iw = np.digitize(w,threshold_vals)-1

    print(w,round(threshold_vals[iw],3))

    df = pd.DataFrame(data_structure[iw], columns=cols)
    # A reminder: cols='zQSO absorber_CDs absorber_zs feature_widths feature_zs'.split()

    allNH1_mads = np.zeros(len(width_mids))
    allNH1_meds = np.zeros(len(width_mids))
    NH1_meds = np.zeros(len(width_mids))
    NH1_mads = np.zeros(len(width_mids))

    true_mask = (df['absorber_CDs'] > 0) # They are real
    detected_mask = (df['feature_widths'] > 0)

    for i, (low, high) in enumerate(QT.binedges(width_bins)):
        range_mask = (low < df['feature_widths']) & (df['feature_widths'] < high)
        tdf = df[ true_mask & detected_mask & range_mask ]
        med_i = np.median(tdf['absorber_CDs'])
        allNH1_meds[i] = med_i
        allNH1_mads[i] = QT.MAD(tdf['absorber_CDs'])
        if reject:
            accept_mask = (abs(tdf['absorber_CDs']-med_i) < Nmad*allNH1_mads[i])
            tdf = tdf[accept_mask]

            NH1_mads[i] = QT.MAD(tdf['absorber_CDs'])
            NH1_meds[i] = np.median(tdf['absorber_CDs'])

            if len(tdf) == 0:
                NH1_mads[i] = allNH1_mads[i]
                NH1_meds[i] = allNH1_meds[i]

        else:
            NH1_mads[i] = allNH1_mads[i]
            NH1_meds[i] = allNH1_meds[i]

    popt,pcov = curve_fit(logfunc,width_mids,NH1_meds,p0=[1,1,1])

    df = df[ true_mask & detected_mask ]

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    cp1 = ax1.scatter(df['feature_widths'],df['absorber_CDs'],c=df['zQSO'],s=1)
    ax1.scatter(width_mids,NH1_meds,color='r')
    ax1.scatter(width_mids,NH1_meds-NH1_mads,marker='x',color='r')
    ax1.plot(width_mids,logfunc(width_mids,*popt),color='r')
    ax1.plot(width_mids,logfunc(width_mids,*popt)+NH1_mads,'r--')
    ax1.plot(width_mids,logfunc(width_mids,*popt)-NH1_mads,'r--')

    #ax1.plot(width_mids,allNH1_meds-Nmad*allNH1_mads,'r:')
    #ax1.plot(width_mids,allNH1_meds+Nmad*allNH1_mads,'r:')

    fig1.colorbar(cp1)
    fig1.suptitle('th=%.2f'%w)
    ax1.set_ylim(19.5,22.5), ax1.set_xlim(0,50)
    #fig1.savefig('./figs/fig1_%03d.png'%k)
    #plt.close()

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    x2 = df['absorber_zs']
    y2 = df['absorber_zs'] - df['feature_zs']
    cp2 = ax2.scatter(x2,y2,c=df['absorber_CDs'],s=1)
    ax2.set_xlabel('z_DLA'),ax2.set_ylabel('z_DLA-z_est')
    ax2.set_xlim(2.0, 4.0), ax2.set_ylim(-1.0, 1.0)
    fig2.colorbar(cp2)
    fig2.suptitle('th=%.2f'%w)
    #fig2.savefig('./figs/fig2_%03d.png' %k)
    #plt.close()

    fig3, ax3 = plt.subplots(figsize=(6, 6))
    x3 = df['absorber_zs'] - df['zQSO']
    y3 = df['absorber_zs'] - df['feature_zs']
    cp3 = ax3.scatter(x3,y3,c=df['absorber_CDs'],s=1)
    ax3.set_xlabel('z_DLA-z_QSO'),ax3.set_ylabel('z_DLA-z_est')
    ax3.set_xlim(-1.4,0.0),ax3.set_ylim(-1.0,1.0)
    fig3.colorbar(cp3)
    fig3.suptitle('th=%.2f'%w)
    #fig3.savefig('./figs/fig3_%03d.png' %k)
    #plt.close()





#plt.show()

vals = [-0.1,0.0,0.4,1.0]
dNH1 = 0.5
NH1_bins = np.arange(19.5,22.5+dNH1,dNH1)
NH1_mids = 0.5*(NH1_bins[:-1] + NH1_bins[1:])

fig1, ax1 = plt.subplots(figsize=(6, 6))
#fig2, ax2 = plt.subplots(figsize=(6, 6))
#fig3, ax3 = plt.subplots(figsize=(6, 6))

for k,w in enumerate(vals):
    iw = np.digitize(w,threshold_vals)-1 # Which df from the structure to select...?
    df = pd.DataFrame(data_structure[iw], columns=cols)

    true_mask = (df['absorber_CDs'] > 0) # They are real
    detected_mask = (df['feature_widths'] > 0) # Did we find them.

    N_pos = df[ true_mask & detected_mask ]
    N_tot = df[ true_mask ]

    pos_hist,_ = np.histogram(N_pos['absorber_CDs'],bins=NH1_bins)
    tot_hist,_ = np.histogram(N_tot['absorber_CDs'],bins=NH1_bins)

    print('+ve:',pos_hist)
    print('tot:',tot_hist)

    ax1.plot(NH1_mids,pos_hist/tot_hist,label='%.1f'%w)

ax1.legend()
plt.show()



