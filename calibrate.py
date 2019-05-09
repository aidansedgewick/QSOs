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

# from linetools.spectra.xspectrum1d import XSpectrum1D as xspec
import QSOtools as QT
import generate_fakespec as FS
import warnings

import pickle
import time
import sys
import os

### Some constants

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


rf_window = 3.5 ## Angstrom.

spec_min = 3800.0
spec_max = 9200.0 # IS IT?! CHECK!!!
spec_res = 2.0


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





sdss = DLASurvey.load_SDSS_DR5(sample='all')
slines, sdict = tset.grab_sightlines(sdss, flg_bal=0)
# qq=0  # i.e. qq goes from 0 to nspec-1...
# isl = full_dict[qq]['sl']
# zqso = slines['ZEM'][isl]

# print(all_absorbers)
# print(full_dict[qq])

# print(isl)
# print(zqso)

DLA_def = 19.5
#min_col_density = 15.0
min_feature_width = 5
# No *single* absorber is likely to be wider than 400 Ang...!?
#MW = 400.0 #max_width
#num_QSOs = 10000

threshold_vals = np.arange(-1,9,0.2) # A list of threshold values

# Initialize the horrible data strucure:
data_structure = [ [] for i in threshold_vals ]

med_snr = []
z_vals = []


# If we've done it before...
#pickle_exists = os.path.exists(pkl_file_loc)

for i,spec in enumerate(all_spec):
    if i>5000:
        break

    absorbers = all_absorbers[i]
    isl = absorbers['sl'] # Find the i'th sightline
    nDLA = all_absorbers[i]['nDLA']

    zQSO = slines['ZEM'][isl]
    z_vals.append(zQSO)

    npix = (1.0+zQSO)*rf_window//spec_res

    wv = spec.wavelength.value
    fl = spec.flux.value
    er = 1.0/np.sqrt(spec.ivar.value)

    em_wv = Lya*(1.0+zQSO)
    wvmin = spec.wvmin.value/Lya

    absorber_CDs = np.array([absorbers[ab]['NHI'] for ab in range(nDLA)])
    absorber_zs = np.array([absorbers[ab]['zabs'] for ab in range(nDLA)])

    #prox_zmin = max(zQSO + v_min/c*(1.0+zQSO), wvmin-1.0 )
    #prox_zmax = max(zQSO + v_max/c*(1.0+zQSO), wvmin-1.0 + 0.01)

    #prox_wvmin,prox_wvmax = (1.0+prox_zmin)*Lya, (1.0+prox_zmax)*Lya

    prox_mask = (spec_min < wv) & (wv < em_wv) #(prox_wvmin < wv) & (wv < prox_wvmax)

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

    snr =  1.0/norm_er # As norm_fl = 1.0 everywhere!
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

    QT.write(i + 1, nspec)


    plotting = False

    if plotting:

        gs = plt.GridSpec(4, 4)

        fig = plt.figure()

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
                    ax.text(wvDLA - 10.0, -2, '%.1f' % DLA_NHI, horizontalalignment='right')

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

        ax3.plot(score_wv,score_vals)
        ax3.set_ylim(-2,5)

        plt.subplots_adjust(hspace=0)

        plt.show()
print('')


bins = np.arange(0,7,0.1)


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
plt.show()

### Rearrange the data into a nicer format.

min_widths = np.arange(3.0,8.0,0.5)

N_pos = np.zeros( (len(min_widths),len(threshold_vals)) )
N_neg = np.zeros( (len(min_widths),len(threshold_vals)) )
N_det = np.zeros( (len(min_widths),len(threshold_vals)) )

cols = 'zQSO absorber_CDs absorber_zs feature_widths feature_zs specID snr'.split()
for j, th in enumerate(threshold_vals):
    df = pd.DataFrame(data_structure[j], columns=cols)

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
plt.show()


fig1,ax1 = plt.subplots(figsize=(6,6))

vals = [0.2]

for k,w in enumerate(vals):


    iw = np.digitize(w,threshold_vals)-1

    print(w,threshold_vals[iw])

    df = pd.DataFrame(data_structure[iw], columns=cols)
    # A reminder: cols='zQSO absorber_CDs absorber_zs feature_widths feature_zs'.split()

    mask = (df['absorber_CDs'] > 0)

    ax1.scatter(df['absorber_CDs'],df['feature_widths'],s=1)

plt.show()

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

plt.show()


z_ranges=[2,2.5,3.0,4.0,9.9]
figs = [plt.figure(figsize=(12,6)) for i in range(3)]

for i,(low,high) in enumerate(zip(z_ranges[:-1],z_ranges[1:])):
    ax1,ax2,ax3 = (fig.add_subplot(2,2,i+1) for fig in figs)

    N_pos_i = np.zeros((len(min_widths), len(threshold_vals)))
    N_neg_i = np.zeros((len(min_widths), len(threshold_vals)))
    N_det_i = np.zeros((len(min_widths), len(threshold_vals)))

    for j, th in enumerate(threshold_vals):
        df = pd.DataFrame(data_structure[j], columns=cols)

        z_mask = (low < df['zQSO']) & (df['zQSO'] < high)
        df = df[ z_mask ]

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
            ax.set_title('%.1f<z<%.1f'%(low,high))

plt.show()


wv_ranges=[3900.0,4500.0,5000.0,6000.0,7000.0]
figs = [plt.figure(figsize=(12,6)) for i in range(3)]

for i,(low,high) in enumerate(zip(wv_ranges[:-1],wv_ranges[1:])):
    ax1,ax2,ax3 = (fig.add_subplot(2,2,i+1) for fig in figs)

    N_pos_i = np.zeros((len(min_widths), len(threshold_vals)))
    N_neg_i = np.zeros((len(min_widths), len(threshold_vals)))
    N_det_i = np.zeros((len(min_widths), len(threshold_vals)))

    for j, th in enumerate(threshold_vals):
        df = pd.DataFrame(data_structure[j], columns=cols)

        wv_mask = (low < Lya*(1+df['absorber_zs'])) & (Lya*(1+df['absorber_zs']) < high)
        df = df[ wv_mask ]

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
            ax.set_title('%.1f<wv<%.1f'%(low,high))

plt.show()