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

nspec = 20
all_spec, all_absorbers = FS.generate_fakespec(nspec)
sdss = DLASurvey.load_SDSS_DR5(sample='all')
slines, sdict = tset.grab_sightlines(sdss, flg_bal=0)
# qq=0  # i.e. qq goes from 0 to nspec-1...
# isl = full_dict[qq]['sl']
# zqso = slines['ZEM'][isl]

# print(all_absorbers)
# print(full_dict[qq])

# print(isl)
# print(zqso)

DLA_def = 20.3
min_col_density = 15.0
min_feature_width = 5
# No *single* absorber is likely to be wider than 400 Ang...!?
#MW = 400.0 #max_width
#num_QSOs = 10000

threshold_vals = np.arange(-10,10,0.2) # A list of threshold values

# Initialize the horrible data strucure:
data_structure = [ [] for i in threshold_vals ]
cols = 'zQSO absorber_CDs absorber_zs feature_widths feature_zs'.split()


# If we've done it before...
#pickle_exists = os.path.exists(pkl_file_loc)

for i,spec in enumerate(all_spec):

    absorbers = all_absorbers[i]
    isl = absorbers['sl'] # Find the i'th sightline
    nDLA = all_absorbers[i]['nDLA']

    zQSO = slines['ZEM'][isl]

    npix = (1.0+zQSO)*rf_window//spec_res

    wvmin = spec.wvmin.value/Lya

    wv = spec.wavelength.value
    fl = spec.flux.value
    er = 1.0/np.sqrt(spec.ivar.value)

    em_wv = Lya*(1.0+zQSO)

    absorber_CDs = np.array([absorbers[ab]['NHI'] for ab in range(nDLA)])
    absorber_zs = np.array([absorbers[ab]['zabs'] for ab in range(nDLA)])

    #prox_zmin = max(zQSO + v_min/c*(1.0+zQSO), wvmin-1.0 )
    #prox_zmax = max(zQSO + v_max/c*(1.0+zQSO), wvmin-1.0 + 0.01)

    #prox_wvmin = (1.0+prox_zmin)*Lya
    #prox_wvmax = (1.0+prox_zmax)*Lya

    prox_mask = (spec_min < wv) & (wv < em_wv) #(prox_wvmin < wv) & (wv < prox_wvmax)

    prox_wv = wv[ prox_mask ]
    prox_fl = fl[ prox_mask ]
    prox_er = er[ prox_mask ]

    #prox_co = co[ prox_mask ]

    absorbers = all_absorbers[i]

    try:
        prox_co,knot_wv,knot_fl = QT.get_continuum(zQSO,prox_wv,prox_fl,prox_er,hspc=10,return_knots=True)
    except:
        print("\033[31m  %i: Couldn't get continuum.\033[0m"%i)
        prox_co = np.ones(len(prox_wv))

    print(np.min(prox_wv),np.max(prox_wv))

    norm_fl = prox_fl/prox_co
    norm_er = prox_er/prox_co


    score_wv,score_vals = QT.evaluate_scores(prox_wv,norm_fl,norm_er,npix)

    for j, th in enumerate(threshold_vals):
        # Look at the width of the
        feature_widths, feature_zs = QT.extract_features(zQSO, score_wv, score_vals, threshold=th)

        # Match the widest feature to the largest column density
        QSO_matches = QT.match_features(zQSO, absorber_CDs, absorber_zs, feature_widths, feature_zs)
        data_structure[j].extend(QSO_matches)

    QT.write(i + 1, nspec)


    plotting = True

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
                ax.axvline((1.0+absorbers[j]['zabs'])*Lya,color='r')

        for ax in [ax1,ax2,ax3]:
            ax.set_xlim(np.min(prox_wv),np.max(prox_wv))

        ax1.fill_between(prox_wv,prox_fl-prox_er,prox_fl+prox_er,alpha=0.5)
        ax1.plot(prox_wv,prox_fl,drawstyle='steps')
        ax1.plot(prox_wv,prox_co)
        ax1.scatter(knot_wv,knot_fl,marker='x',s=40,color='r',zorder=5)

        ax2.plot(prox_wv,norm_fl,drawstyle='steps')
        ax2.plot(prox_wv,prox_co/prox_co,'--')
        ax2.fill_between(prox_wv, norm_fl-norm_er,norm_fl+norm_er,alpha=0.5)
        #ax2.scatter(knot_wv,knot_fl/prox_co,marker='x',s=40,color='r',zorder=5)

        ax3.plot(score_wv,score_vals)

        plt.subplots_adjust(hspace=0)

        plt.show()


### Rearrange the data into a nicer format.

min_widths = np.arange(1.5,6.5,0.5)

N_pos = np.zeros( (len(min_widths),len(threshold_vals)) )
N_neg = np.zeros( (len(min_widths),len(threshold_vals)) )
N_det = np.zeros( (len(min_widths),len(threshold_vals)) )


for j, th in enumerate(threshold_vals):
    df = pd.DataFrame(data_structure[j], columns=cols)

    true_DLA_mask = (df['absorber_CDs'] > DLA_def)
    N_true = float(len(df[true_DLA_mask]))  # This should be the same every loop...!
    print(N_true)

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

# An array to save the values of area under the ROC curve.
AUC = np.zeros(len(min_widths))

# Make some axes to do the plots...
fig1 = plt.figure(figsize=(8, 5))
ax1a = fig1.add_subplot(211)
ax1b = fig1.add_subplot(212)

fig2, ax2 = plt.subplots(figsize=(8, 5))

for k, w in enumerate(min_widths):
    # plt.plot(threshold_vals,completeness[k,:],color='C%i'%k)

    # Recall & contam for min_width[k], the slice is for 'neatness'
    Rk = recall[k, :]
    Ck = contamination[k, :]

    ax1a.plot(threshold_vals, Rk * (1.0 - Ck), color='C%i' % k, label=w)
    ax1b.plot(threshold_vals, 1.0 - Ck, color='C%i' % k)
    ax1b.plot(threshold_vals, Rk, ls='--', color='C%i' % k)

    if k == 0:  # Only want one solid & one dash line in the legend.
        ax1b.plot((0, 0), (0, 0), 'k--', label='Contamination')
        ax1b.plot((0, 0), (0, 0), 'k', label='Recall')
    ax2.plot(Ck, Rk, color='C%i' % k)

    Ck_mask = np.isfinite(Ck)  # Can't do the AUC integration --
    Rk_mask = np.isfinite(Rk)  # -- if we have NaN values!

    try:
        AUC[k] = integrate.trapz(Rk[Ck_mask & Rk_mask][::-1], x=Ck[Ck_mask & Rk_mask][::-1])
    except:
        print("%.1f: can't integrate")

fig1.subplots_adjust(hspace=0)
ax1a.legend(ncol=2)
ax1b.legend()
ax1a.set_ylabel(r'$R\times\left(1-C\right)$', fontsize=18)
ax1b.set_ylabel(r'$R,C$', fontsize=18)
ax1b.set_xlabel(r'Threshold score', fontsize=18)

ax2.set_xlabel(r'Contamination $C$', fontsize=18)
ax2.set_ylabel(r'Recall $R$', fontsize=18)

ax2_sub = fig2.add_axes([0.45, 0.18, 0.43, 0.4])  # ,transform=ax2.transAxes)
ax2_sub.plot(min_widths, AUC, 'k')
ax2_sub.set_ylabel(r'AUC', fontsize=18)

# Choose a few example thresholds to look at...
th_vals = [10.0, 20.0, 50.0]
th_indices = np.digitize(th_vals, threshold_vals) - 1

### Some more plots...

width_bins = np.arange(0, 15, 0.2)
width_mids = QT.midpoints(width_bins)

nHI_bins = np.log10(QT.nHI_vals)
nHI_mids = QT.midpoints(nHI_bins)

fig_scatter, ax_scatter = plt.subplots()
ax_scatter.set_xlabel(r'$\log_{10}\left(n_{\mathrm{HI}}\right)$ Column density')
ax_scatter.set_ylabel('Feature width [pix]')

fig_pcolor, ax_pcolor = plt.subplots()
ax_pcolor.set_xlabel(r'$\log_{10}\left(n_{\mathrm{HI}}\right)$ Column density')
ax_pcolor.set_ylabel(r'$\frac{\mathrm{Feature\;width\;[pix]}}{1+z_{QSO}}$', fontsize=20)

# Contour plot is a failed experiment.
fig_contour, ax_contour = plt.subplots()
ax_contour.set_xlabel(r'$\log_{10}\left(n_{\mathrm{HI}}\right)$ Column density')
ax_contour.set_ylabel(r'$\frac{\mathrm{Feature\;width\;[pix]}}{1+z_{QSO}}$', fontsize=20)

fig_hist, ax_hist = plt.subplots()
ax_hist.set_xlabel(r'$z_{est}-z_{abs}$', fontsize=20)

fig_zerr = plt.figure()
gs = plt.GridSpec(3, 3)
ax_zerr1 = fig_zerr.add_subplot(gs[:2, :])
ax_zerr2 = fig_zerr.add_subplot(gs[2:, :])
ax_zerr2.set_xlabel(r'$z_{QSO}-z_{abs}$', fontsize=20)
ax_zerr1.set_ylabel(r'$z_{est}-z_{abs}$', fontsize=20)

for i, th in enumerate(th_vals):
    df = pd.DataFrame(data_structure[th_indices[i]], columns=cols)

    true_absorber_mask = (df['absorber_CDs'] > 0)
    detected_mask = (df['feature_widths'] > 3)

    # Look only at the true stuff that we've detected...
    df = df[true_absorber_mask & detected_mask]

    ax_scatter.scatter(df['absorber_CDs'], df['feature_widths'], color='C%i' % i, s=2)

    hist2d, _, _ = np.histogram2d(df['absorber_CDs'].values, df['feature_widths'].values, bins=[nHI_bins, width_bins])
    hist1d, _ = np.histogram(df['absorber_CDs'], bins=nHI_bins)
    hist2d = hist2d / hist1d[:, None]
    hist2d[hist2d == 0] = np.nan
    if i == 0:
        cp = ax_pcolor.pcolormesh(nHI_bins, width_bins, hist2d.T, vmin=0)
        plt.colorbar(cp, ax=ax_pcolor)

    xx, yy = np.meshgrid(nHI_mids, width_mids)
    # ax_contour.contour(xx,yy,hist2d.T,levels=[0.01,0.05,0.1],color='C%i'%i)

    ax_hist.hist(df['absorber_zs'] - df['feature_zs'], bins=np.linspace(-0.025, 0.025, 50), color='C%i' % i,
                 histtype='step', density=True)

    ax_zerr1.scatter(df['absorber_zs'] - df['zQSO'], df['feature_zs'] - df['absorber_zs'], s=2)

plt.show()
