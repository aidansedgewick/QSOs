import json, os, urllib, pdb, numpy as np
import astropy.io.fits as fits
from matplotlib import pyplot as plt
import dla_cnn
from dla_cnn.data_model.Sightline import Sightline
parks_dir = '/'.join(dla_cnn.__file__.split('/')[:-1])+'/models/'
model_checkpoint = parks_dir + 'model_gensample_v7.1'

def plot(y, x_label="Rest Frame", y_label="Flux", x=None, ylim=[-2, 12], xlim=None, z_qso=None):
    fig, ax = plt.subplots(figsize=(15, 3.75))
    if x is None:
        ax.plot(y, '-k')
    else:
        ax.plot(x, y, '-k')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.ylim(ylim)
    plt.xlim(xlim)

    return fig, ax


def parks_model(flux, loglam, z_qso, plot=False):
    idnum = 0
    sl = Sightline(idnum, dlas=None, flux=flux, loglam=loglam, z_qso=z_qso)
    sl.process(model_checkpoint)
    if plot:
        lam = 10.0 ** loglam
        lam_rest = lam / (1.0 + z_qso)
        ix_dla_range = np.logical_and(lam_rest >= REST_RANGE[0], lam_rest <= REST_RANGE[1])
        y_plot_range = np.mean(flux[np.logical_not(np.isnan(flux))]) + 10
        plt.plot(10.0**loglam, flux, 'k-', drawstyle='steps')
        for ii in range(len(sl.dlas)):
            plt.axvline(sl.dlas[ii], 'r-')
        for ii in range(len(sl.subdlas)):
            plt.axvline(sl.subdlas[ii], ymin=0.0, ymax=y_plot_range, 'b-')
        plt.show()
    return sl

if __name__ == '__main__':
    #09:03:33.55 +26:28:36.3
    fname = "spec-5778-56328-0546.fits"
    fil = fits.open(fname)
    flux, loglam = fil[1].data['flux'], fil[1].data['loglam']
    z_qso = 3.219
    parks_model(flux, loglam, z_qso, plot=True)

"""
with open(model_checkpoint + "_hyperparams.json", 'r') as fp:
    hyperparameters = json.load(fp)
    loc_pred, loc_conf = predictions_ann_c2(hyperparameters, c2_dataset.fluxes,
                                            c2_dataset.labels, c2_offsets, model_checkpoint)

(fig, ax) = plot(loc_conf, ylim=[0, 1], x=lam_rest[ix_dla_range], xlim=[REST_RANGE[0], 1250],
                 z_qso=z_qso, x_label="DLA Localization confidence & localization prediction(s)")

# Identify peaks from classification-2 results
(peaks, peaks_uncentered, smoothed_sample, ixs_left, ixs_right) = \
predictions_to_central_wavelength(loc_conf, 1, 50, 300)[0]
#     print(np.shape(peaks), np.shape(peaks_uncentered), np.shape(loc_conf), np.shape(smoothed_sample))
ax.plot(lam_rest[ix_dla_range], smoothed_sample, color='blue', alpha=0.9)

for peak, peak_uncentered, ix_left, ix_right in zip(peaks, peaks_uncentered, ixs_left, ixs_right):
    peak_lam_rest = lam_rest[ix_dla_range][peak]
    if peak_lam_rest > 1250 or peak_lam_rest < REST_RANGE[0]:
        print(" > Excluded peak: %0.0fA" % peak_lam_rest)
        continue

    # Plot peak '+' markers
    ax.plot(lam_rest[ix_dla_range][peak_uncentered], loc_conf[peak_uncentered], '+', mew=3, ms=7, color='red',
            alpha=1)
    ax.plot(lam_rest[ix_dla_range][peak], smoothed_sample[peak], '+', mew=7, ms=15, color='blue', alpha=0.9)
    ax.plot(lam_rest[ix_dla_range][ix_left], loc_conf[peak_uncentered] / 2, '+', mew=3, ms=7, color='orange',
            alpha=1)
    ax.plot(lam_rest[ix_dla_range][ix_right], loc_conf[peak_uncentered] / 2, '+', mew=3, ms=7, color='orange',
            alpha=1)

    # Column density estimate
    density_data = DataSet(scan_flux_about_central_wavelength(data1['flux'], data1['loglam'], z_qso,
                                                              peak_lam_rest * (1 + z_qso), 0, 80, 0, 0, 0, 400,
                                                              0.2))

    with open(MODEL_CHECKPOINT_R1 + "_hyperparams.json", 'r') as fp:
        hyperparameters_r1 = json.load(fp)
        density_pred = predictions_ann_r1(hyperparameters_r1, density_data.fluxes,
                                          density_data.labels, MODEL_CHECKPOINT_R1)
        density_pred_np = np.array(density_pred)

    mean_col_density_prediction = np.mean(density_pred_np)

    # Bar plot
    fig_b, ax_b = plt.subplots(figsize=(15, 3.75))
    ax_b.bar(np.arange(0, np.shape(density_pred_np)[1]), density_pred_np[0, :], 0.25)
    ax_b.set_xlabel("Individual Column Density estimates for peak @ %0.0fA, +/- 0.3 of mean. " % (peak_lam_rest) +
                    "Mean: %0.3f - Median: %0.3f - Stddev: %0.3f" % (np.mean(density_pred_np),
                                                                     np.median(density_pred_np),
                                                                     np.std(density_pred_np)))
    plt.ylim([mean_col_density_prediction - 0.3, mean_col_density_prediction + 0.3])
    ax_b.plot(np.arange(0, np.shape(density_pred_np)[1]),
              np.ones((np.shape(density_pred_np)[1],), np.float32) * mean_col_density_prediction)

    # Sightline plot transparent marker boxes
    ax_sight.fill_between(lam_rest[ix_dla_range][peak - 10:peak + 10], y_plot_range, -2, color='gray', lw=0,
                          alpha=0.1)
    ax_sight.fill_between(lam_rest[ix_dla_range][peak - 30:peak + 30], y_plot_range, -2, color='gray', lw=0,
                          alpha=0.1)
    ax_sight.fill_between(lam_rest[ix_dla_range][peak - 50:peak + 50], y_plot_range, -2, color='gray', lw=0,
                          alpha=0.1)
    ax_sight.fill_between(lam_rest[ix_dla_range][peak - 70:peak + 70], y_plot_range, -2, color='gray', lw=0,
                          alpha=0.1)

    print(
        " > DLA central wavelength at: %0.0fA rest / %0.0fA spectrum w/ confidence %0.2f, has Column Density: %0.3f"
        % (peak_lam_rest, peak_lam_rest * (1 + z_qso), smoothed_sample[peak], mean_col_density_prediction))

handles, labels = ax.get_legend_handles_labels()
ax.legend(['DLA pred', 'Smoothed pred', 'Original peak', 'Recentered peak', 'Centering points'],
          bbox_to_anchor=(0.25, 1.1))

# Print URL
print(" > http://dr12.sdss3.org/spectrumDetail?plateid=%d&mjd=%d&fiber=%d" % (pmf[0], pmf[1], pmf[2]))
"""