"""
This script makes synthetic data to compare to the atr- control
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

from gcamp_analysis.plot import df_f_figs
from gcamp_analysis.analysis import Experiment, Traces

from rotation_talk_figs import (FULL_SCREEN, save_fig)

IMG_DIR = Path("/mnt/c/Users/peter/Downloads/ciruits")
ATR_MINUS_PATH = Path("/mnt/c/Users/peter/data/dbdA08a-optogenetics/data/exps/11-UNDRIFTED211115LexA_dbdGal4_lacZ_LexAopGCaMP6m_UASChrim_L2_Animal2_-ATR.tif.pickle")

def prepare_experiment():
    """
    Creates the Experiment from tiff
    """
    img_path = Path("/mnt/c/Users/peter/data/dbdA08a-optogenetics/data/undrifted/UNDRIFTED211115LexA_dbdGal4_lacZ_LexAopGCaMP6m_UASChrim_L2_Animal2_-ATR.tif")
    exp = Experiment.fromTif(img_path)
    exp.save(Path("/mnt/c/Users/peter/data/dbdA08a-optogenetics/data/exps") 
             / f"11-{img_path.name}.pickle")
    
def simulate_data(exp) -> Experiment:
    """
    generate synthetic data based on exp
    """

    t_with_nans, raw_f, _ = exp.get_only_traces().get_stim_f(stim_meta=exp.stim_meta)
    nan_mask = np.logical_not(np.isnan(raw_f))
    t_with_nans = t_with_nans + (raw_f * 0)
    t = t_with_nans[nan_mask]
    raw_f = raw_f[nan_mask]

    m, b = np.polyfit(t, np.log(raw_f), 1)
    log_fit = np.exp(m*t_with_nans + b)

    np.random.seed(1000)
    noise = np.cumsum(np.random.normal(size=len(t_with_nans)))
    noise = filtfilt(*butter(4, .005, 'high'), noise)
    noise = noise * np.std(raw_f) / (2*np.std(noise))
    
    # reconnect all nans
    sim = (log_fit + noise)[nan_mask]

    out_traces = Traces(sim)
    return Experiment("simulated", exp.stim_meta, {"seg": out_traces})

def compare_traces(real, simulated):
    """
    makes sample traces for real and simulated data
    """
    fig, axs = plt.subplots(3, 2, tight_layout=True, sharey="row", sharex=True,
                            gridspec_kw={"height_ratios": [.2, 2, 1]})
    
    df_f_figs.plot_trace_on_ax(axs[0, 0], axs[1, 0], axs[2, 0], experiment=real)
    df_f_figs.plot_trace_on_ax(axs[0, 1], axs[1, 1], axs[2, 1], experiment=simulated)
    # format
    axs[0, 0].set_title("ATR-")
    axs[0, 1].set_title("Simulated")
    fig.set_size_inches(FULL_SCREEN)
    save_fig(IMG_DIR / "sim_traces", fig)

def compare_stim_avgs(real, simulated):
    """
    plots both stim averages
    """
    fig, axs = plt.subplots(ncols=2, sharey=True)
    df_f_figs.plot_stim_avg_on_ax(axs[0], real)
    df_f_figs.plot_stim_avg_on_ax(axs[1], simulated)
    # format
    axs[0].set_title("ATR-")
    axs[1].set_title("Simulated")
    fig.set_size_inches(FULL_SCREEN)
    save_fig(IMG_DIR / "sim_stim_avg", fig)



def main():
    """
    run as a script
    """
    # prepare_experiment()
    exp = Experiment.load(ATR_MINUS_PATH)
    with plt.style.context("dark_background"):
        simulated = simulate_data(exp)
        compare_traces(exp, simulated)
        compare_stim_avgs(exp, simulated)
        plt.show()

if __name__ == '__main__':
    main()
