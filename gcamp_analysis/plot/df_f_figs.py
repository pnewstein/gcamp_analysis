"""
Generates pretty df/f figs, both phase locked average, df/f and raw
fluorescence
"""
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..analysis import StimAvg, StimMeta, Traces
from .scalebar import add_scalebar

# typedefs
Number = Union[float, int]
Color = Union[Tuple[Number, Number, Number], str]

# constants
C_ATR_MINUS: Color = (.98, .59, 0)
C_ATR_PLUS: Color = "m"
ZERO_LINE_ALPHA = .5

C_3d: Color = 0.12, 0.53, 0.9 #(0, 0, 0.88) # blue
C_4d: Color = 1, 0.76, 0.03 #(0.74, 0.02, 0.27) # maroon

def plot_stim_avg_on_ax(ax, stim_meta: StimMeta, stim_avg: StimAvg,
                        atr_status=True):
    """
    Plots the stim_avg error bar plot on axs returning ax
    """
    if atr_status:
        c_stim = C_ATR_PLUS
    else:
        c_stim = C_ATR_MINUS
    # config
    c_trace = "c"
    alpha_err = .3
    alpha_stim = .3

    # make the x vars
    pre_stim_x = np.linspace(
        -stim_meta.frame_time * stim_avg.pre_stim_df.shape[1],
        -stim_meta.frame_time,
        stim_avg.pre_stim_df.shape[1]
    )
    post_stim_x = np.linspace(
        0,
        stim_avg.post_stim_df.shape[1]*stim_meta.frame_time,
        stim_avg.post_stim_df.shape[1]
    )+stim_meta.stim_length

    # plot the means
    ax.plot(pre_stim_x, stim_avg.pre_stim_mean, c_trace)
    ax.plot(post_stim_x, stim_avg.post_stim_mean, c_trace)

    def plot_err_bar(ax, x, y, err, alpha, color):
        ax.fill_between(
            x,
            y + err,
            y - err,
            alpha=alpha, edgecolor=None, facecolor=color
        )
        return ax

    # plot the error bars
    ax = plot_err_bar(ax, pre_stim_x, stim_avg.pre_stim_mean,
                      stim_avg.pre_stim_std, alpha_err, c_trace)
    ax = plot_err_bar(ax, post_stim_x, stim_avg.post_stim_mean,
                      stim_avg.post_stim_std, alpha_err, c_trace)

    # plot the dotted line guess
    ax.plot(
        [-stim_meta.frame_time, stim_meta.stim_length], 
        [stim_avg.pre_stim_mean[-1], stim_avg.post_stim_mean[0]],
        color=c_stim, linestyle="dashed"
    )
    # shade out stim
    ax.axvspan(
        -stim_meta.frame_time, stim_meta.stim_length,
        color=c_stim, alpha=alpha_stim, edgecolor=None
    )
    # add the df/f=0 line
    ax.axhline(0, linestyle="dashed", alpha=ZERO_LINE_ALPHA)
    # add labels
    ax.set_ylabel("ΔF/F")
    ax.set_xlabel("Time (s)")

    # format
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
    return ax

def plot_stim_avg(stim_meta: StimMeta, stim_avg: StimAvg,
                  atr_status=True):
    """
    plots stim_avg on an ax returning fig, axs
    """
    fig, ax = plt.subplots()
    ax = plot_stim_avg_on_ax(ax, stim_meta, stim_avg, atr_status=atr_status)
    return fig, ax

def plot_trace_on_ax(stim_ax, f_ax, df_f_ax,
                     traces: Traces, stim_meta: StimMeta, 
                     stim_color: Optional[Color] = None,
                     chan=0):
    """
    Plots a nice sample trace on axs
    """
    if stim_color is None: stim_color = "m"
    t, f, df_f = traces.get_stim_f(stim_meta, chan=chan)
    # plot stim
    if stim_ax is not None:
        stim = np.arange(
            stim_meta.start_frame, len(f),
            stim_meta.i_stim_i+stim_meta.n_iterations
        )
        for x in stim:
            stim_ax.axvline(x=x*stim_meta.frame_time, color=stim_color)
        stim_ax.set_ylabel("Stimulus")
    # plot raw_f
    if f_ax is not None:
        f_ax.plot(t, f, color="c")
        f_ax.set_ylabel("Fluorescence")
    # plot df_f
    if df_f_ax is not None:
        df_f_ax.plot(t, df_f, color="c")
        df_f_ax.axhline(0, linestyle="dashed")#, alpha=ZERO_LINE_ALPHA)
        df_f_ax.set_ylabel("ΔF/F")

    for ax in (stim_ax, f_ax, df_f_ax):
        if ax is None: continue
        ax.xaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])

def plot_trace(traces: Traces, stim_meta: StimMeta, df_f=True, atr_plus=True):
    """
    plots the trace, returns fig and axs
    """
    if atr_plus:
        stim_color = C_ATR_PLUS
    else:
        stim_color = C_ATR_MINUS
    fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True,
                            gridspec_kw={"height_ratios": [.2, 1]})
    if df_f:
        df_f_ax = axs[1]
        f_ax = None
    else:
        df_f_ax = None
        f_ax = axs[1]
    plot_trace_on_ax(
        axs[0], f_ax, df_f_ax, traces=traces, stim_meta=stim_meta,
        stim_color=stim_color
    )
    fig.set_size_inches(6, 2)
    
    # add_scalebar
    if df_f:
        y_lim = axs[1].get_ylim()
        ytick = round((y_lim[1] - y_lim[0]) / .4) / 10
        y_kwargs = {"sizey": ytick, "labely": f"{ytick: .1f}",
                    "matchy": False}
    else:
        y_kwargs = {}
    add_scalebar(
        axs[1], matchx=False, hidey=False, sizex=60, labelx="1 min", **y_kwargs
    )
    return fig, axs


def plot_stim_summary_on_axs(ax, data: pd.DataFrame, size=None, add_legend=True):
    """
    Plots a stim summary swarm plot on ax

    Args:
    ax: the pyplot axis to plot the data
    data (pd.DataFrame): the data with columns:
        "atr_status", "age", "df_f" 
        (bool)        (int)  (float)
        and index: prep_id (int)
    """
    # fix bool labling
    atr_key = {True: "ATR+", False: "ATR-"}
    data_to_plot = data.replace({"ATR status": atr_key})
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    sns.swarmplot(
        data=data_to_plot, x="ATR status",
        y="ΔF/F", hue="Age", ax=ax,
        palette={4:C_4d, 3:C_3d},
        size=size
    )
    ax.axhline(0, linestyle="dashed")
    # fix legend
    legend = ax.get_legend()
    if add_legend:
        legend.get_texts()[0].set_text("72 ALH")
        legend.get_texts()[1].set_text("96 ALH")
    else:
        legend.remove()
