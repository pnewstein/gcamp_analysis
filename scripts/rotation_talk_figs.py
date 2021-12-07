"""
Some figures for my rotation talk
"""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from svgutils import transform as sg
import seaborn as sns


from gcamp_analysis.plot import df_f_figs
from gcamp_analysis.plot.scalebar import add_scalebar
from gcamp_analysis.stats import get_p_value, format_p_value

from bulk_process import (
    load_stim_avg_from_dir, OPTO_DIR, save_fig, IMGS_DIR, SUMMARY_CSV_PATH
    )


R_TALK_FIGS_DIR = IMGS_DIR / "r_talk"
FULL_SCREEN = (9, 4.6)
EGS = {"-":11, "+":17}
SMEARED = [
    8, 13
]

def get_stim_data():
    """
    gets stim_avg data filted as a pandas
    """
    # get raw dataframe from csv
    agg_stim_avgs = pd.read_csv(SUMMARY_CSV_PATH, index_col=0)
    # filter out the smeared preps (uncorrected drift)
    good_inds = set(agg_stim_avgs.index).difference(SMEARED)
    good_agg_stim_avgs = agg_stim_avgs[agg_stim_avgs.index.isin(good_inds)]
    return good_agg_stim_avgs

def nudge_avg_stim_fig():
    """
    Slightly nudges one of the plots of the avg stim fig
    """
    for full_sum in (True, False):
        path = R_TALK_FIGS_DIR / f"stim_avg{full_sum}.svg"
        fig = sg.fromfile(path)
        # moveto actually just moves relative to start
        fig.find_id("axes_1").moveto(28, 0)
        fig.save(path)

def eg_traces():
    """
    These are examples of atr+ and atr- 
    preps that look good. show raw fluorescence
    and dF/F examples with the same zoom and rectagles
    """
    sample = (514, 553)
    # {"+", (StimMeta, StimAvg, Traces), ...}
    frame_time = None
    stim_data = {
        k:load_stim_avg_from_dir(OPTO_DIR/str(v))
        for k, v in EGS.items()
    }
    # enumerate through fluorescence, df/f
    figs = [None, None]
    for fig_i, df_f in enumerate((False, True)):
        figs[fig_i], axs = plt.subplots(4, 2, tight_layout=True,
                                        gridspec_kw={"height_ratios": [.2, 1, .2, 1]})
        # get the correct complex axis sharing
        axs[0, 0].get_shared_x_axes().join(axs[0, 0], axs[1, 0])
        axs[0, 0].get_shared_x_axes().join(axs[0, 0], axs[0, 1])
        axs[0, 1].get_shared_x_axes().join(axs[0, 1], axs[1, 1])
        axs[2, 0].get_shared_x_axes().join(axs[2, 0], axs[3, 0])
        axs[2, 1].get_shared_x_axes().join(axs[2, 1], axs[3, 1])
        
        if df_f:
            axs[1, 1].get_shared_y_axes().join(axs[1, 1], axs[1, 0])
            axs[3, 1].get_shared_y_axes().join(axs[3, 1], axs[3, 0])
        # iterate by column
        for atr_i, (atr_status,
                    (stim_meta, _, traces)) in enumerate(stim_data.items()):
            # need local var fs
            if frame_time is None:
                frame_time = stim_meta.frame_time
            if atr_status == "+":
                stim_color = df_f_figs.C_ATR_PLUS
            else:
                stim_color = df_f_figs.C_ATR_MINUS
            # iterate by row pair
            for row in (1, 3):
                if df_f:
                    df_f_ax = axs[row][atr_i]
                    f_ax = None
                else:
                    df_f_ax = None
                    f_ax = axs[row][atr_i]
                # plots full trace on zoomed axis for now
                df_f_figs.plot_trace_on_ax(
                    axs[row-1][atr_i], f_ax, df_f_ax, traces=traces,
                    stim_meta=stim_meta, stim_color=stim_color
                )
        # make ylim go through zero
        axs[0, 0].set_xlim((-40, 922))



        # figure out sample rectangle
        y_ranges = [(None, None)] * 2
        for col in (0, 1):
            t, trace = axs[1, col].lines[0].get_data()
            s_range = [int(s/frame_time) for s in sample]
            # make sample fall on points
            sample = tuple(t[s_range])
            trace_in_sample = trace[s_range[0]: s_range[1]]
            y_ranges[col] = (np.nanmin(trace_in_sample), np.nanmax(trace_in_sample))

        # make sure d_df example has same y in both cols
        if df_f:
            y_ranges[0] = [
                (min(y_ranges[0][0], y_ranges[1][0])),
                (max(y_ranges[0][1], y_ranges[1][1]))
            ]
            y_ranges[1] = y_ranges[0]


        # apply sample rectangle
        for col in (0, 1):
            # draw rectangle
            rectangle = Rectangle(
                (sample[0], y_ranges[col][0]),
                (sample[1]-sample[0]),
                y_ranges[col][1] - y_ranges[col][0],
                edgecolor="w", facecolor="k"
            )
            axs[1, col].add_patch(rectangle)
            # zoom in
            axs[3, col].set_ylim(y_ranges[0])
            axs[3, col].set_xlim(sample)

        # add scalebars 
        for ax, x_kwargs in (
                (axs[1, 1], {"sizex":60, "labelx":"1 min"}),
                (axs[3, 1], {"sizex":5, "labelx":"5s"})
        ):
            if df_f:
                y_lim = ax.get_ylim()
                # ensure the tick is has exactly 1 decimal
                ytick = round((y_lim[1] - y_lim[0]) / .8) / 10
                y_kwargs = {"sizey": ytick, "labely": f"{ytick: .1f}",
                            "matchy": False}
            else:
                y_kwargs = {}
            add_scalebar(
                ax, matchx=False, hidey=False,
                **x_kwargs, **y_kwargs
            )
        # remove second col labels
        for ax in axs[:, 1]:
            ax.set_ylabel(None)
        # add titles
        axs[0, 0].set_title("-ATR")
        axs[0, 1].set_title("+ATR")

    for fig, name in zip(figs, ("f_sample_trace", "df_f_sample_trace")):
        fig.set_size_inches(FULL_SCREEN)
        save_fig(R_TALK_FIGS_DIR / name, fig)
        
    plt.show()

def stim_avgs(full_sum=False):
    "Makes a big plot showing mean sample"
    # load the sample stim_avgs
    # {"+", (StimMeta, StimAvg, Traces), ...}
    stim_data = {
        k:load_stim_avg_from_dir(OPTO_DIR/str(v))
        for k, v in EGS.items()
    }
    atr_status_key = {"+": True, "-": False}
    full_sum_ylim: Optional[Tuple[float]] = None
    for full_sum in (True, False):
        # plot the data
        fig, axs = plt.subplots(1, 3, tight_layout=True)
        # share ax on only the single prep data
        axs[0].get_shared_y_axes().join(axs[0], axs[1])
        axs[0].get_shared_x_axes().join(axs[0], axs[1])
        for col, (atr_status, 
                  (stim_meta, stim_avg, _)) in enumerate(stim_data.items()):
            df_f_figs.plot_stim_avg_on_ax(axs[col], stim_meta, 
                                          stim_avg, atr_status_key[atr_status])
        # format
        axs[0].set_title("-ATR")
        axs[1].set_title("+ATR")
        axs[1].set_yticklabels([])
        axs[1].set_ylabel(None)

        fig.set_size_inches(FULL_SCREEN)
        # plot summary
        summary_stim_data = get_stim_data()
        if not full_sum:
            summary_stim_data = summary_stim_data.loc[EGS.values()]
        df_f_figs.plot_stim_summary_on_axs(axs[2], summary_stim_data.copy(),
                                           size=7, add_legend=full_sum)
        if full_sum:
            # add significance annotation
            axs[2].text(0.5, 0.5, format_p_value(get_p_value(summary_stim_data)),
                        horizontalalignment='center',
                        verticalalignment='center', 
                        transform=axs[2].transAxes)
            full_sum_ylim = axs[2].get_ylim()
        else:
            axs[2].set_ylim(full_sum_ylim)

        axs[2].set_xlabel(None)

        save_fig(R_TALK_FIGS_DIR / f"stim_avg{full_sum}", fig)
    nudge_avg_stim_fig()

def swarm_plots():
    """
    The night before my rotation talk, I wanted to know if simulations responces are
    normally distributed within a prep. so I made some quick and dirty swarm plots
    """
    stim_data = {
        k:load_stim_avg_from_dir(OPTO_DIR/str(v))
        for k, v in EGS.items()
    }
    for atr_str, (_, stim_avg, _) in stim_data.items():
        dff = stim_avg.post_stim_df
        df = (pd.DataFrame(dff))
        if atr_str == "+":
            color = df_f_figs.C_ATR_PLUS
        else:
            color = df_f_figs.C_ATR_MINUS
        fig, ax = plt.subplots()
        ax.axhline(0, linestyle="dashed")
        sns.swarmplot(data=df, size=1, color=color, ax=ax)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_title(f"ATR{atr_str}")
        ax.set_ylabel("ΔF/F")
        ax.set_xlabel("post stimulus sample")
        ax.plot(dff.mean(0), color="c")
        fig.set_size_inches(FULL_SCREEN)
        save_fig(R_TALK_FIGS_DIR / f"stim_avg_swarm{atr_str}", fig)

    
def main():
    "make all the figures"
    with plt.style.context("dark_background"):
        eg_traces()
        stim_avgs()
        swarm_plots()
        plt.show()

if __name__ == '__main__':
    main()
