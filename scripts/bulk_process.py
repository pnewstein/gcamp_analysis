"""
Contains code for bulk processing
"""
# ToDo: rewrite this code for moco processing
from pathlib import Path
from typing import List, Dict, Tuple
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from gcamp_analysis.matlab_layer import load_stim_avg, load_stack
from gcamp_analysis.analysis import StimAvg, StimMeta, Traces
from gcamp_analysis.image import save_projection
from gcamp_analysis.plot.df_f_figs import (
    plot_stim_avg, plot_trace_on_ax,
    C_ATR_MINUS, C_ATR_PLUS
)

OPTO_DIR = Path("/mnt/c/Users/peter/data/dbdA08a-optogenetics/")
IMGS_DIR = OPTO_DIR / "imgs"
SUMMARY_CSV_PATH = OPTO_DIR / 'data' / 'stim_avgs.csv'

def load_stim_avg_from_dir(prep_dir: Path)-> Tuple[StimMeta, StimAvg, Traces]:
    "looks for a file with pattern data/*stim_avg.mat in it and runs the script"
    path = [p for p in prep_dir.glob("data/*.mat") if "stim_avg" in p.name][0]
    return load_stim_avg(path)

def save_fig(path: Path, fig):
    "saves a low res png and a svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), transparent=True)
    fig.savefig(path.with_suffix(".png"), transparent=False, dpi=50)

def get_stim_avg_paths(opto_dir: Path) -> List[Path]:
    """
    Gets the stim avg paths
    """
    prep_dirs = [p for p in opto_dir.glob("*")
                 if p.is_dir() and p.name.isnumeric()]

    stim_avgs: List[Path] = []
    for prep_dir in prep_dirs:
        stim_avg = [
            p for p in (prep_dir / "data").glob("*")
            if "stim_avg" in p.name
        ]
        if len(stim_avg) == 0:
            warnings.warn(f"Missing stim_avg at {prep_dir}")
        else:
            stim_avgs.append(stim_avg[0])
    return stim_avgs

def prep_facts_from_names(paths: List[Path]) -> pd.DataFrame:
    """
    extracts as much information as possible
    from the names of the files
    """
    def get_atr_status(name: str) -> bool:
        "returns if atr positive"
        if "+ATR" in name: return True
        if "-ATR" in name: return False
        raise ValueError(f"Could not get ATR status in {name}")

    def age_from_name(name: str) -> int:
        "gets the age from the name"
        if "211115" in name or "211109" in name:
            return 3
        if "211112" in name:
            return 4
        raise ValueError(f"could not get age from {name}")

    prep_nums = [int(p.parents[1].name) for p in paths]
    atr_statuses = [get_atr_status(p.name) for p in paths]
    ages = [age_from_name(p.name) for p in paths]
    return pd.DataFrame({"ATR status": atr_statuses, "Age" :ages}, index=prep_nums)
    

# make this two functions
def get_all_avg_stim(opto_dir: Path) -> pd.DataFrame:
    """
    Saves all of the avg stims giving warnings for missing data
    returns a list of StimAvgs also plots all the data
    """
    # get data paths
    stim_avgs = get_stim_avg_paths(opto_dir)
    # get file names
    prep_facts = prep_facts_from_names(stim_avgs)
    atr_key = {False: "-", True: "+"}
    names = [
        "".join((str(prep_num), atr_key[atr_status], p.name))
        for (prep_num, (atr_status, _)), p
        in zip(prep_facts.iterrows(), stim_avgs)
    ]
    # plot and save all the figs
    stim_avgs_data: Dict[int, StimAvg] = {}
    for fig_path, data_path, (prep_num, (atr_status, _)) in zip(
            names, stim_avgs, prep_facts.iterrows()
    ):
        stim_meta, stim_avg, _ = load_stim_avg(data_path)
        stim_avgs_data[prep_num] = stim_avg
        fig, _ = plot_stim_avg(stim_meta, stim_avg,
                               atr_status=atr_status)
        fig.set_size_inches([6, 7])
        save_fig(IMGS_DIR / "stim_avg" / fig_path, fig)
    post_stim_dffs = pd.Series(
        {pid: sa.post_stim_mean[0] for pid, sa in stim_avgs_data.items()}
    )
    out = prep_facts.copy()
    out.insert(0, "ΔF/F", post_stim_dffs)
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SUMMARY_CSV_PATH)
    return out


def make_t_projections(opto_dir: Path):
    """
    Makes all of the time based projections
    """
    # get data paths
    prep_dirs = [p for p in opto_dir.glob("*")
                 if p.is_dir() and p.name.isnumeric()]

    registereds: List[Path] = []
    for prep_dir in prep_dirs:
        registered = [
            p for p in (prep_dir / "data").glob("*")
            if "registered" in p.name
        ]
        if len(registered) == 0:
            warnings.warn(f"Missing stim_avg at {prep_dir}")
        else:
            registereds.append(registered[0])
    # get file names
    prep_facts = prep_facts_from_names(registereds)
    atr_key = {False: "-", True: "+"}
    names = [
        "".join((str(prep_num), atr_key[atr_status], p.name))
        for (prep_num, (atr_status, _)), p
        in zip(prep_facts.iterrows(), registereds)
    ]
    for registered_path, name in zip(registereds, names):
        stack = load_stack(registered_path)
        # # x, y, z = 500, 256, 20
        # stack = np.random.normal(size=x*y*z).reshape(z, x, y)
        save_projection((IMGS_DIR / name).with_suffix(".png"), stack)
        
def save_all_sample_traces(opto_dir: Path):
    "saves all the sample traces so they are easy to audit"
    stim_avgs = get_stim_avg_paths(opto_dir)
    prep_facts = prep_facts_from_names(stim_avgs)
    traces = {
        pid: load_stim_avg(path)
        for path, (pid, _)
        in zip(stim_avgs, prep_facts.iterrows())
    }
    atr_key = {False: "-", True: "+"}
    names = [
        f"{prep_num}{atr_key[atr_status]}{age}dpf_{p.name}"
        for (prep_num, (atr_status, age)), p
        in zip(prep_facts.iterrows(), stim_avgs)
    ]
    # plot each one
    for name, (pid, (stim_meta, _, trace)) in zip(names, traces.items()):
        fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True)
        color_key = {True: C_ATR_PLUS, False: C_ATR_MINUS}
        color = color_key[prep_facts["ATR status"].loc[pid]]
        plot_trace_on_ax(axs[0], axs[1], axs[2], trace, stim_meta, stim_color=color)
        save_fig(IMGS_DIR / "traces" / name, fig)
        plt.close(fig)
    

def main():
    "run as a script"
    with plt.style.context("dark_background"):
        # make_t_projections(OPTO_DIR)
        get_all_avg_stim(OPTO_DIR)
        save_all_sample_traces(OPTO_DIR)

if __name__ == '__main__':
    main()
