from pathlib import Path
from dataclasses import dataclass

from skimage import io
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class StimParams:
    """
    The parameters of stimulus
    """
    start_frame: int
    period: int
    n_iterations: int

def average(full: np.ndarray, stim_params: StimParams, 
            start: int=None, stop: int=None) -> np.ndarray:
    """
    Averages the image
    Args:
    full (np.ndarray) the image as a 4d array
    start (int) the first frame to start averageing
    period (int) the number of frames to average over
    stop (int) the last frame to study

    returns the average as a 4d array shortened to period
    """
    if stop is None: stop = full.shape[0]
    if start is None: start = stim_params.start_frame
    period = stim_params.period
    out = np.zeros((period, *full.shape[1:]), np.uint16)
    niters = (stop - start) // period
    print(f"{niters=}")
    last_frame = start + (period * niters)
    for i, frame in enumerate(full[start:last_frame]):
        out[i%period] += frame
    out = out / niters
    return out

def recolor_array(in_arr: np.ndarray) -> np.ndarray:
    """
    recolors array to a color map
    """

def pixel_sum(time_series: np.ndarray) -> np.ndarray:
    """
    turns an image stack into a 1D array of pixel sums
    """
    out = np.zeros((time_series.shape[0]), dtype=np.int64)
    # catch overflow error
    assert np.amax(time_series)*time_series.shape[0] < 2**64
    for i, frame in enumerate(time_series):
        out[i] = np.sum(frame)
    return out


def plot_trace_on_ax(stim_ax, trace_ax, trace: np.ndarray, stim_params: StimParams):
    """
    Plots a nice sample trace on axs
    """
    trace_ax.plot(trace, color="c")
    stim = np.arange(stim_params.start_frame, len(trace), stim_params.period)
    for x in stim:
        stim_ax.axvline(x=x, color="m")

    stim_ax.xaxis.set_visible(False)
    for ax in stim_ax, trace_ax:
        ax.xaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
    stim_ax.set_ylabel("Stimulus")
    trace_ax.set_ylabel("Fluorescence")

def plot_trace(trace, stim_params):
    """
    plots the trace, returns fig and axs
    """
    fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True,
                            gridspec_kw={"height_ratios": [.2, 1]})
    plot_trace_on_ax(axs[0], axs[1], trace=trace, stim_params=stim_params)
    fig.set_size_inches(6, 2)
    return fig, axs

def main():
    "Runs the file"
    # style pyplot
    plt.style.use("dark_background")
    dir = Path(r"/mnt/c/Users/peter/data/dbdA08a-optogenetics/1")
    pin = dir / "raw.tif"
    try:
        series = np.load(dir/"intensity.npy")
    except FileNotFoundError:
        img = io.imread(pin, plugin="tifffile")
        series = pixel_sum(img)
        np.save(dir/"intensity", series)

    fig0, axs = plot_trace(series, StimParams(150, 30, 3))
    
    plt.show()

    # smooth = average(img, 15+66-15, 30)
    # io.imsave(pin.parent/ "out.tif", smooth, plugin="tifffile")

if __name__ == '__main__':
    main()
    
