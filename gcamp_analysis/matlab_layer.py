"""
Gets objects from matlab
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import h5py

from .analysis import StimAvg, StimMeta, Traces

def load_stack(path: Path, name="registered") -> np.ndarray:
    """
    gets the image from h5 as a np array
    """
    with h5py.File(path, 'r') as f:
        shape = f[name].shape
        out = np.zeros(shape, dtype=np.int16)
        # out[:] = f[name][:]
        # to save memory, read and cast one frame at a time
        # this may be further optimized by reading ~100 frames at a time
        for i in range(shape[0]):
            out[i, :] = f[name][i, :]
        return out

def load_stim_avg(path: Path) -> Tuple[StimMeta, StimAvg, Traces]:
    """
    Loads stim_avg mat file and extracts data to form 
    the stim meta and stim avg
    returns (stim_meta, stim_avg, traces)
    """
    with h5py.File(path, 'r') as f:
        # get stim meta
        frame_time = f["Trace_Data"]["stim_meta"]["fs"][0][0]
        start_frame = int(
            f["Trace_Data"]["stim_meta"]["Start_Index"][0][0]
        )
        nchans = int(
            f["Trace_Data"]["stim_meta"]["nchans"][0][0]
        )
        n_iterations = int(
            f["Trace_Data"]["stim_meta"]["Iterations"][0][0]
        )
        i_stim_i = int(
            f["Trace_Data"]["stim_meta"]["Repetitions"][0][0]
        )
        stim_meta = StimMeta(
            start_frame=start_frame,
            i_stim_i=i_stim_i,
            n_iterations=n_iterations,
            frame_time=frame_time,
            n_chans=nchans
        )
        # get stim_avg
        pre_stim_df = np.array(f["Trace_Data"]["pre_stim_df"])
        post_stim_df = np.array(f["Trace_Data"]["post_stim_df"])
        stim_avg = StimAvg(
            pre_stim_df=pre_stim_df,
            post_stim_df=post_stim_df
        )

        # get traces
        traces_array = (f["Trace_Data"]["raw_traces"])
        if len(traces_array) == 1:
            traces = Traces(traces_array[0])
        elif len(traces_array) == 2:
            traces = Traces(
                chan0=traces_array[0],
                chan1=traces_array[1]
            )
        else:
            raise ValueError(">2 channels not supported")
        return stim_meta, stim_avg, traces
