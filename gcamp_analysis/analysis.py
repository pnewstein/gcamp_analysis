"""
Some classes for high level analysis
"""
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Union, List, Iterable
from pathlib import Path
import pickle

import numpy as np

from .image import load_imagej_tiff

PathLike = Union[str, os.PathLike]

PRE_SIM_FRAMES = 3

@dataclass
class StimAvg:
    """
    represents the stim avg for one experiment
    """
    pre_stim_df: np.ndarray
    "An array of prestim data with shape (n_stim, 3)"
    post_stim_df: np.ndarray
    "An array of post_stim data with shape (n_stim, time)"

    @property
    def pre_stim_mean(self) -> np.ndarray:
        "the mean of pre_stim_df as a 1d array"
        return np.mean(self.pre_stim_df, 0)

    @property
    def post_stim_mean(self) -> np.ndarray:
        "the mean of post_stim_df as a 1d array"
        return np.mean(self.post_stim_df, 0)
    
    @property
    def pre_stim_std(self) -> np.ndarray:
        "the mean of pre_stim_df as a 1d array"
        return np.std(self.pre_stim_df, 0)

    @property
    def post_stim_std(self) -> np.ndarray:
        "the mean of post_stim_df as a 1d array"
        return np.std(self.post_stim_df, 0)

@dataclass
class StimMeta:
    """
    The parameters of stimulus
    """
    start_frame: int
    i_stim_i: int
    n_iterations: int
    frame_time: float
    n_chans: int

    @property
    def stim_length(self) -> float:
        """
        the duration of the stimulus in seconds
        """
        return self.frame_time * self.n_iterations

# maybe this should be based on a list or a 2d array
@dataclass
class Traces:
    "contains raw fluorescence traces"
    chan0: Optional[np.ndarray] = None
    chan1: Optional[np.ndarray] = None

    def get_chan(self, chan_num: int):
        """
        Gets the array at that channel or 
        raises value error
        """
        chan_num = int(chan_num)
        trace = self.__getattribute__(f"chan{chan_num}")
        if trace is None:
            raise AttributeError(f"This prep does not have a channel {chan_num}")
        return trace

    def _set_chan(self, chan_num: int, trace: np.ndarray):
        """
        sets a trace
        """
        chan_num = int(chan_num)
        self.__setattr__(f"chan{chan_num}", trace)

    @property
    def n_chans(self) -> int:
        """
        the number of channels in this traces
        """
        n_chans = 0
        try:
            while True:
                _ = self.get_chan(n_chans)
                n_chans += 1
        except AttributeError:
            return n_chans
    
    @property
    def itertraces(self) -> Iterable[np.ndarray]:
        "A generator that iterates through all the traces"
        return (self.get_chan(i) for i in range(self.n_chans))

    def resize(self, final_sample):
        """
        Shortens all traces to final sample
        """
        if final_sample >= len(self.chan0):
            pass

        for chan_num, trace in enumerate(self.itertraces):
            self._set_chan(chan_num, trace[:final_sample])
        

    # this function should probably move to Experiment
    def get_stim_index(self, stim_meta: StimMeta, chan=0) -> np.ndarray:
        """
        The frame of each stimulation
        """
        trace = self.get_chan(chan) 
        stim_index = np.arange(stim_meta.start_frame, len(trace), stim_meta.i_stim_i)
        return stim_index

    # this function should probably move to Experiment
    def get_stim_f(self, stim_meta: StimMeta, chan=0, 
                   prestim_frames: Optional[int] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets the (time, fluorescence, dff) arrays with nans 

        Args:
        stim_meta (StimMeta): the metadata for stimulus
        chan (Optional[int]): the channel, defaults to 0
        prestim_frames (int): the number of frames to average for
            F0. defaults to 3

        Returns:
        The time array and fluorescence array with nans where data is lost
        from stimulus and the ΔF/F array
        """
        # handle default args
        if prestim_frames is None:
            prestim_frames = PRE_SIM_FRAMES
        trace = self.get_chan(chan) 
        # get number of frames lost to stimulus
        stim_index = self.get_stim_index(stim_meta=stim_meta, chan=chan)
        nstims = len(stim_index)
        n_stim_frames = nstims * stim_meta.n_iterations
        # get time
        t = np.arange(len(trace) + n_stim_frames) * stim_meta.frame_time
        assert len(t) == len(trace) + n_stim_frames
        # get f by pasting trace leaving nans
        f = np.empty(t.shape)
        f[:] = np.nan
        # also initialize df_f 
        df_f = np.copy(f)
        # prestim
        f[:stim_meta.start_frame] = trace[:stim_meta.start_frame]
        for stim_count, first_stim_frame in enumerate(stim_index):
            try:
                next_stim = stim_index[stim_count + 1]
            except IndexError:
                next_stim = len(trace) - 1
            total_nan_frames = (stim_count + 1) * stim_meta.n_iterations
            f_chunk = trace[first_stim_frame : next_stim] 

            f[
                first_stim_frame + total_nan_frames : 
                next_stim + total_nan_frames
            ] = f_chunk
            # get the last few frames before stimulation
            f_0 = np.mean(trace[first_stim_frame-1-prestim_frames: first_stim_frame-1]) 
            df_f[
                first_stim_frame + total_nan_frames : 
                next_stim + total_nan_frames
            ] = (f_chunk - f_0) / f_0
        return t, f, df_f

@dataclass(frozen=True)
class Experiment:
    "Class representing an calcium imaging experiment"
    experiment_name: str
    "the name of the experiment. usually file name without extention"
    stim_meta: StimMeta
    "the important metdata"
    traces_dict: Dict[str, Traces]
    "the traces for each roi"

    @classmethod
    def fromTif(cls, path: PathLike, stop_when_roi_is_lost=True):
        """
        Gets an experiment from a tif created from imagej that has
        at least one embedded roi and embedded metadata. ROI names 
        are conserved

        Args:
        path (PathLike): the path to the roi
        stop_when_roi_is_lost (bool): Whether to cut the trace short when
            there is no more fluorescence in the roi. Defaults to True.

        """
        path = Path(path)
        # get experiment name from filename
        undrifted_prefix = "UNDRIFTED"
        experiment_name_with_prefix = path.with_suffix("").name
        if experiment_name_with_prefix.startswith(undrifted_prefix):
            experiment_name = experiment_name_with_prefix[len(undrifted_prefix):]
        else:
            experiment_name = experiment_name_with_prefix
        # load the tiff
        data, masks, metadata = load_imagej_tiff(path)
        # get stim_meta
        stim_meta = StimMeta(
            start_frame=int(metadata["Experiment|AcquisitionBlock|BleachingSetup|StartIndex #1"]),
            i_stim_i=int(metadata["Experiment|AcquisitionBlock|BleachingSetup|Repetition #1"]),
            n_iterations=int(metadata["Experiment|AcquisitionBlock|BleachingSetup|Iterations #1"]),
            n_chans=len(data),
            frame_time=float(metadata["Information|Image|Channel|LaserScanInfo|FrameTime #1"])
        )
        # get traces
        traces_dict: Dict[str, Traces] = {}
        for roi_name, roi_mask in masks.items():
            # multiply across each channel and each frame
            masked_data = data * roi_mask
            traces = Traces()
            for chan_num, img3d in enumerate(masked_data):
                # sum the x and y axis
                trace = img3d.sum(axis=(1, 2))
                traces._set_chan(chan_num, trace) #pylint: disable=protected-access
                if stop_when_roi_is_lost:
                    # check for empty data
                    max_trace = np.max(img3d, axis=(1, 2))
                    # get the first 0 of median_trace
                    first_zero = np.argmax(np.logical_not(max_trace))
                    # if there is no zero in max_trace, then first_zero will be incorrectly set to 0
                    if first_zero == 0:
                        if max_trace[0] == 0:
                            raise ValueError(
                                f"The first data point in roi {roi_name} is 0. fix this ROI"
                            )
                    else:
                        # Roi is really lost
                        traces.resize(first_zero) #pylint: disable=W0212
                        
            traces_dict[roi_name] = traces
        # return the class
        return cls(
            experiment_name=experiment_name,
            stim_meta=stim_meta,
            traces_dict=traces_dict
        )

    @property
    def roi_names(self) -> List[str]:
        "The names of the rois"
        return list(self.traces_dict.keys())

    def save(self, path: PathLike):
        """
        Saves as a pickle to the path
        saves a small file containing only the metadata and 
        the traces as 1d arrays
        """
        path = Path(path)
        path.write_bytes(pickle.dumps(self))
        
    @classmethod
    def load(cls, path: PathLike) -> "Experiment":
        """
        loads an Experiment saved with save method
        """
        path = Path(path)
        return pickle.loads(path.read_bytes())

    def get_stim_avg(self, roi_name: Optional[str] = None, 
                     channel: Optional[int] = None,
                     prestim_frames: Optional[int] = None):
        """
        Gets a StimAvg object for a roi, given our stim_meta
        Args:
        roi_name (str) the name of the roi exactly as it appears in 
            .roi_names method. Defaults to the first roi alphabetically
        channel (int) the channel to analyze
        prestim_frames (int): the number of frames to average for
            F0. defaults to 3
        """
        # set default args
        if channel is None:
            channel = 0
        roi_names = self.roi_names
        roi_names.sort()
        if roi_name is None:
            roi_name = roi_names[0]
        elif roi_name not in roi_names:
            raise ValueError(
                f"{roi_name} is not an acceptable roi_name. "
                f"try one of {roi_names}"
            )
        if prestim_frames is None:
            prestim_frames = PRE_SIM_FRAMES
        traces = self.traces_dict[roi_name]
        trace = traces.get_chan(channel)
        # count the number of stim_index within traces minus 1 
        # (so it doesn't end in a stimulation)
        stim_meta = self.stim_meta
        stim_index = traces.get_stim_index(stim_meta=stim_meta, chan=channel)[:-1]
        "The indices of the first data point after stimulation"
        number_of_stimuli = sum(stim_index < len(trace) - 1)
        print(number_of_stimuli, len(stim_index))
        # initialize arrays
        prestim_dffs = np.zeros((number_of_stimuli, prestim_frames))
        post_stim_dffs = np.zeros((number_of_stimuli, stim_meta.i_stim_i))
        # make prestim and post stims sum
        for stim_ind, stim in enumerate(stim_index):
            prestim_f = trace[stim-1-prestim_frames : stim-1]
            post_stim_f = trace[stim : stim+stim_meta.i_stim_i]
            f0 = np.mean(prestim_f)
            # add this dff to dff
            prestim_dffs[stim_ind] = (prestim_f - f0) / f0
            post_stim_dffs[stim_ind] = (post_stim_f - f0) / f0
        # turn the dff arrays from sums to means
        return StimAvg(prestim_dffs, post_stim_dffs)

    def get_only_traces(self) -> Traces:
        """
        Gets the only traces object. Raises error if there is more than 
        one traces in the traces dict
        """
        if len(self.roi_names) != 1:
            raise ValueError("there are more than one traces!")
        return next(iter(self.traces_dict.values()))

