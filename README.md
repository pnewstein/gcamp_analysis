# GCaMP analysis
Some code that analyzes calcium imaging data.

This branch is for use with the imagej plugin moco. The main branch is for use with imagej

# Installation
## Dependencies
- [Fiji](https://imagej.net/software/fiji/downloads) (it's just image j)
- Install [python](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/installation/)

### Installing moco
Unfortunately, moco is not as straight forward to install. Follow the instructions on https://github.com/NTCColumbia/moco .
- clone or otherwise download the repo
- navigate to ```moco/03-18-2016_release/jars/```
- paste all the ```.jar``` files into your imagej plugins directory. On my system, this directory is located at ```C:\Users\peter\bin\Fiji.app\jars```
- Launch imagej
- drag the file ```moco/03-18-2016_release/moco_.jar``` to imagej
- imagej should install the plugin then ask to restart
- after the restart, you should be able to find moco in the plugins menu, or by using the imagej search bar


## Installing gcamp_analysis with pip
### Download through the command line
- clone the repo (this will copy the files into a new folder called "gcamp_analysis"): 
```
git clone https://github.com/pnewstein/gcamp_analysis
```
- enter the repo: 
```
cd gcamp_analysis
```
### Or you can just use the GitHub gui to do the same
### Install with pip
- install *GCaMP analysis* with python dependencies
```
python -m pip install -e .
```
to test if it worked, try running the python code:
```python
import gcamp_analysis
```
if you don't get ```ModuleNotFoundError```, the install was successful!

# Usage
## MotionCorrecting with imagej
- First, load the data into imagej (if your files are big, this could take a few seconds)
- drag the file [one_channel_moco_wrapper.ijm](imagej_macros/one_channel_moco_wrapper.ijm) onto imagej. This should open an editor where you can also run the macro
- click run
- in a couple of minutes, you should see a drift corrected stack as well as a maximum projection
- Use the rectangle tool to draw a roi
- save the undrifted file to a directory containing all of your undrifted files

## Analysis with Python
Now you can use this python repo to analyze the data and make figures.

To get additional information on how to use any of these functions, use ```help``` in an interactive python session. For example:
``` 
help(help(gcamp_analysis.analysis.Experiment))
```
### Import gcamp_analysis into python
```python
import gcamp_analysis
```
### Make an Experiment object from the tiff file
```python
registered_path = Path(r"C:\Users\peter\data\dbdA08a-optogenetics\1\data\211109LexA_dbdGal4_lacZ_LexAopGCaMP6m_UASChrim_L2_Animal1_-ATR_registered.mat")
# the following line requires lot of ram
stack = gcamp_analysis.matlab_layer.load_stack(registered_path)
gcamp_analysis.save_projection(Path("projection.png"), stack)
```
### Load the data from a stim_avg
```python
from pathlib import Path
stim_avg_path = Path(r"C:\Users\peter\data\dbdA08a-optogenetics\1\data\211109LexA_dbdGal4_lacZ_LexAopGCaMP6m_UASChrim_L2_Animal1_-ATR_stim_avg.mat")
stim_meta, stim_avg, traces = gcamp_analysis.matlab_layer.load_stim_avg(stim_avg_path)
```
### Make the traces figure
```python
import matplotlib
matplotlib.style.use("dark_background")
from matplotlib import pyplot as plt
from gcamp_analysis.plot import df_f_figs
# make the figure and axes in advance
fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True)
# plot all traces and stimulus on axs
color = df_f_figs.C_ATR_MINUS
df_f_figs.plot_trace_on_ax(
    axs[0], axs[1], axs[2], traces, stim_meta, stim_color=color
)
# show the figure
fig.show()
```
### Make the stim average figure
```python
# make the figure and axes in advance
fig, ax = plt.subplots()
# plot the figure
df_f_figs.plot_stim_avg_on_ax(ax, stim_meta, stim_avg)
fig.show()
```
