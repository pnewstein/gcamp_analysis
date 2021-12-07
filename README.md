# GCaMP analysis
Some code that analyzes calcium imaging data.

This branch is for use with matlab. The main branch is for use with imagej

One thing this code has to offer, is an easier way to programmatically combine figures using matplotlib. Also, this code has a way to show traces with the correct gaps for missing data during the stimuli. This code also takes less user interaction. This makes it easier to use with large files.

# Installation
## Dependencies
- Instal MatLab and find the code written by Brandon Mark plus my edits.
- Make sure you have [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- Install [python](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/installation/)

## Installing with pip
- clone the repo (this will copy the files into a new folder called "gcamp_analysis"): 
```
git clone https://github.com/pnewstein/gcamp_analysis
```
- enter the repo: 
```
cd gcamp_analysis
```
- switch to this branch:
```
git checkout matlab
```
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
## Preparation
- First, use imagej or some other tool to convert the CZI to a tiff containing the frames you want to analyze. *This step may not be necessary with minor changes to the matlab code.* You can use use "show info" command to get all of the metadata you need to create ```stim.json``` which should look something like this 
```json
{
    "startIndex": 15,
    "Iterations": 3,
    "Repetition": 30
}
```
- Run the matlab registration algorithm and save the files as a .mat h5 file. This takes a very long time, so I did this step in bulk with the command 
```
prepare_bulk("C:\Users\peter\data\dbdA08a-optogenetics\", [1:3])
```
Assuming the structure of ```C:\Users\peter\data\dbdA08a-optogenetics\``` is something like
```
├── 1
│   ├── 211109LexA_dbdGal4_lacZ_LexAopGCaMP6m_UASChrim_L2_Animal1_-ATR.czi
│   ├── data
│   ├── imgs
│   ├── shrunk.tiff
│   └── stim.json
├── 2
│   ├── 211109LexA_dbdGal4_lacZ_LexAopGCaMP6m_UASChrim_L2_Animal2_+ATR.czi
│   ├── data
│   ├── imgs
│   ├── shrunk.tiff
│   └── stim.json
├── 3
│   ├── 211112LexA_dbdGal4_lacZ_LexAopGCaMP6m_UASChrim_L2_Animal1_-ATR.czi
│   ├── data
│   ├── imgs
│   ├── shrunk.tiff
│   └── stim.json
```
- Use matlab to create the ROIs. This does require user interaction, so I did this one at a time with the matlab command
```
fast_calcium_imaging('C:\Users\peter\data\dbdA08a-optogenetics\1\')
```
## Analysis
Now you can use this python repo to analyze the data and make figures.
### Import gcamp_analysis into python
```python
import gcamp_analysis
```
### Make the maximum intensity projection of the drift corrected stack
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