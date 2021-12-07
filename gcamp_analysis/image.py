"""
Code to read and write images
"""
from pathlib import Path
import warnings
from typing import Dict, Tuple

from PIL import Image
import numpy as np
from aicsimageio import AICSImage
from roifile import roiread, ROI_TYPE
from tifffile import TiffFile

def save_projection(path: Path, stack: np.ndarray):
    """
    Takes a stack and makes a max projection on first axis
    """
    projection = stack.max(0)
    int_projection = (projection / np.max(projection) * 255).astype(np.int8)
    Image.fromarray(int_projection).convert("RGB").save(path)
    # OmeTiffWriter.save(projection, path, dim_order="YX")

def load_imagej_tiff(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, str]]:
    """
    Loads an imagej stack rois, and metadata
    returns:
        array with size CTYX
        Dict {
            Roi_name: roi mask array
        }
        metadata as a dict
    """
    im = AICSImage(path)
    rois = roiread(path)
    if len(rois) == 0:
        raise ValueError("could not load any ROIs")
    roi_masks: Dict[str, np.ndarray] = {}
    # accept both z stacks and T stacks
    if im.shape[0] > 1:
        stack_lbl = "T"
        load_kwargs = {"Z": 0}
    else:
        stack_lbl = "Z"
        load_kwargs = {"T": 0}
    xdim, ydim = im.shape[4], im.shape[3]
    for roi in rois:
        if roi.roitype != ROI_TYPE.RECT:
            warnings.warn("Only rectangle ROIs are currently supported. "
                          "using circumscribing rectangle")
        coords = list(roi.coordinates().astype(int))
        minx = min(c[0] for c in coords)
        maxx = max(c[0] for c in coords)
        miny = min(c[1] for c in coords)
        maxy = max(c[1] for c in coords)
        # create the array mask, which is 0 everywhere except the ROI where it is 1
        yind, xind = np.mgrid[:ydim, :xdim]
        mask = np.logical_and(
            np.logical_and(maxx > xind, xind > minx), 
            np.logical_and(maxy > yind, yind > miny)
        )
        roi_masks[roi.name] = mask
    # get metadata
    tf = TiffFile(path)
    with TiffFile(path) as tiff:
        info = tiff.imagej_metadata["Info"]

    metadata = {
        line.split(" = ")[0]: line.split(" = ")[1]
        for line in info.split("\n")[:-1]
    }
    # 
    return im.get_image_data(f"C{stack_lbl}YX", **load_kwargs), roi_masks, metadata
