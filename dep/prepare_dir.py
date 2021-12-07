"""
contains code to prepare a directory for analysis
"""
from pathlib import Path
import shutil

import numpy as np
from aicsimageio import AICSImage 
from aicsimageio.writers import OmeTiffWriter


def reduce_file_stack(path: Path, size: int) -> np.ndarray:
    """
    reads a czi file and reduces dimentions number of slices yeilding
    a 3D TYX np array
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    img = AICSImage(path)
    array = img.get_image_dask_data("TYX", C=0, Z=0)
    return array[:size, :, :]

def get_metadata(dir_path: Path):
    "returns the metadata object for the file"
    czis = [p for p in dir_path.glob("*.czi") if "snap" not in p.name]
    if len(czis) == 0: raise FileNotFoundError(f"no czi files in {dir_path}")
    if len(czis) != 1: raise FileExistsError(f"there are more than one czi files in {dir_path}")
    img = AICSImage(czis[0])
    return img.metadata

def shrink_czi_to_tif(dir_path: Path, size: int):
    """
    Finds a czi file in a path and adds a shrinked version
    of that file to the path with the name raw.tif
    Args:
    dir_path: the path to the directory with the czi
    """
    czis = [p for p in dir_path.glob("*.czi") if "snap" not in p.name]
    if len(czis) == 0: raise FileNotFoundError(f"no czi files in {dir_path}")
    if len(czis) != 1: raise FileExistsError(f"there are more than one czi files in {dir_path}")
    shrunk = reduce_file_stack(czis[0], size)
    OmeTiffWriter.save(shrunk, dir_path / "shrunk.tiff", dim_order="TYX")


def sort_files(parent: Path):
    """
    Sort files into folders
    """
    czis = [p for p in parent.glob("*.czi") if "snap" not in str(p)]
    def get_nex_dir() -> str:
        current_dirs = [int(p.name) for p in parent.glob("*") if p.is_dir()]
        return str(max(current_dirs) + 1)
    for czi in czis:
        next_dir = parent / get_nex_dir()
        next_dir.mkdir()
        next_path = next_dir / czi.name
        shutil.move(czi, next_path) 

def fix_file(parent: Path):
    """
    i made a mistake
    """
    for i in range(7, 20):
        path = parent / str(i)
        assert path.exists
        dst = parent / str(i-2)
        assert not dst.exists(), f"{dst}"
        shutil.move(path, dst)

def fill_files(parent: Path):
    "fill the files with subdirs"
    dirs = [p for p in parent.glob("*") if p.is_dir()]
    subdir_names = ["imgs", "data", "avg-results"]
    for path in dirs:
        subdirs = [path / n for n in subdir_names]
        for subdir in subdirs:
            if not subdir.exists():
                subdir.mkdir()
        print(path)
        shrink_czi_to_tif(path, 5000)


        

if __name__ == '__main__':
    dir_path = Path(r"/mnt/c/Users/peter/data/dbdA08a-optogenetics/4/")
    shrink_czi_to_tif(dir_path, 5000)
    # md = get_metadata(Path("/mnt/c/Users/peter/data/dbdA08a-optogenetics/"))
    # fill_files(Path("/mnt/c/Users/peter/data/dbdA08a-optogenetics/"))


    # small = reduce_file_stack(r"/mnt/c/Users/peter/data/dbdA08a-optogenetics/4/211112LexA_dbdGal4_lacZ_LexAopGCaMP6m_UASChrim_L2_Animal2_-ATR.czi", 1000)
    # OmeTiffWriter.save(small, "test.tiff", dim_order="TYX")
    
