"""Installs package and all dependencies"""
from setuptools import setup, find_packages


setup(
    name="gcamp_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=["h5py", "roifile", 
                      "tifffile", "pillow", 
                      "scipy", "matplotlib",
                      "numpy", "pandas", "seaborn",
                      "svgutils", "aicsimageio"]
)
