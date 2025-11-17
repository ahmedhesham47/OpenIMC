# SPDX-License-Identifier: GPL-3.0-or-later
#
# Thin entry script to start the OpenIMC GUI for PyInstaller builds.

# ---------------------------------------------------------------------------
# STATIC IMPORTS FOR PYINSTALLER (NEVER EXECUTED AT RUNTIME)
# ---------------------------------------------------------------------------
# PyInstaller scans these imports statically. Because they appear in the code,
# PyInstaller includes these packages + their submodules in the frozen binary.
# The condition is always False, so at runtime this does *nothing*.
if False:  # pragma: no cover
    # Core scientific stack
    import numpy
    import pandas
    import matplotlib
    import seaborn
    import PIL
    import tifffile
    import scipy
    import statsmodels
    import h5py
    import lxml
    import imageio

    # UI
    from PyQt5 import QtWidgets, QtGui, QtCore

    # IMC I/O and datasets
    import readimc
    import imcdatasets

    # Segmentation / models
    import cellpose
    import cellSAM

    # Dimensionality reduction / clustering
    import umap
    import hdbscan
    import igraph
    import leidenalg

    # Batch correction
    import combat
    import harmonypy

    # Spatial / single-cell stack
    import squidpy
    import scanpy
    import anndata

    # ML / image stack
    import sklearn
    import skimage
    import torch
    import torchvision

    # Dask / xarray ecosystem
    import dask
    import dask.dataframe
    import xarray
    import xarray_schema

    # API clients
    import openai


# ---------------------------------------------------------------------------
# REAL ENTRY POINT
# ---------------------------------------------------------------------------
from openimc.__main__ import run_gui

if __name__ == "__main__":
    run_gui()
