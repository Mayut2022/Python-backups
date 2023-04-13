# %%
import netCDF4 as nc
import numpy as np
import os
import rasterio
import xarray as xr
from scipy.stats import linregress

# %%


def read_nc():
    global sif, lat, lon
    inpath = r"/mnt/e/Gosif_Monthly/data_Global/GOSIF_01_20_SPEI0.5X0.5.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        print(f.variables['sif'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        sif = (f.variables['sif'][:]).data

# %%
read_nc()
# %%
