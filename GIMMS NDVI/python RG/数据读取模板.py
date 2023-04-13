# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:03:34 2022

@author: MaYutong
"""

import netCDF4 as nc
import numpy as np
import os
import rasterio
import xarray as xr
from scipy.stats import linregress

#%%
def read_nc(inpath):
    global ndvi, lat, lon, time
    
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        print(f.variables['time'], "\n")
        print(f.variables['ndvi'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        ndvi = (f.variables['ndvi'][:])
        time = (f.variables['time'][:])
    
        
#%%
inpath = r"E:/GIMMS_NDVI/data/ndvi3g_geo_v1_1981_0712.nc4"
read_nc(inpath)
ndvi2 = ndvi.reshape(6, 2, 2160, 4320).mean(axis=1)

test1 = ndvi[0, :, :].reshape(1, 2160, 4320)
test2 = ndvi[1, :, :].reshape(1, 2160, 4320)
test = np.vstack((test1, test2)).mean(axis=0)

#%%
import matplotlib.pyplot as plt

plt.figure(3, dpi=500)
plt.imshow(test, cmap='Set3', vmin=0, vmax=10000)
plt.colorbar(shrink=0.75)
plt.show()

for i in range(5):
    plt.figure(3, dpi=500)
    plt.imshow(ndvi2[i, :, :], cmap='Set3', vmin=0, vmax=10000)
    plt.colorbar(shrink=0.75)
    plt.savefig(f"{i}.jpg")
    plt.show()