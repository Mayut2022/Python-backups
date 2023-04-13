# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:18:55 2022

@author: MaYutong
"""

import netCDF4 as nc
import numpy as np
import os
import rasterio
import xarray as xr
from scipy.stats import linregress

#%%
def read_nc():
    global sif, lat, lon
    inpath = r"/mnt/e/Gosif_Monthly/data_RG/GOSIF_01_20_RG.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        print(f.variables['sif'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        sif = (f.variables['sif'][:])
        
        sif = sif*0.0001
        sif[sif>3] = np.nan


#%%
def sif_xarray(band1):
    lat_spei = np.linspace(35.25, 59.75, 50)
    lon_spei = np.linspace(100.25, 149.75, 100)
    yr = np.arange(20)
    mn = np.arange(12)
    sif=xr.DataArray(band1, dims=['yr', 'mn', 'y','x'],coords=[yr, mn, lat, lon])
    
    sif_ESA = sif.interp(yr=yr, mn=mn, y=lat_spei, x=lon_spei, method="linear")
    lat_ESA = sif_ESA.y
    lon_ESA = sif_ESA.x
    
    return sif_ESA, lat_ESA, lon_ESA

#%%        
read_nc()
sif_ESA, lat_ESA, lon_ESA = sif_xarray(sif)

#%%生成新的nc文件
def CreatNC(data):
    year = np.arange(2001, 2021, 1)
    month = np.arange(1, 13, 1)
    
    new_NC = nc.Dataset(r"E:/Gosif_Monthly/data_RG/GOSIF_01_20_RG_SPEI0.5X0.5.nc", 'w', format='NETCDF4')
    
    new_NC.createDimension('year', 20)
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 50)
    new_NC.createDimension('lon', 100)
    
    var=new_NC.createVariable('sif', 'f', ("year", "month","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['sif'][:]=data
    new_NC.variables['lat'][:]=lat_ESA
    new_NC.variables['lon'][:]=lon_ESA
        
    var.sif= ("包含此前尝试的三个区域，region1 ESA IM")
    var.lat_range="[60, 35], 500, 精度：0.05, 边界：[59.975, 35.025]"
    var.lon_range="[100, 150], 1001, 精度：0.05, 边界：[100, 150]"
    var.data="scale factor和缺测值均已处理，乘过0.0001"
    
    #最后记得关闭文件
    new_NC.close()            


CreatNC(sif_ESA)
# %%
'''
import matplotlib.pyplot as plt

plt.figure(3, dpi=500)
plt.imshow(sif_ESA[0, 6, :, :], cmap='Set3')
plt.colorbar(shrink=0.75)
plt.show()
'''


