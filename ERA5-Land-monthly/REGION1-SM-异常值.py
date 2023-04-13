# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 09:51:20 2022

@author: MaYutong
"""

import cftime
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
import xarray as xr

#%%
def read_nc():
    global sm, lat, lon
    inpath = rf"E:/ERA5-Land-monthly/REGION1 SM/region1_SM_month_ORINGINAL.nc"
    with nc.Dataset(inpath) as f:
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        sm = (f.variables['sm'][:])
        
def read_nc2():
    global sm_ave
    inpath = rf"E:/ERA5-Land-monthly/REGION1 SM/region1_SM_month_ave.nc"
    with nc.Dataset(inpath) as f:

        sm_ave = (f.variables['sm'][:])
        
#%%
def CreatNC():
    new_NC = nc.Dataset(
        rf"E:/ERA5-Land-monthly/REGION1 SM/region1_SM_month_anomaly.nc", 
        'w', format='NETCDF4')
    
    new_NC.createDimension('time', 480)
    new_NC.createDimension('layer', 4)
    new_NC.createDimension('lat', len(lat))
    new_NC.createDimension('lon', len(lon))
    
    var = new_NC.createVariable('sm', 'f', ("time", "layer", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['sm'][:]=sm_anom
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
     
    var.description = "1981.1-2020.12 SM 每月总和，原数据为月平均"
    var.time = 'time = pd.date_range("1981-01-01", periods=480, freq="MS")'
    var.unit = "m**3 m**-3"
    var.origin = "存放: 原数据lat和数据相同，Xarray插值后数据翻了过来？？"
    
    new_NC.close()
        
#%%
read_nc()
read_nc2()

sm_anom = np.empty((480, 4, 151, 351))

for mn in range(12):
    # print("")
    i = mn
    for yr in range(40):
        sm_anom[i, :, :, :] = sm[i, :, :, :]-sm_ave[mn, :, :, :]
        i += 12

a = sm_ave[0, 0, :, :]
# CreatNC()
