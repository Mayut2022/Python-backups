# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 21:38:41 2022

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
    # global a1, a2, o1, o2
    inpath = rf"E:/ERA5-Land-monthly/REGION1 SM/region1_SM_month_ESALUC.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        sm = (f.variables['sm'][:])
        
read_nc()

#%% 多年月平均
def sm_month(mn, data):
    sm_mn = []
    ind = mn-1
    for i in range(40):
        #print(ind)
        sm_mn.append(data[ind, :, :, :])
        ind+=12
    sm_mn = np.array(sm_mn)
    return sm_mn

def sm_month_reverse(mn, data, data_z):
    ind = mn-1
    
    for i in range(40):
        data_z[ind, :, :] = data[i, :, :]
        ind+=12
    
    return data_z

#%%
def CreatNC3():
    new_NC = nc.Dataset(
        rf"E:/ERA5-Land-monthly/REGION1 SM/region1_SM_month_Z_Score_ESALUC.nc", 
        'w', format='NETCDF4')
    
    new_NC.createDimension('time', 480)
    new_NC.createDimension('layer', 4)
    new_NC.createDimension('lat', len(lat))
    new_NC.createDimension('lon', len(lon))
    
    var = new_NC.createVariable('sm', 'f', ("time", "layer", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['sm'][:]=data_z
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
     
    var.description = "1981.1-2020.12 SM 每月总和，原数据为月平均"
    var.time = 'time = pd.date_range("1981-01-01", periods=480, freq="MS")'
    var.unit = "m**3 m**-3"
    var.origin = "存放: 原数据lat和数据相同，Xarray插值后数据翻了过来？？"
    
    new_NC.close()

#%% Z-Score值
sm_mn_z = np.zeros((40, 4, 30, 70))
data_z = np.zeros((480, 4, 30, 70))
for mn in range(1, 13):
    sm_mn = sm_month(mn, sm)
    for r in range(30):
        for c in range(70):
            for l in range(4):
                sm_mn_z[:, l, r, c] = preprocessing.scale(sm_mn[:, l, r, c])
                data_z = sm_month_reverse(mn, sm_mn_z, data_z)

CreatNC3()