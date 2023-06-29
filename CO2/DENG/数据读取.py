# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:12:12 2022

@author: MaYutong
"""

import netCDF4 as nc
import numpy as np
import xarray as xr

def read_nc1(inpath):
    global lat, lon, t
    with nc.Dataset(inpath, mode='r') as f:
        
        print(f.variables.keys(), "\n")
        print(f.variables['value'], "\n")
        print(f.variables['Times'], "\n")
        
        time = (f.variables['Times'][:])
        t = nc.num2date(time, 'days since 1850-01-01').data
        t = t[1572:]
        
        lat = (f.variables['Latitude'][:])
        lon = (f.variables['Longitude'][:])
        co2 = (f.variables['value'][1572:, :, :]) ##units: ppm; monthly mean
        
        return co2
    
# %%
def region1(data):
    pre_rg1global = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                               t, lat, lon])  # 原SPEI-base数据
    pre_rg1 = pre_rg1global.loc[:, 55:40, 100:125]

    return np.array(pre_rg1)        


#%%
inpath = r"E:/CO2/DENG/CO2_1deg_month_1850-2013.nc"
co2 = read_nc1(inpath)
co2_MG = region1(co2)