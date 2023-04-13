# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 13:18:32 2022

@author: MaYutong
"""

import datetime as dt
import netCDF4 as nc
from matplotlib import cm
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr

#%% 480 -> 月 年
def mn_yr(data):
    q_mn = []
    for mn in range(12):
        q_ = []
        for yr in range(40):
            q_.append(data[mn])
            mn += 12
        q_mn.append(q_)
            
    q_mn = np.array(q_mn)
    return q_mn

#%%
def vpd_xarray(band1):
    t = np.arange(40)
    mn = np.arange(1, 13, 1)
    vpd=xr.DataArray(band1,dims=['mn', 't', 'y','x'],coords=[mn, t, lat, lon])
    vpd_MG = vpd.loc[:, :, 40:55, 100:125]
    lat_MG = vpd_MG.y
    lon_MG = vpd_MG.x
    vpd_MG = np.array(vpd_MG)
    return vpd_MG, lat_MG, lon_MG

def vpd_xarray2(band1):
    t = np.arange(22)
    mn = np.arange(1, 13, 1)
    vpd=xr.DataArray(band1,dims=['t', 'mn', 'y','x'],coords=[t, mn, lat2, lon2])
    vpd_MG = vpd.loc[:, :, 40:55, 100:125]
    lat_MG = vpd_MG.y
    lon_MG = vpd_MG.x
    vpd_MG = np.array(vpd_MG)
    vpd_ESA = vpd.interp(t=t, mn=mn, y=lat, x=lon, method="linear")
    
    return vpd_MG, lat_MG, lon_MG, np.array(vpd_ESA)

#%%
def read_nc1():
    global vpd, lat, lon
    inpath = r"E:/CRU/VAP_DATA/vpd_CRU_MONTH_81_20.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        lat = (f.variables['lat'][:]).data
        lon = (f.variables['lon'][:]).data
        vpd = (f.variables['vpd'][:])
        

     
#%% 师姐ERA5数据验证
def read_nc3():
    global vpd_E, lat2, lon2
    inpath = r"E:/ERA5-Land-monthly/GLOBAL VPD/01_ERA5-land-vpd_2000-2021-month_World-180_1.0_yuan.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        #print(f.variables['time'])
        #print(f.variables['vpd'])
        
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        vpd_E = (f.variables['vpd'][:])*10
        time = (f.variables['time'][:])
     

#%%
read_nc1()
read_nc3()

vpd_MG, _, _ = vpd_xarray(vpd)
vpd_E_MG, _, _, vpd_E_G = vpd_xarray2(vpd_E)
#%%

import matplotlib.pyplot as plt
a = vpd_MG[6, 19:, :, :].mean(axis=0)

plt.figure(1, dpi=500)
plt.imshow(a, cmap='Reds', origin="lower", vmin=0, vmax=30)
plt.colorbar(shrink=0.75)
plt.title("CRU")
plt.show()


b = vpd_E_MG[:-1, 6, :, :].mean(axis=0)

plt.figure(1, dpi=500)
plt.imshow(b, cmap='Reds', origin="lower", vmin=0, vmax=30)
plt.colorbar(shrink=0.75)
plt.title("ERA5")
plt.show()

#%% 年际变率对比
a2 = vpd_MG[6, 19:, :, :]; a2=np.nanmean(a2, axis=(1, 2))
b2 = vpd_E_MG[:-1, 6, :, :]; b2=np.nanmean(b2, axis=(1, 2))

t = np.arange(2000, 2021, 1)
plt.figure(1, dpi=500)
plt.plot(t, a2, "r", label="CRU")
plt.plot(t, b2, "b", label="ERA5")
plt.legend()
plt.title("MG ZONE Comparision")
plt.show()

#%% 全球相关对比