# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 11:29:41 2022

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
def tmp_xarray(band1):
    t = np.arange(40)
    mn = np.arange(1, 13, 1)
    tmp=xr.DataArray(band1,dims=['mn', 't', 'y','x'],coords=[mn, t, lat2, lon2])
    tmp_MG = tmp.loc[:, :, 40:55, 100:125]
    lat_MG = tmp_MG.y
    lon_MG = tmp_MG.x
    tmp_MG = np.array(tmp_MG)
    return tmp_MG, lat_MG, lon_MG

#%%
def read_nc1():
    global vap
    inpath = r"E:/CRU/VAP_DATA/vap_CRU_ORIGINAL_81_20.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        vap = (f.variables['vap'][:])
        
        vap = mn_yr(vap)

#%%
def read_nc2():
    global tmp, lat2, lon2
    inpath = r"E:/CRU/TMP_DATA/TMP_CRU_MONTH_81_20.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        tmp = (f.variables['tmp'][:]).data
     
#%% 师姐ERA5数据验证
def read_nc3():
    global vpd_ERA
    inpath = r"E:/ERA5-Land-monthly/GLOBAL VPD/01_ERA5-land-vpd_2000-2021-month_World-180_1.0_yuan.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        #print(f.variables['time'])
        print(f.variables['vpd'])
        
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        vpd_ERA = (f.variables['vpd'][:])
        time = (f.variables['time'][:])
     
read_nc3()
#%%计算饱和水汽压
def es(t):
    data = 6.11*np.exp(17.269*t/(237.3+t))
    ###### unit: hPa, Celsius degree
    return data


#%%
read_nc1()
read_nc2()
vap[vap==9.96921e+36]=np.nan
tmp[tmp==9.96921e+36]=np.nan
vap_s = es(tmp)
vpd = vap_s-vap

#%%生成新的nc文件
def CreatNC(data):
    new_NC = nc.Dataset(r"E:/CRU/VAP_DATA/vpd_CRU_MONTH_81_20.nc", 'w', format='NETCDF4')
    
    year = np.arange(1, 41, 1)
    month = np.arange(1, 13, 1)
    
    new_NC.createDimension('month', 12)
    new_NC.createDimension('year', 40)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)
    
    var=new_NC.createVariable('vpd', 'f', ("month","year","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['vpd'][:]=data
    new_NC.variables['lat'][:]=lat2
    new_NC.variables['lon'][:]=lon2
        
    var.units="hPa"
    var.Fillvalues="np.nan 缺测值在tmp vap时已经处理过"
    var.time="1981.1-2020.12"
    
    #最后记得关闭文件
    new_NC.close()
    
CreatNC(vpd)
#%%
import matplotlib.pyplot as plt
a = vpd[6, -1, :, :]
plt.figure(1, dpi=500)
plt.imshow(a, cmap='Reds', origin="lower", vmin=0, vmax=30)
plt.colorbar(shrink=0.75)
plt.title("CRU")
plt.show()


b = vpd_ERA[-2, 6, :, :]*10
plt.figure(1, dpi=500)
plt.imshow(b, cmap='Reds', origin="lower", vmin=0, vmax=30)
plt.colorbar(shrink=0.75)
plt.title("ERA5")
plt.show()