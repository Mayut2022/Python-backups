# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:24:20 2022

@author: MaYutong
"""

import cftime
import netCDF4 as nc
import numpy as np

import xarray as xr
from scipy.stats import pearsonr
#%%
def read_nc(inpath):
    global lat, lon, e
    with nc.Dataset(inpath, mode='r') as f:
        
        '''
        print(f.variables.keys())
        print(f.variables['time'])
        print(f.variables['lat'])
        print(f.variables['lon'])
        print(f.variables['E'])
        '''
        t = (f.variables['time'][:])
        
        e = (f.variables['E'][:, :, :])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        
#%%
def read_nc2():
    global spei
    inpath = (r"E:/SPEI_base/data/spei03.nc")
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        spei = (f.variables['spei'][960:])
        time = (f.variables['time'][960:])
        t2 = nc.num2date(time, 'days since 1900-01-01 00:00:0.0').data
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

#%%
def spei_month(mn, data):
    e_mn = []
    ind = mn-1
    for i in range(40):
        #print(ind)
        e_mn.append(data[ind, :, :])
        ind+=12
    e_mn = np.array(e_mn)
    return e_mn

#%%
def corr(data1, data2):
    r,p = np.zeros((360, 720)), np.zeros((360, 720))
    
    for i in range(len(lat)):
        if i%30 == 0:
            print(f"column {i} is done!")
        for j in range(len(lon)):
            a, b = data1[:, i, j], data2[:, i, j]
            if np.isnan(a).any() or np.isnan(b).any():
                r[i,j], p[i,j] = np.nan, np.nan
                
            else:
                r[i,j], p[i,j]  = pearsonr(a, b)
                
    return r, p

#%%生成新的nc文件
def CreatNC(data1, data2, mn):
    new_NC = nc.Dataset(rf'E:/Gleamv3.6a/v3.6a/global/Corr/SPEI_E_81_20_month{mn}.nc', 
        'w', format='NETCDF4')
    
    new_NC.createDimension('lat', len(lat))
    new_NC.createDimension('lon', len(lon))
    
    var = new_NC.createVariable('r', 'f', ("lat", "lon"))
    new_NC.createVariable('p', 'f', ("lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['r'][:]=data1
    new_NC.variables['p'][:]=data2
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
     
    var.description = "1981.1-2020.12 E (actual e) 仅linear插值，其他未处理的标准化距平"
    var.time = f"month(mn) SPEI-E Pearson相关"
    
    new_NC.close()
    
#%% 逐月相关

read_nc2()
'''
for mn in range(1, 13):
    inpath = rf"E:/Gleamv3.6a/v3.6a/global/Zscore/E_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5_Zscore_month{mn}.nc"
    read_nc(inpath)
    spei_mn = spei_month(mn, spei)
    r, p = corr(spei_mn, e)
    CreatNC(r, p, mn)
    '''
#%% 验证数据存放正反 matplotlib
'''
import matplotlib.pyplot as plt

a = e[0, :, :]
b = spei[0, :, :]

plt.figure(1, dpi=500)
plt.imshow(b, cmap='Blues')
plt.colorbar(shrink=0.75)
plt.show()
'''
#%%生成新的nc文件
def CreatNC2(data1, data2, mn):
    new_NC = nc.Dataset(rf'E:/Gleamv3.6a/v3.6a/global/Corr/SPEI_E_81_20_allmonth.nc', 
        'w', format='NETCDF4')
    
    new_NC.createDimension('lat', len(lat))
    new_NC.createDimension('lon', len(lon))
    
    var = new_NC.createVariable('r', 'f', ("lat", "lon"))
    new_NC.createVariable('p', 'f', ("lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['r'][:]=data1
    new_NC.variables['p'][:]=data2
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
     
    var.description = "1981.1-2020.12 E (actual e) 仅linear插值，其他未处理的标准化距平"
    var.time = f"480个月的相关 SPEI-E Pearson相关"
    
    new_NC.close()
#%% 所有月份的相关
for mn in range(1, 13):
    inpath = rf"E:/Gleamv3.6a/v3.6a/global/Zscore/E_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5_Zscore_month{mn}.nc"
    read_nc(inpath)
    if mn == 1:
        e_all = e
    else:
        e_all = np.vstack((e_all, e))
        
r, p = corr(spei_mn, e)
CreatNC2(r, p, mn)