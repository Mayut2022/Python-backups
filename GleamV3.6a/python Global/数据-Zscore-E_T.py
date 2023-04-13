# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 20:01:48 2022

@author: MaYutong
"""
import cftime
import netCDF4 as nc
import numpy as np
from sklearn import preprocessing
import xarray as xr

import warnings
warnings.filterwarnings("ignore")

#%%
def read_nc():
    global lat, lon, t, e
    inpath = r'E:/Gleamv3.6a/v3.6a/global/E_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5.nc'
    with nc.Dataset(inpath, mode='r') as f:
        '''
        print(f.variables.keys())
        print(f.variables['time'])
        print(f.variables['lat'])
        print(f.variables['lon'])
        print(f.variables['E'])
        '''
        time = (f.variables['time'][:])
        t = nc.num2date(time, 'days since 1980-01-31 00:00:00').data
        
        e = (f.variables['E'][:, :, :])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])

#%%
def e_month(mn, data):
    e_mn = []
    ind = mn-1
    for i in range(40):
        #print(ind)
        e_mn.append(data[ind, :, :])
        ind+=12
    e_mn = np.array(e_mn)
    return e_mn

def e_month_reverse(mn, data, data_z):
    ind = mn-1
    
    for i in range(40):
        data_z[ind, :, :] = data[i, :, :]
        ind+=12
    
    return data_z

#%%生成新的nc文件
def CreatNC(data, mn):
    new_NC = nc.Dataset(
        rf'E:/Gleamv3.6a/v3.6a/global/E_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5_Zscore_month{mn}.nc', 
        'w', format='NETCDF4')
    
    year = np.arange(1981, 2021, 1)
    
    new_NC.createDimension('time', 40)
    new_NC.createDimension('lat', len(lat))
    new_NC.createDimension('lon', len(lon))
    
    var = new_NC.createVariable('E', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('time', 'f', ("time"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['E'][:]=data
    new_NC.variables['time'][:]=year
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
     
    var.description = "1981.1-2020.12 E (actual e) 仅linear插值，其他未处理的标准化距平"
    var.time = "days since 1980-01-31 00:00:00"
    
    new_NC.close()


#%%
read_nc()
#%% Z-Score值
def Zscore(data, mn):
    e_mn_z = np.zeros((40, 360, 720))
    e_mn = e_month(mn, data)
    for r in range(360):
        if r%10 == 0:
            print(f"month{mn} {r} is done!")
        for c in range(720):
            e_mn_z[:, r, c] = preprocessing.scale(e_mn[:, r, c])
                
    return e_mn_z
#%%
for mn in range(1, 13):
    e_z = Zscore(e, mn)
    CreatNC(e_z, mn)