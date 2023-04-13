# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:06:28 2022

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

def read_nc2():
    global sm, lat, lon, t, time
    inpath = (r"E:/ERA5-Land-monthly/REGION1 SM/adaptor.mars.internal-1663809448.0435355-26012-17-a84c7d99-b8d7-464f-86f3-c65dd79b0d43.nc")
    with nc.Dataset(inpath) as f:
        print(f.variables.keys())
        print(f.variables['time'], "\n")
        print(f.variables['swvl1']) # units: m**3 m**-3 monthly averaged
        lat = (f.variables['latitude'][:])
        lon = (f.variables['longitude'][:])
        for i in range(1, 5):
            exec(f"sm{i} = (f.variables['swvl{i}'][:])")
            # exec(f"sm{i}.reshape(480, 1, 151, 351)")
            # exec(f"print(sm{i}.shape)")
        sm = np.vstack((sm1, sm2, sm3, sm4))
        
        time = (f.variables['time'][:])
        t = nc.num2date(time, 'hours since 1900-01-01 00:00:00.0').data

#read_nc2()


#%%
'''
a = sm[0, 0, :, :]
import matplotlib.pyplot as plt
plt.figure(1, dpi=500)
plt.imshow(a, cmap='Blues')
plt.colorbar(shrink=0.75)
plt.show()
'''

#%%
df = pd.read_excel('E:/ERA5/每月天数.xlsx')
mn_str = df['月份']
mn_num = df['平年1']


#%%
inpath = (r"E:/ERA5-Land-monthly/REGION1 SM/adaptor.mars.internal-1663809448.0435355-26012-17-a84c7d99-b8d7-464f-86f3-c65dd79b0d43.nc")
with nc.Dataset(inpath) as f:
    print(f.variables.keys())
    print(f.variables['time'], "\n")
    print(f.variables['swvl1']) # units: m**3 m**-3 monthly averaged
    lat = (f.variables['latitude'][:])
    lon = (f.variables['longitude'][:])
    time = (f.variables['time'][:])
    for i in range(1, 5):
        exec(f"sm{i} = (f.variables['swvl{i}'][:])")
        exec(f"sm{i} = sm{i}.reshape(1, 480, 151, 351)")
        exec(f"print(sm{i}.shape)")
    sm = np.vstack((sm1, sm2, sm3, sm4))
    sm = sm.swapaxes(0, 1)

#%%
ind = 0
for yr in range(40):
    for mn in range(12):
        sm[ind, :, :, :] = sm[ind, :, :, :]*mn_num[mn]
        ind += 1        

    
#%%
def CreatNC():
    new_NC = nc.Dataset(
        rf"E:/ERA5-Land-monthly/REGION1 SM/region1_SM_month_ORINGINAL.nc", 
        'w', format='NETCDF4')
    
    time = pd.date_range("1981-01-01", periods=480, freq="MS")
    
    new_NC.createDimension('time', 480)
    new_NC.createDimension('layer', 4)
    new_NC.createDimension('lat', len(lat))
    new_NC.createDimension('lon', len(lon))
    
    var = new_NC.createVariable('sm', 'f', ("time", "layer", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['sm'][:]=sm
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
     
    var.description = "1981.1-2020.12 SM 每月总和，原数据为月平均"
    var.time = 'time = pd.date_range("1981-01-01", periods=480, freq="MS")'
    var.unit = "m**3 m**-3"
    var.origin = "存放: lat和数据相同"
    
    new_NC.close()
            
            
# CreatNC()            

#%%
def CreatNC2():
    new_NC = nc.Dataset(
        rf"E:/ERA5-Land-monthly/REGION1 SM/region1_SM_month_ave.nc", 
        'w', format='NETCDF4')
    
    new_NC.createDimension('time', 12)
    new_NC.createDimension('layer', 4)
    new_NC.createDimension('lat', len(lat))
    new_NC.createDimension('lon', len(lon))
    
    var = new_NC.createVariable('sm', 'f', ("time", "layer", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['sm'][:]=sm_all_mn
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
     
    var.description = "1981.1-2020.12 SM 每月总和，原数据为月平均"
    var.time = 'time = pd.date_range("1981-01-01", periods=480, freq="MS")'
    var.unit = "m**3 m**-3"
    var.origin = "存放: lat和数据相同"
    
    new_NC.close()
    
def CreatNC3():
    new_NC = nc.Dataset(
        rf"E:/ERA5-Land-monthly/REGION1 SM/region1_SM_month_Z_Score.nc", 
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
    var.origin = "存放: lat和数据相同"
    
    new_NC.close()

 
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

#%% 多年月平均
sm_all_mn = []
for mn in range(1, 13):
    sm_mn = sm_month(mn, sm)
    sm_mn_ave = np.nanmean(sm_mn, axis=0)
    sm_all_mn.append(sm_mn_ave)
sm_all_mn = np.array(sm_all_mn)

# CreatNC2()

#%% Z-Score值
sm_mn_z = np.zeros((40, 4, 151, 351))
data_z = np.zeros((480, 4, 151, 351))
for mn in range(1, 13):
    sm_mn = sm_month(mn, sm)
    for r in range(151):
        for c in range(351):
            for l in range(4):
                sm_mn_z[:, l, r, c] = preprocessing.scale(sm_mn[:, l, r, c])
                data_z = sm_month_reverse(mn, sm_mn_z, data_z)

# CreatNC3()
