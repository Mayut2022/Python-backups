# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 19:18:15 2022

@author: MaYutong
"""

import netCDF4 as nc
import cftime
import xarray as xr
import numpy as np
# %%
inpath = (r'E:/SPEI_base/data/spei03.nc')
with nc.Dataset(inpath) as f:
    # print(f.variables.keys())

    spei = (f.variables['spei'][:])
    lat = (f.variables['lat'][:])
    lon = (f.variables['lon'][:])

    #print(f.variables['time'])
    time = (f.variables['time'][:])
    t=nc.num2date(f.variables['time'][:],'days since 1900-01-01 00:00:0.0').data

# %%
def region1(data):
    spei_global = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                               time, lat, lon])  # 原SPEI-base数据
    spei_ASA = spei_global.loc[:, 25:55, 30:150] ########

    return spei_ASA


spei_ASA = region1(spei)
lat_ASA = spei_ASA.y
lon_ASA = spei_ASA.x

#%%

a = 0
b = 12
for i in range(120):
    c = np.nanmean(spei_ASA[a:b, :, :], axis=0)
    c = c.reshape(1, 60, 240)
    if a == 0:
        spei_ASA_ave = c
        a = a+12
        b = b+12
    else:
        spei_ASA_ave = np.vstack((spei_ASA_ave, c))
        a = a+12
        b = b+12
        
del a, b, c        


#%%生成新的nc文件
def CreatNC(data):
    new_NC = nc.Dataset(r"E:/SPEI_base/data/spei03_ASA_annual.nc", 'w', format='NETCDF4')
    
    time = np.arange(1901, 2021, 1)
    new_NC.createDimension('time', 120)
    new_NC.createDimension('lat', 60)
    new_NC.createDimension('lon', 240)
    
    var=new_NC.createVariable('spei', 'f', ("time","lat","lon"))
    new_NC.createVariable('time', 'f', ("time"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['spei'][:]=data
    new_NC.variables['time'][:]=time
    new_NC.variables['lat'][:]=lat_ASA
    new_NC.variables['lon'][:]=lon_ASA
        
    
    #var.lat_range="[-37, 35], 144, 精度：0.5, 边界：[-34.75, 36.75]"
    #var.lon_range="[-18, 52], 140, 精度：0.5, 边界：[-17.75, 51.75]"
    var.Fillvalues="nan"
    var.time="用SPEI03算出来的年平均，1-12月SPEI03指数"
    
    #最后记得关闭文件
    new_NC.close()
    
# CreatNC(spei_ASA_ave)

#%%生成新的nc文件
def CreatNC2(data):
    new_NC = nc.Dataset(r"E:/SPEI_base/data/spei03_ASA.nc", 'w', format='NETCDF4') ########
    
    new_NC.createDimension('time', 1440)
    new_NC.createDimension('lat', 60) ########
    new_NC.createDimension('lon', 240) ########
    
    var=new_NC.createVariable('spei', 'f', ("time","lat","lon"))
    new_NC.createVariable('time', 'f', ("time"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['spei'][:]=data
    new_NC.variables['time'][:]=time
    new_NC.variables['lat'][:]=lat_ASA
    new_NC.variables['lon'][:]=lon_ASA
        
    
    var.lat_range="[25, 55], 60, 精度：0.5, 边界：[25.25, 54.75]"
    var.lon_range="[30, 150], 240, 精度：0.5, 边界：[30.25, 149.75]"
    var.Fillvalues="nan"
    var.time="1901.1-2020.12 'days since 1900-01-01 00:00:0.0' "
    
    #最后记得关闭文件
    new_NC.close()
    
# CreatNC2(spei_ASA)

#%% (1440)->(12, 120)->(4, 120) 序列-月年-季节年

def mn_yr(data):
    spei_mn = []
    for mn in range(12):
        spei_ = []
        for yr in range(120):
            spei_.append(data[mn])
            mn += 12
        spei_mn.append(spei_)
            
    spei_mn = np.array(spei_mn)
    
    return spei_mn

def season_yr(data):
    spei_s = np.vstack((data[2:, :], data[:2, :]))
    spei_sea = []
    for mn1, mn2 in zip(range(0, 12, 3), range(3, 15, 3)):
        spei_sea.append(spei_s[mn1:mn2, :])

    spei_sea = np.array(spei_sea)
    spei_sea = spei_sea.mean(axis=1)

    return spei_sea

spei_mn = mn_yr(spei_ASA)
spei_sea = season_yr(spei_mn)

# %%
def CreatNC3(data):
    new_NC = nc.Dataset(r"E:/SPEI_base/data/spei03_ASA_season.nc", 'w', format='NETCDF4') ########
    
    new_NC.createDimension('season', 4)
    new_NC.createDimension('time', 120)
    new_NC.createDimension('lat', 60) ########
    new_NC.createDimension('lon', 240) ########
    
    var=new_NC.createVariable('spei', 'f', ("season", "time","lat","lon"))
    new_NC.createVariable('time', 'f', ("time"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['spei'][:]=data
    # new_NC.variables['time'][:]=time
    new_NC.variables['lat'][:]=lat_ASA
    new_NC.variables['lon'][:]=lon_ASA
        
    
    var.lat_range="[25, 55], 60, 精度：0.5, 边界：[25.25, 54.75]"
    var.lon_range="[30, 150], 240, 精度：0.5, 边界：[30.25, 149.75]"
    var.Fillvalues="nan"
    var.time="1901.1-2020.12 'days since 1900-01-01 00:00:0.0' "
    
    #最后记得关闭文件
    new_NC.close()

CreatNC3(spei_sea)
# %%
