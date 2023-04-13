# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 16:23:00 2022

@author: MaYutong
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr

#%%
def read_nc(inpath):
    global lat, lon
    
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        #print(f.variables['ndvi'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        gpp = (f.variables['GPP'][:])
        
    return gpp

#%%
def read_nc2():
    global e, lat2, lon2
    inpath = r"E:/Gleamv3.6a/v3.6a/global/E_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        e = (f.variables['E'][:])
 
#%% 提取区域
def pre_xarray(band1):
    t = np.arange(40)
    mn = np.arange(1, 13, 1)
    pre=xr.DataArray(band1,dims=['mn', 't', 'y','x'],coords=[mn, t, lat2, lon2])
    pre_MG = pre.loc[:, :, 35:60, 100:150]
    lat_MG = pre_MG.y
    lon_MG = pre_MG.x
    pre_MG = np.array(pre_MG)
    return pre_MG, lat_MG, lon_MG


#%% 插值
def GLASS_SPEI(data):
    t = np.arange(12)
    gpp_GLASS = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                               t, lat, lon])  # 原SPEI-base数据
    gpp_spei = gpp_GLASS.interp(t=t, y=lat_RG, x=lon_RG, method="linear")

    return np.array(gpp_spei)

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
def CreatNC(yr, data):
    month = np.arange(12)
    
    new_NC = nc.Dataset(
        rf"E:/GLASS-GPP/Month RG/GLASS_GPP_RG_SPEI_0.5X0.5_{yr}.nc", 
        'w', format='NETCDF4')
    
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 50)
    new_NC.createDimension('lon', 100)
    
    var=new_NC.createVariable('GPP', 'f', ("month","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['GPP'][:]=data
    new_NC.variables['lat'][:]=lat_RG
    new_NC.variables['lon'][:]=lon_RG
    
    var.description = "Units: gC m-2 month-1; FillValue: 65535, 已处理为np.nan, scale也已处理过"
    
    #最后记得关闭文件
    new_NC.close()
    
def CreatNC2(yr, data):
    month = np.arange(12)
    
    new_NC = nc.Dataset(
        rf"E:/GLASS-GPP/Month RG WUE/GLASS_WUE_RG_SPEI_0.5X0.5_{yr}.nc", 
        'w', format='NETCDF4')
    
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 50)
    new_NC.createDimension('lon', 100)
    
    var=new_NC.createVariable('WUE', 'f', ("month","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['WUE'][:]=data
    new_NC.variables['lat'][:]=lat_RG
    new_NC.variables['lon'][:]=lon_RG
    
    var.description = "Units: gC m-2 mm-1; FillValue: np.nan, scale也已处理过"
    
    #最后记得关闭文件
    new_NC.close()

#%% Anomaly    
def CreatNC3(yr, data):
    month = np.arange(12)
    
    new_NC = nc.Dataset(
        rf"E:/GLASS-GPP/Month RG/GLASS_GPP_RG_Anomaly_SPEI_0.5X0.5_{yr}.nc", 
        'w', format='NETCDF4')
    
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 50)
    new_NC.createDimension('lon', 100)
    
    var=new_NC.createVariable('GPP', 'f', ("month","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['GPP'][:]=data
    new_NC.variables['lat'][:]=lat_RG
    new_NC.variables['lon'][:]=lon_RG
    
    var.description = "Units: gC m-2 month-1; FillValue: 65535, 已处理为np.nan, scale也已处理过"
    
    #最后记得关闭文件
    new_NC.close()
    
def CreatNC4(yr, data):
    month = np.arange(12)
    
    new_NC = nc.Dataset(
        rf"E:/GLASS-GPP/Month RG WUE/GLASS_WUE_RG_Anomaly_SPEI_0.5X0.5_{yr}.nc", 
        'w', format='NETCDF4')
    
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 50)
    new_NC.createDimension('lon', 100)
    
    var=new_NC.createVariable('WUE', 'f', ("month","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['WUE'][:]=data
    new_NC.variables['lat'][:]=lat_RG
    new_NC.variables['lon'][:]=lon_RG
    
    var.description = "Units: gC m-2 mm-1; FillValue: np.nan, scale也已处理过"
    
    #最后记得关闭文件
    new_NC.close()
    
#%%
read_nc2()
e_mn = mn_yr(e)
e_RG, lat_RG, lon_RG = pre_xarray(e_mn)

gpp_all = []
wue_all = []

for i, yr in enumerate(range(1982, 2019)):
    inpath =  rf"E:/GLASS-GPP/Month RG/GLASS_GPP_RG_{yr}.nc"
    #print(i+1, yr)
    gpp = read_nc(inpath)
    gpp_SPEI = GLASS_SPEI(gpp)
    gpp_all.append(gpp_SPEI)
    #CreatNC(yr, gpp_SPEI)
    
    et = e_RG[:, i+1, :, :]
    
    wue = gpp_SPEI/et
    wue_all.append(wue)
    #CreatNC2(yr, wue)

#%%
gpp_all = np.array(gpp_all)
wue_all = np.array(wue_all)    
gpp_ave = np.nanmean(gpp_all, axis=0)
wue_ave = np.nanmean(wue_all, axis=0)

for i, yr in enumerate(range(1982, 2019)):
    inpath =  rf"E:/GLASS-GPP/Month RG/GLASS_GPP_RG_{yr}.nc"
    #print(i+1, yr)
    gpp = read_nc(inpath)
    gpp_SPEI = GLASS_SPEI(gpp)
    #CreatNC(yr, gpp_SPEI)
    
    et = e_RG[:, i+1, :, :]
    
    wue = gpp_SPEI/et
    #CreatNC2(yr, wue)
    wue_a = np.zeros((12, 50, 100))
    gpp_a = np.zeros((12, 50, 100))
    
    for mn in zip(range(12)):
        gpp_a[mn, :, :] = gpp_SPEI[mn, :, :]-gpp_ave[mn, :, :]
        wue_a[mn, :, :] = wue[mn, :, :]-wue_ave[mn, :, :]
        CreatNC3(yr, gpp_a)
        CreatNC4(yr, wue_a)


#%%
'''
a = gpp_SPEI[7, :, :]
b = wue[7, :, :]
import matplotlib.pyplot as plt

plt.figure(1, dpi=500)
plt.imshow(a, cmap="Blues", origin="lower")
plt.colorbar(shrink=0.75)
plt.show()
'''
