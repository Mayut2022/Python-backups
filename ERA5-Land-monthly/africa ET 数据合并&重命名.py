# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 19:01:44 2022

@author: MaYutong
"""

import netCDF4 as nc
import numpy as np

import os
import pandas as pd
import xarray as xr

#%%
path = r"E:/ERA5-Land-monthly/africa Total evporation 1982-2020/"
filelist = os.listdir(path)

def read_data(inpath):
    global lat, lon
    with nc.Dataset(inpath) as f:
        
        f = nc.Dataset(inpath, mode='r')
        print(f.variables.keys())
        # print(f.variables['time'], '\n')
        # print(f.variables['e'], '\n')
        
        et = (f.variables['e'][:])
        lat = (f.variables['latitude'][:])
        lon = (f.variables['longitude'][:])
        
        time=(f.variables['time'][:])
        t=nc.num2date(time, 'hours since 1900-01-01 00:00:0.0').data
        
        return et, time
        
#%%
def cg(data):
    e_af = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                               time_all, lat, lon])  # 原SPEI-base数据
    e_cg = e_af.loc[:, 5:-6, 14:31]

    return  e_cg

#%%
def CreatNC(data1, data2):
    new_NC = nc.Dataset(
        rf"E:/ERA5-Land-monthly/africa Total evporation 1982-2020/africa Total evporation 1982-2020.nc",
        'w', format='NETCDF4')
    
    new_NC.createDimension('lat', 721)
    new_NC.createDimension('lon', 701)
    new_NC.createDimension('time', 468)
    
    var=new_NC.createVariable('ET', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('time', 'f', ("time"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['ET'][:]=data1
    new_NC.variables['time'][:]=data2
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
        
    var.units = "m of water equivalent (换算 1 m of water equivalent = 1000 mm)"
    var.description = "scalar offset可忽略, 缺测值: -32767, 使用时须 x1000x-1"
    
    #最后记得关闭文件
    new_NC.close()

def CreatNC2(data1, data2):
    new_NC = nc.Dataset(
        rf"E:/ERA5-Land-monthly/africa Total evporation 1982-2020/congo Total evporation 1982-2020.nc", 
        'w', format='NETCDF4')
    
    new_NC.createDimension('time', 468)
    new_NC.createDimension('lat', len(latcg))
    new_NC.createDimension('lon', len(loncg))
    
    var = new_NC.createVariable('ET', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    new_NC.createVariable('time', 'f', ("time"))
    
    new_NC.variables['ET'][:]=data1
    new_NC.variables['time'][:]=data2
    new_NC.variables['lat'][:]=latcg
    new_NC.variables['lon'][:]=loncg
     
    var.description = "1982.1-2020.12 ET 每月实际蒸散发总和平均值 mm/month"
        
    #最后记得关闭文件
    new_NC.close()
    
#%%


for i, file in enumerate(filelist[:4]):
    inpath = path + file
    et, time = read_data(inpath)
    
    if i == 0:
        et_all = et
        time_all = time
    else:
        et_all = np.vstack((et_all, et))
        time_all = np.hstack((time_all, time))
    
#CreatNC(et_all, time_all)

#%%
et_all_cg = cg(et_all)
latcg = et_all_cg.y
loncg = et_all_cg.x
CreatNC2(et_all_cg, time_all)
