# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:51:21 2022

1982-2020 和 1982-2018 非洲、刚果的月平均数据

@author: MaYutong
"""

import netCDF4 as nc
import numpy as np
import xarray as xr

#%%
def read_data():
    global lat, lon
    inpath = r"E:/ERA5-Land-monthly/africa Total evporation 1982-2020/africa Total evporation 1982-2020.nc"
    
    with nc.Dataset(inpath) as f:
        
        f = nc.Dataset(inpath, mode='r')
        # print(f.variables.keys())
        # print(f.variables['time'], '\n')
        # print(f.variables['e'], '\n')
        
        et = (f.variables['ET'][:])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        
        time=(f.variables['time'][:])
        t=nc.num2date(time, 'hours since 1900-01-01 00:00:0.0').data
        
        return et, t
    
#%%
def cg(data):
    e_af = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                               month, lat, lon])  # 原SPEI-base数据
    e_cg = e_af.loc[:, 5:-6, 14:31]

    return  e_cg


    
#%%
def CreatNC(data):
    new_NC = nc.Dataset(
        rf"E:/ERA5-Land-monthly/africa Total evporation 1982-2020/africa_ET_month_ave_82-20.nc", 
        'w', format='NETCDF4')
    
    new_NC.createDimension('time', 12)
    new_NC.createDimension('lat', len(lat))
    new_NC.createDimension('lon', len(lon))
    
    var = new_NC.createVariable('E', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['E'][:]=data
    new_NC.variables['lat'][:]=lat
    new_NC.variables['lon'][:]=lon
     
    var.description = "1982.1-2020.12 ET 每月实际蒸散发总和平均值 mm/month"
        
    #最后记得关闭文件
    new_NC.close()
    
def CreatNC2(data):
    new_NC = nc.Dataset(
        rf"E:/ERA5-Land-monthly/africa Total evporation 1982-2020/congo_ET_month_ave_82-20.nc", 
        'w', format='NETCDF4')
    
    new_NC.createDimension('time', 12)
    new_NC.createDimension('lat', len(latcg))
    new_NC.createDimension('lon', len(loncg))
    
    var = new_NC.createVariable('E', 'f', ("time", "lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['E'][:]=data
    new_NC.variables['lat'][:]=latcg
    new_NC.variables['lon'][:]=loncg
     
    var.description = "1982.1-2020.12 ET 每月实际蒸散发总和平均值 mm/month"
        
    #最后记得关闭文件
    new_NC.close()

#%%
def e_month(mn):
    e_mn = []
    ind = mn-1
    for i in range(39):
        e_mn.append(et[ind, :, :])
        ind+=12
        
    e_mn = np.array(e_mn)
    return e_mn


#%%
et, t = read_data()
et = et*1000*(-1)

month = np.arange(1, 13, 1)

e_all_mn = []
for mn in range(1, 13):
    e_mn = e_month(mn)
    e_mn_ave = np.mean(e_mn, axis=0)
    e_all_mn.append(e_mn_ave)
e_all_mn = np.array(e_all_mn)
CreatNC(e_all_mn)

#%%
e_all_mn_cg = cg(e_all_mn)
latcg = e_all_mn_cg.y
loncg = e_all_mn_cg.x
CreatNC2(e_all_mn_cg)


