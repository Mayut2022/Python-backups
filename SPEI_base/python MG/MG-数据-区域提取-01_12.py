# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:37:39 2022

@author: MaYutong
"""

import netCDF4 as nc
import cftime
import xarray as xr
import numpy as np
# %%
def read_nc(scale):
    global time, lat, lon
    inpath = (rf'E:/SPEI_base/data/spei{scale}.nc')
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
    
        spei = (f.variables['spei'][:])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
    
        #print(f.variables['time'])
        time = (f.variables['time'][:])
        t=nc.num2date(f.variables['time'][:],'days since 1900-01-01 00:00:0.0').data
        
    return spei

# %%
def region1(data):
    spei_global = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                               time, lat, lon])  # 原SPEI-base数据
    spei_IM = spei_global.loc[:, 40:55, 100:125] ########

    return spei_IM, spei_IM.y, spei_IM.x


#%%

a = 0
b = 12
for i in range(120):
    c = np.nanmean(spei_IM[a:b, :, :], axis=0)
    c = c.reshape(1, 30, 50)
    if a == 0:
        spei_IM_ave = c
        a = a+12
        b = b+12
    else:
        spei_IM_ave = np.vstack((spei_IM_ave, c))
        a = a+12
        b = b+12
        
del a, b, c        


#%%生成新的nc文件
def CreatNC(data):
    new_NC = nc.Dataset(r"E:/SPEI_base/data/spei03_MG_annual.nc", 'w', format='NETCDF4')
    
    time = np.arange(1901, 2021, 1)
    new_NC.createDimension('time', 120)
    new_NC.createDimension('lat', 30)
    new_NC.createDimension('lon', 50)
    
    var=new_NC.createVariable('spei', 'f', ("time","lat","lon"))
    new_NC.createVariable('time', 'f', ("time"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['spei'][:]=data
    new_NC.variables['time'][:]=time
    new_NC.variables['lat'][:]=lat_IM
    new_NC.variables['lon'][:]=lon_IM
        
    
    #var.lat_range="[-37, 35], 144, 精度：0.5, 边界：[-34.75, 36.75]"
    #var.lon_range="[-18, 52], 140, 精度：0.5, 边界：[-17.75, 51.75]"
    var.Fillvalues="nan"
    var.time="用SPEI03算出来的年平均，1-12月SPEI03指数"
    
    #最后记得关闭文件
    new_NC.close()
    
#CreatNC(spei_IM_ave)

#%%生成新的nc文件
def CreatNC2(data, scale):
    new_NC = nc.Dataset(rf"E:/SPEI_base/data/spei{scale}_MG.nc", 'w', format='NETCDF4') ########
    
    new_NC.createDimension('time', 1440)
    new_NC.createDimension('lat', 30) ########
    new_NC.createDimension('lon', 50) ########
    
    var=new_NC.createVariable('spei', 'f', ("time","lat","lon"))
    new_NC.createVariable('time', 'f', ("time"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['spei'][:]=data
    new_NC.variables['time'][:]=time
    new_NC.variables['lat'][:]=lat_IM
    new_NC.variables['lon'][:]=lon_IM
        
    
    
    var.Fillvalues="nan"
    var.time="1901.1-2020.12 'days since 1900-01-01 00:00:0.0' "
    
    #最后记得关闭文件
    new_NC.close()
    
#CreatNC2(spei_IM)

#%%
scale = [str(x).zfill(2) for x in range(1, 13)]

for s in scale:
    data = read_nc(s)
    spei_IM, lat_IM, lon_IM = region1(data)
    CreatNC2(spei_IM, s)