# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:50:59 2022

@author: MaYutong
"""

import netCDF4 as nc
import numpy as np
import os
import rasterio
import xarray as xr
from scipy.stats import linregress

lat = np.linspace(89.975, -89.975, 3600)
lon = np.linspace(-180, 179.95, 7200)

#%%
def sif_xarray(band1):
    sif=xr.DataArray(band1,dims=['y','x'],coords=[lat, lon])
    return sif.loc[60:35, 100:150]

#%%
def SIF_all():
    for yr in range(2001, 2021, 1):
        inpath = (f'E:/Gosif_annual/01_20/GOSIF_{yr}.tif')
        ds = rasterio.open(inpath)
        sif = ds.read(1)
        
        sif_RG=sif_xarray(sif)
        lat_nc= np.array(sif_RG.y)
        lon_nc= np.array(sif_RG.x)
        sif_RG=np.array(sif_RG)
            
        sif_RG=sif_RG.reshape(1, 500, 1001)
            
        if(yr==2001):
            sif_all=sif_RG
        else:
            sif_all=np.vstack((sif_all, sif_RG))
        ds.close()

    return sif_all, lat_nc, lon_nc

#%%生成新的nc文件
def CreatNC(data):
    year = np.arange(2001, 2021, 1)
    
    new_NC = nc.Dataset(r"E:/Gosif_annual/01_20/GOSIF_01_20_RG.nc", 'w', format='NETCDF4')
    
    new_NC.createDimension('year', 20)
    new_NC.createDimension('lat', 500)
    new_NC.createDimension('lon', 1001)
    
    var=new_NC.createVariable('sif', 'f', ("year","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['sif'][:]=data
    new_NC.variables['lat'][:]=lat_nc
    new_NC.variables['lon'][:]=lon_nc
        
    var.sif= ("包含此前尝试的三个区域，region1 ESA IM")
    var.lat_range="[60, 35], 500, 精度：0.05, 边界：[59.975, 35.025]"
    var.lon_range="[100, 150], 1001, 精度：0.05, 边界：[100, 150]"
    var.data="scale factor和缺测值均未处理"
    var.Fillvalues="32767 (water bodies) and 32766 (lands under snow/ice throughout the year)"
    var.veg_nonveg="annual mean <0 & 32767/32766"
    
    #最后记得关闭文件
    new_NC.close()

#%%生成新的nc文件
def CreatNC2(data1, data2):
    year = np.arange(2001, 2021, 1)
    
    new_NC = nc.Dataset(r"E:/Gosif_annual/01_20/GOSIF_01_20_RG_Trend.nc", 'w', format='NETCDF4')
    
    new_NC.createDimension('lat', 500)
    new_NC.createDimension('lon', 1001)
    
    var=new_NC.createVariable('s', 'f', ("lat","lon"))
    new_NC.createVariable('p', 'f', ("lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['s'][:]=data1
    new_NC.variables['p'][:]=data2
    new_NC.variables['lat'][:]=lat_nc
    new_NC.variables['lon'][:]=lon_nc
        
    var.sif= ("包含此前尝试的三个区域，region1 ESA IM")
    var.lat_range="[60, 35], 500, 精度：0.05, 边界：[59.975, 35.025]"
    var.lon_range="[100, 150], 1001, 精度：0.05, 边界：[100, 150]"
    var.data="scale factor和缺测值均已处理，乘过0.0001"
    var.Fillvalues="32767 (water bodies) and 32766 (lands under snow/ice throughout the year)"
    var.veg_nonveg="annual mean <0 & 32767/32766"
    
    #最后记得关闭文件
    new_NC.close()
#%%
def trend(data):
    data = data*0.0001
    data[data==3.2767] = np.nan
    data[data==3.2766] = np.nan
    t = np.arange(1, 21, 1)
    s, r0, p = np.zeros((500, 1001)), np.zeros((500, 1001)), np.zeros((500, 1001))
    for r in range(500):
        if r%25 == 0:
            print(f"{r} is done!")
        for c in range(1001):
            a = data[:, r, c]
            if np.isnan(a).any():
                s[r, c], r0[r, c], p[r, c] = np.nan, np.nan, np.nan
            else:
                s[r, c], _, r0[r, c], p[r, c], _  = linregress(t, a)
    
    return s, p

#%%
if __name__ == '__main__':
    sif_all, lat_nc, lon_nc = SIF_all()
    #CreatNC(sif_all)
    s, p = trend(sif_all)
    CreatNC2(s, p)
    #print("%d is done!" %yr)