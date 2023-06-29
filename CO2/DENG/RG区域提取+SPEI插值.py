# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:24:55 2022

@author: MaYutong
"""


import netCDF4 as nc
import numpy as np
import xarray as xr

def read_nc1(inpath):
    global lat, lon, time, t
    with nc.Dataset(inpath, mode='r') as f:
        
        print(f.variables.keys(), "\n")
        print(f.variables['value'], "\n")
        print(f.variables['Times'], "\n")
        
        time = (f.variables['Times'][1572:])
        t = nc.num2date(time, 'days since 1850-01-01').data
        
        lat = (f.variables['Latitude'][:])
        lon = (f.variables['Longitude'][:])
        co2 = (f.variables['value'][1572:, :, :]).data ##units: ppm; monthly mean
        
        return co2
    
# %%
def region(data):
    lat_spei = np.linspace(35.25, 59.75, 50)
    lon_spei = np.linspace(100.25, 149.75, 100)
    data_global = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                               t, lat, lon])  # 原SPEI-base数据
    data_rg = data_global.loc[:, 61:34, 99:151] ### 不然边界没有值
    
    data_ESA = data_rg.interp(t=t, y=lat_spei, x=lon_spei, method="linear")
    lat_ESA = data_ESA.y
    lon_ESA = data_ESA.x

    return data_ESA, lat_ESA, lon_ESA   

def MCD_to_ESA(data, lat, lon):
    lat_spei = np.linspace(-89.75, 89.75, 360)
    lon_spei = np.linspace(-179.75, 179.75, 720)

    data = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[t,
        lat, lon])  # 原SPEI-base数据

    data_ESA = data.interp(t=t, y=lat_spei, x=lon_spei, method="linear")
    lat_ESA = data_ESA.y
    lon_ESA = data_ESA.x

    return data_ESA, lat_ESA, lon_ESA



#%%生成新的nc文件
def CreatNC(data):
    new_NC = nc.Dataset(r"/mnt/e/CO2/DENG/CO2_81_13_RG_SPEI0.5x0.5.nc", 'w', format='NETCDF4')
    
    new_NC.createDimension('time', 396)
    new_NC.createDimension('lat', 50)
    new_NC.createDimension('lon', 100)
    
    var=new_NC.createVariable('co2', 'f', ("time","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['co2'][:]=data
    new_NC.variables['lat'][:]=lat_RG
    new_NC.variables['lon'][:]=lon_RG
        
    var.ndvi= ("包含此前尝试的三个区域，region1 ESA IM")
    var.data="scale factor和缺测值均未处理，似乎不用处理"
    var.units="units: ppm; 1ppm=10^-6mol/mol=1μmol/mol; the average of the month"
    var.time="units: days since 1850-01-01; 81-01~13-12"
    
    #最后记得关闭文件
    new_NC.close()            

def CreatNC3(data):
    new_NC = nc.Dataset(r"/mnt/e/CO2/DENG/CO2_81_13_Global_SPEI0.5x0.5.nc", 'w', format='NETCDF4')
    
    new_NC.createDimension('time', 396)
    new_NC.createDimension('lat', 360)
    new_NC.createDimension('lon', 720)
    
    var=new_NC.createVariable('co2', 'f', ("time","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['co2'][:]=data
    new_NC.variables['lat'][:]=lat_ESA
    new_NC.variables['lon'][:]=lon_ESA
        
    var.data="scale factor和缺测值均未处理，似乎不用处理"
    var.units="units: ppm; 1ppm=10^-6mol/mol=1μmol/mol; the average of the month"
    var.time="units: days since 1850-01-01; 81-01~13-12"
    
    #最后记得关闭文件
    new_NC.close()            

#%%
inpath = r"/mnt/e/CO2/DENG/CO2_1deg_month_1850-2013.nc"
co2 = read_nc1(inpath)

# co2_RG, lat_RG, lon_RG = region(co2)
#CreatNC(co2_RG)

co2_ESA, lat_ESA, lon_ESA = MCD_to_ESA(co2, lat, lon)
CreatNC3(co2_ESA)

#%% 480 -> 月 年
def mn_yr(data):
    q_mn = []
    for mn in range(12):
        q_ = []
        for yr in range(33):
            q_.append(data[mn])
            mn += 12
        q_mn.append(q_)
            
    q_mn = np.array(q_mn)
    return q_mn

co2_mn = mn_yr(co2_RG)
#%%生成新的nc文件
def CreatNC2(data):
    new_NC = nc.Dataset(r"/mnt/e/CO2/DENG/CO2_81_13_RG_ANOM_SPEI0.5x0.5.nc", 'w', format='NETCDF4')
    
    year = np.arange(1, 34, 1)
    month = np.arange(1, 13, 1)
   
    new_NC.createDimension('month', 12)
    new_NC.createDimension('year', 33)
    new_NC.createDimension('lat', 50)
    new_NC.createDimension('lon', 100)
    
    var=new_NC.createVariable('co2', 'f', ("month","year","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['co2'][:]=data
    new_NC.variables['lat'][:]=lat_RG
    new_NC.variables['lon'][:]=lon_RG
        
    var.ndvi= ("包含此前尝试的三个区域，region1 ESA IM")
    var.data="scale factor和缺测值均未处理，似乎不用处理"
    var.units="units: ppm; 1ppm=10^-6mol/mol=1μmol/mol; the average of the month"
    var.time="units: days since 1850-01-01; 81-01~13-12"
    
    #最后记得关闭文件
    new_NC.close()
    
#%%
########## 生成Anom数据
def Anom(data):
    co2_mn_ave = np.nanmean(data, axis=1)
    co2_mn_a = np.zeros((12, 33, 50, 100))
    for mn in range(12):
        print(f"Month{mn+1} is in programming!")
        for yr in range(33):
            for r in range(50):
                if r%30 == 0:
                    print(f"columns {r} is done!")
                for c in range(100):
                    co2_mn_a[mn, yr, r, c] = data[mn, yr, r, c]-co2_mn_ave[mn, r, c] #########
    
    return co2_mn_a

# co2_mn_anom = Anom(co2_mn)
# CreatNC2(co2_mn_anom)

