# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:32:02 2022

@author: MaYutong
"""
import os 

import netCDF4 as nc
import numpy as np
import pandas as pd
import pprint 
from pyhdf.SD import SD,SDC

import xarray as xr

def read_hdf(full_path):
    global data1
    hdf = SD(full_path)
    '''
    print(hdf.info())
    data = hdf.datasets()
    for idx,sds in enumerate(data.keys()):
    	print (idx,sds)
    '''
    sds_obj = hdf.select('GPP')
    data1 = sds_obj.get()
    #pprint.pprint(sds_obj.attributes())
    
    return data1

#%%

lat = np.linspace(89.975, -89.975, 3600)
lon = np.linspace(-180, 179.95, 7200)

def region(data):
    gpp_global = xr.DataArray(data, dims=['y', 'x'], coords=[
                               lat, lon])  # 原SPEI-base数据
    gpp_af = gpp_global.loc[60:35, 100:150]
    
    #print(gpp_af.x)
    #print(gpp_af.y)
    
    return gpp_af

#%%
def os_data(t_str, yr):
    global gpp
    # 想要移动文件所在的根目录
    rootdir = rf"/mnt/e/GLASS-GPP/AVHRR/{yr}/"
    # 获取目录下文件名清单
    files = os.listdir(rootdir)
    
    for file in files:
        for i, x in enumerate(t_str):
            if str(yr)+x in file:  # 因为索要移动的文件名均有‘_’,因此利用此判断其是否是所需要移动的文件
                full_path = os.path.join(rootdir, file) #完整的路径
                print(full_path)
                gpp = read_hdf(full_path)
                
                gpp = region(gpp)
                
                gpp = np.array(gpp, dtype="float64")
                gpp[gpp==65535]=np.nan
                gpp = gpp.reshape(1, 500, 1001)
                if i==0:
                    gpp_all = gpp
                else:
                    gpp_all = np.vstack((gpp_all, gpp))
                    
    return gpp_all
         
#%%
def CreatNC(yr, data):
    lat_rg = np.linspace(59.975, 35.025, 500)
    lon_rg = np.linspace(100, 150, 1001)
    
    month = np.arange(12)
    
    new_NC = nc.Dataset(
        rf"/mnt/e/GLASS-GPP/Month RG/GLASS_GPP_RG_{yr}.nc", 
        'w', format='NETCDF4')
    
    new_NC.createDimension('month', 12)
    new_NC.createDimension('lat', 500)
    new_NC.createDimension('lon', 1001)
    
    var=new_NC.createVariable('GPP', 'f', ("month","lat","lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['GPP'][:]=data
    new_NC.variables['lat'][:]=lat_rg
    new_NC.variables['lon'][:]=lon_rg
    
    var.description = "Units: gC m-2 month-1; FillValue: 65535, 已处理为np.nan"
    '''
    var.units = "gC m-2 month-1"
    var.FillValue = "65535, 已处理为np.nan"
    var.scale_factor = "0.01"
    var.validrange = "[0, 3000]"
    '''
    #最后记得关闭文件
    new_NC.close()

#%% 最终运行
t = np.arange(1, 365, 8)
df = pd.read_excel('/mnt/e/ERA5/每月天数.xlsx')
days = df['平年2']

df2 = pd.read_excel('/mnt/e/GLASS-ET/加权算法.xlsx')
df2.columns = np.arange(1, 13, 1)

#%%
def month_weighted(gpp, t):
    t = t.dropna(axis=0)
    if len(t) == 5:
        gpp_mn = gpp[0, :, :]*(t[1]-t[0])+\
                gpp[2, :, :]*(t[4]-t[3])+\
                (gpp[0, :, :]+gpp[1, :, :])*(t[2]-t[1])/2+\
                (gpp[1, :, :]+gpp[2, :, :])*(t[3]-t[2])/2
    elif len(t) == 6:
        gpp_mn = gpp[0, :, :]*(t[1]-t[0])+\
                gpp[3, :, :]*(t[5]-t[4])+\
                (gpp[0, :, :]+gpp[1, :, :])*(t[2]-t[1])/2+\
                (gpp[1, :, :]+gpp[2, :, :])*(t[3]-t[2])/2+\
                (gpp[2, :, :]+gpp[3, :, :])*(t[4]-t[3])/2
            
    return gpp_mn

def main(yr):
    global gpp_mn_weighted, gpp_yr_weighted, gpp_mn
    gpp_yr_weighted = []
    for mn in range(12):
        if mn<11:
            ind = np.logical_and(t>days[mn], t<days[mn+1])
        else:
            ind = t>days[mn]
            
        a = t[ind]
        t_str = []
        [t_str.append(str(x).zfill(3)) for x in a]
        print("\n", t_str, "\n")
        
        gpp_mn = os_data(t_str, yr)
        gpp_mn_weighted = month_weighted(gpp_mn, df2[mn+1])
        gpp_yr_weighted.append(gpp_mn_weighted)
    
    gpp_yr_weighted = np.array(gpp_yr_weighted)
            
    #CreatNC(yr, gpp_yr_weighted*0.01) ##scale_factor

yr = np.arange(1982, 1983)
for year in yr:
    main(year)
    print(f"{year} data  is done!")

#%%
a = gpp_yr_weighted[5, :, :]*0.01

import matplotlib.pyplot as plt

plt.figure(1, dpi=500)
plt.imshow(a, cmap="Set3", origin="upper")
plt.colorbar(shrink=0.75)
plt.show()

# %%
