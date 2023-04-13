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

def africa(data):
    et_global = xr.DataArray(data, dims=['y', 'x'], coords=[
                               lat, lon])  # 原SPEI-base数据
    et_af = et_global.loc[37:-35, -18:52]

    return et_af

#%%
def os_data(t_str, yr):
    # 想要移动文件所在的根目录
    rootdir = rf"E:/GLASS-GPP/AVHRR/{yr}/"
    # 获取目录下文件名清单
    files = os.listdir(rootdir)
    
    for file in files:
        for i, x in enumerate(t_str):
            if str(yr)+x in file:  # 因为索要移动的文件名均有‘_’,因此利用此判断其是否是所需要移动的文件
                full_path = os.path.join(rootdir, file) #完整的路径
                print(full_path)
                et = read_hdf(full_path)
                et = africa(et)
                et = np.array(et)
                et = et.reshape(1, 1440, 1401)
                if i==0:
                    et_all = et
                else:
                    et_all = np.vstack((et_all, et))
                    
    return et_all
         
#%%

def CreatNC(yr, data, i, t_str):
    new_NC = nc.Dataset(rf"E:/GLASS-GPP/AVHRR/{yr}/Glass_GPP_{yr}_month{i}.nc", 'w', format='NETCDF4')
    
    new_NC.createDimension('doys', len(t_str))
    new_NC.createDimension('lat', 1440)
    new_NC.createDimension('lon', 1401)
    
    var=new_NC.createVariable('ET', 'f', ("doys","lat","lon"))
    new_NC.createVariable('doys', 'd', ("doys"))
    
    new_NC.variables['ET'][:]=data
    new_NC.variables['doys'][:]=t_str
        
    
    #最后记得关闭文件
    new_NC.close()
    
#%% 最终运行
t = np.arange(1, 365, 8)
df = pd.read_excel('E:/ERA5/每月天数.xlsx')
days = df['平年2']

def main(yr):
    for i in range(12):
        if i<11:
            ind = np.logical_and(t>days[i], t<days[i+1])
        else:
            ind = t>days[i]
            
        a = t[ind]
        t_str = []
        [t_str.append(str(x).zfill(3)) for x in a]
        print("\n", t_str, "\n")
        
        et_all = os_data(t_str, yr)
        #CreatNC(yr, et_all, i+1, t_str)

yr = np.arange(1984, 2019)
for year in yr:
    main(year)
    print(f"{year} data  is done!")


