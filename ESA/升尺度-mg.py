# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:12:46 2022

@author: MaYutong
"""

from collections import Counter

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
import xlsxwriter
#%%
def read_nc(lat1, lat2, lon1, lon2):
    
    global a1, a2, o1, o2
    global df
    inpath = (r"E:/ESA/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2000-v2.0.7cds.nc")
    with nc.Dataset(inpath) as f:
        print(f.variables.keys())
        print(f.variables['lccs_class'])
        
        lat_all = (f.variables['lat'][:])
        lon_all = (f.variables['lon'][:])
        
        ''' (75-30N, 0-180E) Global: (90N-90S, -180-180)
        a1 = np.where(lat_all<=75)[0][0]
        a2 = np.where(lat_all>=30)[-1][-1]
        o1 = np.where(lon_all>=0)[0][0]
        o2 = np.where(lon_all<=180)[-1][-1]
        '''
        a1 = np.where(lat_all<=lat1)[0][0]
        a2 = np.where(lat_all>=lat2)[-1][-1]
        o1 = np.where(lon_all>=lon1)[0][0]
        o2 = np.where(lon_all<=lon2)[-1][-1]
        
        lat = (f.variables['lat'][a1:a2])
        lon = (f.variables['lon'][o1:o2])
        
        lcc = (f.variables['lccs_class'][:, a1:a2, o1:o2])
        
        lcc = np.squeeze(lcc)
        color1 = ["#000000"]
        flag_colors = f.variables['lccs_class'].flag_colors.split(" ")
        flag_colors = color1+flag_colors
        flag_values = f.variables['lccs_class'].flag_values
        flag_meanings = f.variables['lccs_class'].flag_meanings.split(" ")
        
        df = pd.DataFrame(dict(values=flag_values, colors=flag_colors, meanings=flag_meanings))

        
    return lcc, lat, lon


#%%
def lcc_spei(data, a1, a2, o1, o2):
    lcc = xr.DataArray(data, dims=['y', 'x'], coords=[lat, lon])
    lcc_rg = lcc.loc[a2:a1, o1:o2]
    
    lcc_rg = np.array(lcc_rg)
    return lcc_rg

#%%
def lcc_count(data, df):
    sp = data.shape
    data = data.reshape(1, sp[0]*sp[1])
    data = np.squeeze(data)
    
    lcc_c = Counter(data)
    a = dict(lcc_c)
    b = sorted(a.keys())
    a = sorted(a.items())
    print(a)

    b =list(map(int,b))
    b = pd.Series(b) #uint8 类型，存储0-255，虽然为负数，但是数值相等。
    b.name = "IM"
    
    df = pd.merge(df, b, left_on='values', right_on='IM')
    
    a_per = []
    for x in a:
        a_per.append(x[-1]/(sp[0]*sp[1]))
    df['IM_per'] = a_per
    
    return df

#%%
def lcc_count2(data):
    sp = data.shape
    data = data.reshape(1, sp[0]*sp[1])
    data = np.squeeze(data)
    
    return Counter(data)

#%%
def down_scaling():
    lat_spei = np.arange(40., 55.1, 0.5) ########
    lon_spei = np.arange(100, 125.1, 0.5) ########
    spei_lcc = np.zeros((30, 50)) ########
    
    ind = 0
    for j, y in enumerate(lat_spei[:-1]):
        for i, x in enumerate(lon_spei[:-1]):
            a1, a2, o1, o2 = y, lat_spei[j+1], x, lon_spei[i+1]
            print(a1, a2, o1, o2, "\n")
            lcc_pixel = lcc_spei(lcc, a1, a2, o1, o2)
            c = lcc_count2(lcc_pixel)
            spei_lcc[j, i] = c.most_common(1)[0][0]
            ind += 1
    print(ind)
    return spei_lcc

spei_lat = np.arange(40.25, 55, 0.5) ########
spei_lon = np.arange(100.25, 125, 0.5) ########
#%%
lcc, lat, lon = read_nc(55, 40, 100, 125) ########
#%%
spei_lcc = down_scaling()
df = lcc_count(lcc, df)

#%%
writer = pd.ExcelWriter("MG ESA LUC降尺度SPEI03.xlsx", engine='xlsxwriter')

df.to_excel(writer, sheet_name="MG", index=False)
    
writer.save()
#%% 存储NC文件
def CreatNC(data):
    new_NC = nc.Dataset(r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc", 'w', format='NETCDF4')
    
    new_NC.createDimension('lat', 30) ########
    new_NC.createDimension('lon', 50) ########
    
    var=new_NC.createVariable('lcc', 'f', ("lat", "lon"))
    new_NC.createVariable('lat', 'f', ("lat"))
    new_NC.createVariable('lon', 'f', ("lon"))
    
    new_NC.variables['lcc'][:] = data
    new_NC.variables['lat'][:] = spei_lat
    new_NC.variables['lon'][:] = spei_lon
        
    var.describe = "粗略分类，降尺度，颜色及分类存储在excel LUC class里"
    
    new_NC.close()

CreatNC(spei_lcc)
