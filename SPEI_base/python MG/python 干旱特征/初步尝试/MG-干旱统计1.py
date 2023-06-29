# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:27:24 2022

@author: MaYutong
"""

import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import openpyxl
import pandas as pd


#%%
def read_nc():
    global lcc, lat, lon, df
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
        
#%%
def read_nc2():
    global spei, t
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        spei = (f.variables['spei'][960:])
        time = (f.variables['time'][960:])
        t = nc.num2date(time, 'days since 1900-01-01 00:00:0.0').data


#%%
read_nc()

# lcc2 = np.flip(lcc, axis=0)

read_nc2()
#%% mask数组
def mask(x):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((480, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(480):
        a = spei[i, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = spei_ma.mean(axis=(1, 2))
    
    return spei_ma_ave

#%%
def drought_thres_df(spei3, t3, threshold1, threshold2):
    for i in range(len(spei3)):
        if spei3[i]<threshold1:
            for j, y in enumerate(spei3[i+1:]):
                if y<threshold2:
                    continue      
                else:
                    break
            if j+1 > 2:
                print(f"start year: {t3[i].year}; start month: {t3[i].month}")
                print("duration month: ", j+1)
                print(f"end year: {t3[i+j].year}; end month: {t3[i+j].month}")
                print("")
                df2.loc[len(df2.index)] = [t3[i].year, t3[i].month, j+1, t3[i+j].year, t3[i+j].month]
            

spei_ave = mask(130)
df2 = pd.DataFrame(columns=["SY", "SM", "DD", "EY", "EM"])
thre = spei_ave.std()
drought_thres_df(spei_ave, t, -thre, -thre)
df2.drop_duplicates(["EY", "EM"], inplace=True)

#%%
writer = pd.ExcelWriter(r"E:/SPEI_base/python MG/MG Grassland干旱统计.xlsx", mode="a", engine="openpyxl")
df2.to_excel(writer, index=False, sheet_name="Grassland")
writer.save()

  
