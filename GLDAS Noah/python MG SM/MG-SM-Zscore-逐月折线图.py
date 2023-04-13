# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:53:40 2022

@author: MaYutong
"""

import datetime as dt
import netCDF4 as nc
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr


#%%
def read_nc():
    global lcc, lat, lon
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
        
#%%
def read_nc2(inpath):
    global sm_z, lat2, lon2
    
    with nc.Dataset(inpath, mode='r') as f:
        
        print(f.variables.keys())
        
        sm_z = (f.variables['sm_z'][:, :, :])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        
# %%
def MG(data):
    t = np.arange(1, 41, 1)
    l = np.arange(1, 5, 1)
    spei_global = xr.DataArray(data, dims=['t', 'l', 'y', 'x'], coords=[
                               t, l, lat2, lon2])  # 原SPEI-base数据
    spei_rg1 = spei_global.loc[:, :, 40:55, 100:125]
    spei_rg1 = np.array(spei_rg1)
    return spei_rg1

#%% mask数组
def mask(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((40, 4, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(40):
        for l in range(4):
            a = data[i, l, :, :]
            a = ma.masked_array(a, mask=lcc2)
            spei_ma[i, l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(2, 3))
    
    return spei_ma_ave


#%%
def plot(data, title):
    fig, axs = plt.subplots(4, 3, figsize=(20, 12), dpi=150, sharey=True, sharex=True)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=0.05, hspace=0.1)
    
    df = pd.read_excel('E:/ERA5/每月天数.xlsx')
    mn_str = df['月份']

    t = np.arange(1981, 2021)
    
    ind = 0
    for i in range(4):
        for j in range(3):
            axs[i, j].scatter(t, data[ind], c='brown', s=20) #
            axs[i, j].plot(t, data[ind], c='brown', label="GLDAS") #
            axs[i, j].text(1980.5, 1.8, f"{mn_str[ind]}", fontsize=15) #
            ind += 1
            axs[i, j].set_ylim(-2.2, 2.2)
            axs[i, j].set_yticks(np.arange(-2, 2.1, 1))
            axs[i, j].tick_params(labelsize=15)
            
    plt.legend(loc='lower right', fontsize=15)
            
    plt.suptitle(f"{title}", fontsize=25)
    plt.savefig(rf'E:/GLDAS Noah/JPG_RG/{title}.jpg', 
                bbox_inches='tight')
    plt.show()
        
#%%
read_nc()
sm_all = []
for mn in range(1, 13):
    inpath = rf"E:/GLDAS Noah/DATA_RG/Zscore_Anomaly/SM_81_20_SPEI0.5x0.5_Zscore_Anomaly_month{mn}.nc"
    read_nc2(inpath)
    
    sm_IM = MG(sm_z)
    
    sm_IM_ave = mask(130, sm_IM)
    sm_all.append(sm_IM_ave)
sm_all = np.array(sm_all)
#%%

print(sm_all.max(), sm_all.min())
for layer in range(4):
    print(sm_all[:, :, layer].max(), sm_all[:, :, layer].min())
    plot(sm_all[:, :, layer], f"MG Grassland SM Layer{layer+1} Zscore")


