# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:41:37 2022

@author: MaYutong
"""
#%%
import datetime as dt
import netCDF4 as nc
from matplotlib import cm
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
def read_nc2():
    global co2, lat3, lon3
    inpath = r"E:/CO2/DENG/CO2_81_13_RG_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        lat3 = (f.variables['lat'][:])
        lon3 = (f.variables['lon'][:])
        co2 = (f.variables['co2'][:])

        
#%%
def read_nc3():
    global co2_anom
    inpath = r"E:/CO2/DENG/CO2_81_13_RG_ANOM_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        co2_anom = (f.variables['co2'][:])


#%%
def co2_xarray(band1):
    t = np.arange(33)
    mn = np.arange(1, 13, 1)
    co2=xr.DataArray(band1,dims=['mn', 't', 'y','x'],coords=[mn, t, lat3, lon3])
    co2_MG = co2.loc[:, :, 40:55, 100:125]
    lat_MG = co2_MG.y
    lon_MG = co2_MG.x
    co2_MG = np.array(co2_MG)
    return co2_MG, lat_MG, lon_MG

#%% 480(输入data) -> 月 年
def region(data):
    t = np.arange(396)
    data_global = xr.DataArray(data, dims=['t', 'y', 'x'], coords=[
                               t, lat3, lon3])  # 原SPEI-base数据
    data_rg = data_global.loc[:, 40:55, 100:125] 
    

    return data_rg


#%% mask数组
def mask(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((12, 33, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(12):
        for l in range(33):
            a = data[i, l, :, :]
            a = ma.masked_array(a, mask=lcc2)
            spei_ma[i, l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(2, 3))
    
    return spei_ma_ave

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

#%%
def plot(data1, data2, title):
    fig, axs = plt.subplots(4, 3, figsize=(20, 12), dpi=150, sharey=True)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=0.1, hspace=0.25)
    
    df = pd.read_excel('E:/ERA5/每月天数.xlsx')
    mn_str = df['月份']

    t = np.arange(1981, 2021)
    t2 = np.arange(1981, 2014)
    
    ind = 0
    for i in range(4):
        for j in range(3):
            ax2 = axs[i, j].twinx()
            lin1 = ax2.bar(t2, data2[ind, :], color='forestgreen', label="CO2 Anom", alpha=0.5) #
            
            axs[i, j].scatter(t2, data1[ind, :], c='k', s=20) #
            lin2 = axs[i, j].plot(t2, data1[ind, :], c='k', label="CO2", zorder=2) #
            
            axs[i, j].text(1980.5, 410, f"{mn_str[ind]}", fontsize=15) #
            ind += 1
            axs[i, j].set_ylim(320, 420)
            axs[i, j].set_yticks(np.arange(320, 421, 20))
            axs[i, j].tick_params(labelsize=15)
            
            ax2.set_ylim(-40, 40)
            ax2.set_yticks(np.arange(-40, 40.1, 20))
            ax2.tick_params(axis='y', labelsize=15, colors='forestgreen')
            ax2.spines["right"].set_color('forestgreen')
            if j!=2:
                ax2.get_yaxis().set_visible(False)
    '''
    #合并图例
    lins=lin1+lin2
    labs=[l.get_label() for l in lins]
    ax.legend(lins,labs,loc="upper right", fontsize=15)
    '''
    plt.ylabel("CO2 (ppm)", fontsize=15)
    plt.suptitle(f"{title}", fontsize=25)
    plt.savefig(rf'E:/CO2/JPG/{title}.jpg', 
                bbox_inches='tight')
    plt.show()
    
#%%
read_nc()
read_nc2()
read_nc3()

co2 = region(co2)
co2_MG = mn_yr(co2)
co2_ave = mask(130, co2_MG)


co2anom_MG, _, _ = co2_xarray(co2_anom)
co2anom_ave = mask(130, co2anom_MG)

print(np.nanmax(co2_ave), np.nanmin(co2_ave))
print(np.nanmax(co2anom_ave), np.nanmin(co2anom_ave))

plot(co2_ave, co2anom_ave, f"MG Grassland CO2")

