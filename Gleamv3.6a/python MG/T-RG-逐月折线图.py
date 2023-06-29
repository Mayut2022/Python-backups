# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:14:23 2022

@author: MaYutong
"""

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
    global e
    inpath = r"E:/Gleamv3.6a/v3.6a/global/Et_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        e = (f.variables['Et'][:])

        
#%%
def read_nc3():
    global e_anom, lat2, lon2
    inpath = r"E:/Gleamv3.6a/v3.6a/global/Etanom_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        e_anom = (f.variables['Et'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        
#%%
def pre_xarray(band1):
    t = np.arange(40)
    mn = np.arange(1, 13, 1)
    pre=xr.DataArray(band1,dims=['mn', 't', 'y','x'],coords=[mn, t, lat2, lon2])
    pre_MG = pre.loc[:, :, 40:55, 100:125]
    lat_MG = pre_MG.y
    lon_MG = pre_MG.x
    pre_MG = np.array(pre_MG)
    return pre_MG, lat_MG, lon_MG

#%% mask数组
def mask(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((12, 40, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(12):
        for l in range(40):
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
        for yr in range(40):
            q_.append(data[mn])
            mn += 12
        q_mn.append(q_)
            
    q_mn = np.array(q_mn)
    return q_mn

#%%
def plot(data1, data2, title):
    fig, axs = plt.subplots(4, 3, figsize=(20, 12), dpi=150, sharey=True)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=0.05, hspace=0.15)
    
    df = pd.read_excel('E:/ERA5/每月天数.xlsx')
    mn_str = df['月份']

    t = np.arange(1981, 2021)
    
    ind = 0
    for i in range(4):
        for j in range(3):
            ax2 = axs[i, j].twinx()
            #ax2.scatter(t2, data2[:, ind], c='Green', s=20) #
            #lin1 = ax2.plot(t2, data2[:, ind], color='Green', label="NDVI") #
            lin1 = ax2.bar(t, data2[ind, :], color='purple', label="ET", alpha=0.5) #
            
            axs[i, j].scatter(t, data1[ind, :], c='k', s=20) #
            lin2 = axs[i, j].plot(t, data1[ind, :], c='k', label="SPEI03", zorder=2) #
            
            axs[i, j].text(1980.5, 70, f"{mn_str[ind]}", fontsize=15) #
            ind += 1
            axs[i, j].set_ylim(0, 80)
            axs[i, j].set_yticks(np.arange(0, 81, 25))
            axs[i, j].tick_params(labelsize=15)
            '''
            ax2.set_ylim(-60, 60)
            ax2.set_yticks(np.arange(-60, 61, 20))
            '''
            ax2.set_ylim(-20, 20)
            ax2.set_yticks(np.arange(-20, 21, 5))
            ax2.tick_params(axis='y', labelsize=15, colors="purple")
            ax2.spines["right"].set_color("purple")
            if j!=2:
                ax2.get_yaxis().set_visible(False)
    '''
    #合并图例
    lins=lin1+lin2
    labs=[l.get_label() for l in lins]
    ax.legend(lins,labs,loc="upper right", fontsize=15)
    '''
    plt.ylabel("Transpiration (mm/month)", fontsize=15)
    plt.suptitle(f"{title}", fontsize=25)
    plt.savefig(rf'E:/Gleamv3.6a/JPG MG/{title}.jpg', 
                bbox_inches='tight')
    plt.show()
    
#%%
read_nc()
read_nc2()
read_nc3()

e_mn = mn_yr(e)
e_MG, _, _ = pre_xarray(e_mn)
e_ave = mask(130, e_MG)

eanom = mn_yr(e_anom)
eanom_MG, _, _ = pre_xarray(eanom)
eanom_ave = mask(130, eanom_MG)

print(np.nanmax(e_ave), np.nanmin(e_ave))
print(np.nanmax(eanom_ave), np.nanmin(eanom_ave))

plot(e_ave, eanom_ave, f"MG Grassland T new")