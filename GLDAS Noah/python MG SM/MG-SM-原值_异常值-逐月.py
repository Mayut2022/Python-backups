# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:07:17 2022

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
def read_nc2():
    global sm, lat2, lon2
    inpath = rf"E:/GLDAS Noah/DATA_RG/SM_81_20_ORINGINAL_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath, mode='r') as f:
        
        #print(f.variables.keys())
        
        sm = (f.variables['sm'][:, :, :])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        
#%%
def read_nc3():
    global sm_a
    inpath = rf"E:/GLDAS Noah/DATA_RG/SM_Anom_81_20_ORINGINAL_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath, mode='r') as f:
        
        #print(f.variables.keys())
        
        sm_a = (f.variables['sm'][:, :, :])

        
# %%
def MG(data):
    t = np.arange(1, 481, 1)
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
    
    spei_ma = np.empty((480, 4, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(480):
        for l in range(4):
            a = data[i, l, :, :]
            a = ma.masked_array(a, mask=lcc2)
            spei_ma[i, l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(2, 3))
    
    return spei_ma_ave


#%%
def plot(data1, data2, layer, title):
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
            lin1 = ax2.bar(t, data2[ind, :], color='brown', label="SM", alpha=0.5) #
            
            axs[i, j].scatter(t, data1[ind, :], c='k', s=20) #
            lin2 = axs[i, j].plot(t, data1[ind, :], c='k', label="SPEI03", zorder=2) #
            
            if layer==1:
                axs[i, j].set_ylim(0, 40)
                axs[i, j].set_yticks(np.arange(0, 41, 10))
                axs[i, j].text(1980.5, 35, f"{mn_str[ind]}", fontsize=15) #
            elif layer==2:
                axs[i, j].set_ylim(0, 80)
                axs[i, j].set_yticks(np.arange(0, 81, 20))
                axs[i, j].text(1980.5, 70, f"{mn_str[ind]}", fontsize=15) #
            elif layer==3:
                axs[i, j].set_ylim(40, 120)
                axs[i, j].set_yticks(np.arange(40, 121, 20))
                axs[i, j].text(1980.5, 110, f"{mn_str[ind]}", fontsize=15) #
            elif layer==4:
                axs[i, j].set_ylim(160, 240)
                axs[i, j].set_yticks(np.arange(160, 241, 20))
                axs[i, j].text(1980.5, 230, f"{mn_str[ind]}", fontsize=15) #
            ind += 1
            
            axs[i, j].tick_params(labelsize=15)
            
            '''
            ax2.set_ylim(-60, 60)
            ax2.set_yticks(np.arange(-60, 61, 20))
            '''
            ax2.set_ylim(-40, 40)
            ax2.set_yticks(np.arange(-40, 41, 10))
            ax2.tick_params(axis='y', labelsize=15, colors="brown")
            ax2.spines["right"].set_color("brown")
            if j!=2:
                ax2.get_yaxis().set_visible(False)
    '''
    #合并图例
    lins=lin1+lin2
    labs=[l.get_label() for l in lins]
    ax.legend(lins,labs,loc="upper right", fontsize=15)
    '''
    plt.ylabel("Soil Moisture (kg m-2 month mean)", fontsize=15)
    plt.suptitle(f"{title}", fontsize=25)
    plt.savefig(rf'E:/GLDAS Noah/JPG_MG/{title}.jpg', 
                bbox_inches='tight')
    plt.show()
        
#%%
read_nc()
read_nc2()
read_nc3()

sm_MG = MG(sm)
sm_a_MG = MG(sm_a)

sm_MG_ave = mask(130, sm_MG)
sm_a_MG_ave = mask(130, sm_a_MG)

#%%
for l in range(4):
    print(np.nanmax(sm_MG_ave[:, l]), np.nanmin(sm_MG_ave[:, l]))
    print(np.nanmax(sm_a_MG_ave[:, l]), np.nanmin(sm_a_MG_ave[:, l]), "\n")
    
    data1 = sm_MG_ave[:, l].reshape(40, 12).T
    data2 = sm_a_MG_ave[:, l].reshape(40, 12).T
    plot(data1, data2, l+1, f"MG Grassland SM Layer{l+1}")
    
#%%
import matplotlib.pyplot as plt

ind = 6
for yr in range(36):
    
    data = sm[yr, 1, :, :]
    print(np.nanmax(data), np.nanmin(data))
    
    
    plt.figure(3, dpi=500)
    plt.imshow(data, cmap='Blues', vmin=0, vmax=120, origin="lower")
    plt.title(f"{yr+1}")
    plt.colorbar(shrink=0.75)
    plt.show()