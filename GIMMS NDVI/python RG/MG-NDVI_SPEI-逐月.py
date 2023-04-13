# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:09:14 2022

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
from sklearn import preprocessing
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
    global spei, t
    inpath = (rf"E:/SPEI_base/data/spei03_MG_season.nc")
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        spei = (f.variables['spei'][[1, 2],80:, :,])

        
#%%
def read_nc3():
    global ndvi, lat2, lon2
    inpath = r"E:/GIMMS_NDVI/data_RG/NDVI_82_15_RG_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        ndvi = (f.variables['ndvi'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        
#%%
def ndvi_xarray(band1):
    t = np.arange(34)
    mn = np.arange(1, 13, 1)
    ndvi=xr.DataArray(band1,dims=['t','mn', 'y','x'],coords=[t, mn, lat2, lon2])
    ndvi_IM = ndvi.loc[:, :, 40:55, 100:125]
    lat_IM = ndvi_IM.y
    lon_IM = ndvi_IM.x
    ndvi_IM = np.array(ndvi_IM)
    return ndvi_IM, lat_IM, lon_IM

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

#%% mask数组
def mask2(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((34, 12, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(34):
        for l in range(12):
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
    fig, axs = plt.subplots(4, 3, figsize=(20, 12), dpi=150, sharey=True, sharex=True)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=0.05, hspace=0.15)
    
    df = pd.read_excel('E:/ERA5/每月天数.xlsx')
    mn_str = df['月份']

    t = np.arange(1981, 2021)
    t2 = np.arange(1982, 2016)
    
    ind = 0
    for i in range(4):
        for j in range(3):
            
            ax2 = axs[i, j].twinx()
            #ax2.scatter(t2, data2[:, ind], c='Green', s=20) #
            ax2.plot(t2, data2[:, ind], color='Green', label="NDVI") #
            ax2.scatter(t2, data2[:, ind], color='Green', alpha=0.5) #
            
            #渐变色柱状图
            #归一化
            norm = plt.Normalize(-2.3, 2.3) #值的范围
            norm_values = norm(data1[ind, :])
            map_vir = cm.get_cmap(name='bwr_r')
            colors = map_vir(norm_values)
            #axs[i, j].scatter(t, data1[ind, :], c='k', s=20) #
            axs[i, j].bar(t, data1[ind, :], color=colors, label="SPEI03", zorder=2) #
            
            axs[i, j].text(1980.5, 1.8, f"{mn_str[ind]}", fontsize=15) #
            ind += 1
            axs[i, j].set_ylim(-2.4, 2.4)
            axs[i, j].set_yticks(np.arange(-2, 2.1, 1))
            axs[i, j].tick_params(labelsize=15)
            
            
            ax2.set_ylim(-2, 2)
            ax2.set_yticks(np.arange(-2, 2.1, 1))
            ax2.tick_params(axis='y', labelsize=15, colors="Green")
            ax2.spines["right"].set_color("Green")
            if j!=2:
                ax2.get_yaxis().set_visible(False)
            
    line1, label1 = axs[0, 0].get_legend_handles_labels()
    line2, label2 = ax2.get_legend_handles_labels()
    lines = line1+line2
    labels = label1+label2
    fig.legend(lines, labels, loc = 'upper right', bbox_to_anchor=(0.95, 1), fontsize=20) 
    
    plt.suptitle(f"{title}", fontsize=25)
    plt.savefig(rf'E:/GIMMS_NDVI/JPG_RG/{title}.jpg', 
                bbox_inches='tight')
    plt.show()
 
#%% NDVI 标准化
def Zscore(data):
    data_z = np.zeros((34, 12, 30, 50))
    for mn in range(12):
        for r in range(30):
            for c in range(50):
                data_z[:, mn, r, c] = preprocessing.scale(data[:, mn, r, c])
    
    return data_z

#%% 逐月spei、ndvi标准化分布图分布图
'''
read_nc()
read_nc2()
spei_mn = mn_yr(spei)
read_nc3()
ndvi_IM, lat_IM, lon_IM = ndvi_xarray(ndvi)

spei_mn_ave = mask(130, spei_mn)
ndvi_IM_ave = mask2(130, ndvi_IM)

print(np.nanmax(spei_mn_ave), np.nanmin(spei_mn_ave))
print(np.nanmax(ndvi_IM_ave), np.nanmin(ndvi_IM_ave))

### NDVI标准化
ndvi_IM_z = Zscore(ndvi_IM)
ndvi_IM_z_ave = mask2(130, ndvi_IM_z)
print(np.nanmax(ndvi_IM_z_ave), np.nanmin(ndvi_IM_z_ave))


plot(spei_mn_ave, ndvi_IM_z_ave, f"MG Grassland SPEI03 Month")
'''

#%% spei 夏季、秋季分布图

read_nc()
read_nc2()



