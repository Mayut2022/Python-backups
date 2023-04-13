# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 19:58:59 2022

@author: MaYutong
"""
import cmaps
import datetime as dt
import netCDF4 as nc
from matplotlib import font_manager
import matplotlib.dates as mdate
from matplotlib import cm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr

plt.rcParams['font.sans-serif']='times new roman'

# %%
def MG(data):
    t = np.arange(1, 481, 1)
    l = np.arange(1, 5, 1)
    spei_global = xr.DataArray(data, dims=['t', 'l', 'y', 'x'], coords=[
                               t, l, lat2, lon2])  # 原SPEI-base数据
    spei_rg1 = spei_global.loc[:, :, 40:55, 100:125]
    spei_rg1 = np.array(spei_rg1)
    return spei_rg1


#%%
def read_nc():
    global lat, lon
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
    return lcc
        
#%%
def read_nc2():
    global lat2, lon2
    inpath = rf"E:/GLDAS Noah/DATA_RG/SM_81_20_ORINGINAL_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath, mode='r') as f:
        #print(f.variables.keys())
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        sm = (f.variables['sm'][:, :, :])
        sm_MG = MG(sm)
        sm = sm_MG.reshape(40, 12, 4, 30, 50)
        sm_gsl = sm[:, 4:9, :]
        sm_gsl = np.nanmean(sm_gsl, axis=1)
        sm_diff = sm_gsl[1:, ]-sm_gsl[:-1, ]
    return sm_diff    
    
#%%
def read_nc3():
    inpath = rf"E:/GLDAS Noah/DATA_RG/SM_Anom_81_20_ORINGINAL_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath, mode='r') as f:
        
        #print(f.variables.keys())
        
        sm_a = (f.variables['sm'][:, :, :])
    return sm_a
        


#%% mask数组
def mask(x, data):
    sp = data.shape[0]
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((sp, 4, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(sp):
        for l in range(4):
            a = data[i, l, :, :]
            a = ma.masked_array(a, mask=lcc2)
            spei_ma[i, l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(2, 3))
    
    return spei_ma_ave


#%%
def plot(data1, title):
    fig, axs = plt.subplots(4, 1, figsize=(10, 16), dpi=150, sharex=True)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.93, wspace=0.05, hspace=0.15)
    
    t = np.arange(39)
    t_date = pd.date_range(f'1982', periods=39, freq="YS")
    tt = []
    for j, x in enumerate(t_date):
        tt.append(x.strftime("%Y"))
    
    for i in range(4):
        axs[i].axhline(y=0, c="k", linestyle="--")
        axs[i].axvline(18, color="b", linewidth=2, zorder=3)
        # 渐变色柱状图
        # 归一化
        norm = plt.Normalize(-8, 8)  # 值的范围
        norm_values = norm(data1[:, i])
        cmap = "bwr"
        map_vir = cm.get_cmap(cmap)
        colors = map_vir(norm_values)
        axs[i].bar(t, data1[:, i], color=colors, alpha=0.6, edgecolor='lightgray') #
        
        if i+1==1:
            axs[i].text(18, 4, "2000(drought turning point)", c='b', fontsize=20)
            axs[i].set_ylim(-5, 5)
            axs[i].set_yticks(np.arange(-4, 4.1, 2))
            axs[i].set_title(f"Layer1 (0-10 cm)", fontsize=20, loc="left") #
        elif i+1==2:
            axs[i].text(18, 8, "2000(drought turning point)", c='b', fontsize=20)
            axs[i].set_ylim(-10, 10)
            axs[i].set_yticks(np.arange(-9, 9.1, 3))
            axs[i].set_title(f"Layer2 (10-40 cm)", fontsize=20, loc="left") #
        elif i+1==3:
            axs[i].text(18, 16, "2000(drought turning point)", c='b', fontsize=20)
            axs[i].set_ylim(-20, 20)
            axs[i].set_yticks(np.arange(-18, 19, 6))
            axs[i].set_title(f"Layer3 (40-100 cm)", fontsize=20, loc="left") #
        elif i+1==4:
            axs[i].text(18, 6.5, "2000(drought turning point)", c='b', fontsize=20)
            axs[i].set_ylim(-8, 8)
            axs[i].set_yticks(np.arange(-8, 8.1, 4))
            axs[i].set_title(f"Layer4 (100-200 cm)", fontsize=20, loc="left") #
        
        axs[i].set_xticks(t[::6])
        axs[i].set_xticklabels(tt[::6])
        axs[i].tick_params(labelsize=20)
        axs[i].set_ylabel("units: kg/m2/GSL", fontsize=20)
    # plt.suptitle(f"{title}", fontsize=25)
    plt.savefig(rf'E:/GLDAS Noah/JPG_MG/{title}.jpg', 
                bbox_inches='tight')
    plt.show()
        
#%%
lcc = read_nc()
sm = read_nc2()
sm_ave = mask(130, sm)

#%%
for i in range(4):
    print(f"Layer{i+1}", np.nanmax(sm_ave[:,i]), np.nanmin(sm_ave[:,i]))
plot(sm_ave, f"MG Grassland SM Annual Diff")
    
