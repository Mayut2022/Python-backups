# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:20:24 2022

@author: MaYutong
"""
import cmaps
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
def pre_xarray(band1):
    t = np.arange(40)
    mn = np.arange(1, 13, 1)
    pre=xr.DataArray(band1,dims=['mn', 't', 'y','x'],coords=[mn, t, lat2, lon2])
    pre_MG = pre.loc[:, :, 40:55, 100:125]
    lat_MG = pre_MG.y
    lon_MG = pre_MG.x
    pre_MG = np.array(pre_MG)
    return pre_MG, lat_MG, lon_MG



#%%
def read_nc():
    global lcc, lat, lon
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
        

#%%
def read_nc2():
    global lat2, lon2
    inpath = r"E:/Gleamv3.6a/Data/global/E_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        e = (f.variables['E'][:])
        e_mn = mn_yr(e)
        e_MG, _, _ = pre_xarray(e_mn)
        e_gs = np.nanmean(e_MG[4:9, 1:, ], axis=0)
        e_ave = mask(130, e_gs)
    return e_ave



#%%
def read_nc3():
    global lat2, lon2
    inpath = r"E:/Gleamv3.6a/Data/global/Et_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5.nc"
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        e = (f.variables['Et'][:])
        e_mn = mn_yr(e)
        e_MG, _, _ = pre_xarray(e_mn)
        e_gs = np.nanmean(e_MG[4:9, 1:, ], axis=0)
        e_ave = mask(130, e_gs)
    return e_ave
        
# %% mask数组


def mask(x, data):
    sp = data.shape[0]
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((sp, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for i in range(sp):
        a = data[i, :, :]
        a = ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave


# %%


def plot(data1, data2, title):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=500, sharex=True)

    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.9, wspace=None, hspace=0.2)

    t_date = pd.date_range(f'1982', periods=39, freq="YS")
    tt = []
    for j, x in enumerate(t_date):
        tt.append(x.strftime("%Y"))
    t2 = np.arange(39)

    for i in range(2):
        ax = axes[i]
        ax.axhline(y=0, c="k", linestyle="--")
        ax.axvline(17, color="b", linewidth=2, zorder=3)
        
        ##########################
        if i == 0:
            ax.text(17, 57, "1999(drought turning point)", c='b', fontsize=15)
            # 渐变色柱状图
            # 归一化
            norm = plt.Normalize(-80, 80)  # 值的范围
            norm_values = norm(data1)
            cmap = cmaps.MPL_PRGn[:]
            map_vir = cm.get_cmap(cmap)
            colors = map_vir(norm_values)

            ax.bar(t2, data1, color=colors, edgecolor='lightgray',
                   label="SPEI03", zorder=1)
            ax.tick_params(labelsize=15)
            ax.set_ylabel("units: mm/month", fontsize=15)
            ax.set_ylim(20, 60)

            ax.set_xticks(t2[::3])
            ax.set_xticklabels(tt[::3])
            ax.set_title("ET", loc="left", fontsize=15)
        ##########################
        elif i == 1:
            ax.text(17, 57, "1999(drought turning point)", c='b', fontsize=15)
            ax = axes[i]
            # 渐变色柱状图
            # 归一化
            norm = plt.Normalize(-80, 80)  # 值的范围
            norm_values = norm(data2)
            cmap = cmaps.MPL_PRGn[:]
            map_vir = cm.get_cmap(cmap)
            colors = map_vir(norm_values)

            ax.bar(t2, data2, color=colors,
                   edgecolor='lightgray', label="LAI", zorder=1)
            ax.tick_params(labelsize=15)
            ax.set_ylabel("units: mm/month", fontsize=15)
            ax.set_ylim(20, 60)
            ax.set_title("T", loc="left", fontsize=15)
    # label
    # handles1, labels1 = ax.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    # ax.legend(handles1+handles2, labels1+labels2, ncol=2, fontsize=15, loc="upper left")

    # plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/Gleamv3.6a/JPG MG/{title}.jpg',
              bbox_inches='tight')
    plt.show()
    
#%%
read_nc()
e_ave = read_nc2()
et_ave = read_nc3()

#%%
print(np.nanmax(e_ave), np.nanmin(e_ave))
print(np.nanmax(et_ave), np.nanmin(et_ave))




# %%
plot(e_ave, et_ave, title=r"MG Grassland ET&T Original")