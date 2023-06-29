# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:09:30 2023

@author: MaYutong
"""


import warnings
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
import xlsxwriter

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
warnings.filterwarnings("ignore")
# %%


def read_lcc():
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

    return lcc


# %%


def read_LAI():
    global lat, lon
    # inpath = (r"E:/LAI4g/data_MG/LAI_Detrend_Anomaly_82_20_MG_SPEI0.5x0.5.nc")
    inpath = (r"E:/LAI4g/data_MG/LAI_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gs = np.nanmean(lai, axis=1)
        lai_diff = lai_gs[1:,]-lai_gs[:-1,]

        lai_std = np.nanstd(lai_diff, axis=0)
        
        lai_std_all = []
        for i in range(38):
            lai_std_all.append(lai_std)
            
        lai_std_all = np.array(lai_std_all)
            
    return lai_diff, lai_std_all

# %%


def read_spei_1():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][960:, :, :])
        spei = spei.reshape(40, 12, 30, 50)
        spei_gsl = spei[:, 4:9, :]
        spei_gs = np.nanmean(spei_gsl, axis=1)
        spei_diff = spei_gs[1:,]-spei_gs[:-1,]

    return spei_diff[1:]



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
lcc = read_lcc()
lai1, lai2 = read_LAI()
spei1 = read_spei_1()



# %%

lcc2 = lcc.copy()
lon2, lat2 = np.meshgrid(lon, lat)

var1 = ["lcc2", "lon2", "lat2"]
for v in var1:
    exec(f"{v}={v}.reshape(1500)")

var2 = ["lai1", "lai2", "spei1"]
for v in var2:
    exec(f"{v}_re={v}.reshape(38, 1500)")
    exec(f"{v}_re = np.vstack(({v}_re, lcc2))")
    index = pd.Index(list(np.arange(1983, 2021))+["LCC"], name="Year")
    columns = pd.MultiIndex.from_arrays([lat2, lon2], names=('Lat', 'Lon'))
    exec(
        f"{v}_df = pd.DataFrame({v}_re, index=index, columns=columns)")
    exec(f"{v}_df[{v}_df.isnull()]=-999")
    exec(f"{v}_df.loc['LCC'][{v}_df.loc['LCC']!=130]=np.nan")
    exec(f"{v}_df = {v}_df.dropna(axis=1)")
    exec(f"{v}_df = {v}_df.drop(['LCC'])")
    exec(f"{v}_df2 = {v}_df.stack().stack()")

# %%
data = pd.concat([lai1_df2, lai2_df2, spei1_df2], axis=1)
data.columns = ["LAI Diff", "LAI Std", "SPEI Diff"]

#%% 从LAI出发进行筛选， 筛选植被显著增长年份
df_wet = data.copy()
df_wet["LAI Diff"][df_wet["LAI Diff"]==-999]=np.nan

ind = df_wet["LAI Diff"]>=df_wet["LAI Std"]
df_wet["LAI Diff"][~ind] = np.nan

df_wet = df_wet.dropna(axis=0)
df_wet.reset_index(inplace=True)

#%% 从LAI出发进行筛选， 筛选植被显著退化年份
df_dry = data.copy()
df_dry["LAI Diff"][df_dry["LAI Diff"]==-999]=np.nan

ind = df_dry["LAI Diff"]<=df_dry["LAI Std"]*-1
df_dry["LAI Diff"][~ind] = np.nan

df_dry = df_dry.dropna(axis=0)
df_dry.reset_index(inplace=True)


#%% 保存 excel数据

dfs = {'Sig Up':df_wet, 'Sig Down':df_dry}
writer = pd.ExcelWriter('LAI Sig Change2.xlsx', engine='xlsxwriter')
 
for sheet_name in dfs.keys():
    dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
    
writer.save()
