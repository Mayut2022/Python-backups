# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 21:55:20 2023

@author: MaYutong
"""

# %%
import netCDF4 as nc

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import xlsxwriter

import seaborn as sns

from scipy.stats import linregress
from scipy.stats import pearsonr

plt.rcParams['font.sans-serif'] = ['times new roman']

# %% 480(输入data) -> 月 年


def mn_yr(data):
    tmp_mn = []
    for mn in range(12):
        tmp_ = []
        for yr in range(40):
            tmp_.append(data[mn])
            mn += 12
        tmp_mn.append(tmp_)

    tmp_mn = np.array(tmp_mn)

    return tmp_mn

# %%


def read_nc():
    # global lat_g, lon_g
    inpath = r"E:/Gleamv3.6a/v3.6a/global/E_1980-2021_GLEAM_v3.6a_MO_SPEI_0.5X0.5.nc"
    with nc.Dataset(inpath, mode='r') as f:

        e = (f.variables['E'][:, :, :])
        lat_g = (f.variables['lat'][:])
        lon_g = (f.variables['lon'][:])

    def sif_xarray(band1):
        mn = np.arange(12)
        yr = np.arange(40)
        sif = xr.DataArray(band1, dims=['mn', 'yr', 'y', 'x'], coords=[
                           mn, yr, lat_g, lon_g])

        sif_MG = sif.loc[:, :, 40:55, 100:125]
        return np.array(sif_MG)

    e_mn = mn_yr(e)
    e_mn_MG = sif_xarray(e_mn)

    return e_mn_MG

# %%


def read_nc2(scale):
    inpath = (rf"E:/SPEI_base/data/spei{scale}.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        spei = (f.variables['spei'][960:])
        time = (f.variables['time'][960:])
        t2 = nc.num2date(time, 'days since 1900-01-01 00:00:0.0').data
        lat_g = (f.variables['lat'][:])
        lon_g = (f.variables['lon'][:])

    def sif_xarray(band1):
        mn = np.arange(12)
        yr = np.arange(40)
        sif = xr.DataArray(band1, dims=['mn', 'yr', 'y', 'x'], coords=[
                           mn, yr, lat_g, lon_g])

        sif_MG = sif.loc[:, :, 40:55, 100:125]
        return np.array(sif_MG)

    spei_mn = mn_yr(spei)
    spei_mn_MG = sif_xarray(spei_mn)

    return spei_mn_MG

# %%


def read_nc3():
    global lcc, lat, lon
    # global a1, a2, o1, o2
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

# %%


def corr(data1, data2, df, scale):
    r = np.zeros((30, 50))
    p = np.zeros((30, 50))
    for i in range(30):
        for j in range(50):
            if np.isnan(data1[:, i, j]).any() or np.isnan(data2[:, i, j]).any():
                r[i, j], p[i, j] = np.nan, np.nan
            else:
                r[i, j], p[i, j] = pearsonr(data1[:, i, j], data2[:, i, j])

    r = r.reshape(30*50)
    p = p.reshape(30*50)
    df1 = eval(f"pd.DataFrame(dict(R{scale}=r, P{scale}=p))")
    df = pd.concat([df, df1], axis=1)

    return r, p, df1, df

# %%
def sub_ax(fig, pos, data, ind):
    ax1 = fig.add_axes(pos)
    
    x = np.arange(1, 13)
    x_str = [str(xx).zfill(2) for xx in x]
    cmap = plt.get_cmap('RdBu')
    newcolors = cmap(np.linspace(0, 1, 12))
    ax1.bar(x, data, color=newcolors, width=0.7)
    ax1.tick_params(labelsize=12)
    ax1.set_xticks(np.arange(1, 13))
    ax1.set_xticklabels(x_str)
    if ind==1:
        ax1.set_ylim(0.2, 0.7)
        ax1.set_yticks(np.arange(0.2, 0.71, 0.2))
    else:
        ax1.set_ylim(0.2, 1)
        ax1.set_yticks(np.arange(0.2, 1.1, 0.2))
        
    return ax1
    
def sns_plot(df_g, title):
    
    fig, axes = plt.subplots(1, figsize=(10, 10), dpi=500)
    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95,
                        top=0.93, wspace=0.1, hspace=0.15)
    
    rc = {'font.sans-serif': 'times new roman'}
    sns.set(rc=rc, style="ticks")
    
    cmap = plt.get_cmap('RdBu')
    newcolors = cmap(np.linspace(0, 1, 12))
    
    data = df_g.iloc[:, 3::2]
    p_v = df_g.iloc[:, 4::2]
    
    ### 平均值
    mean = []
    area = []
    for i in range(12):
        ave = data.iloc[:, i].mean()
        mean.append(ave)
        num = (p_v.iloc[:, i] < 0.05).value_counts()
        num = num[True]/len(data.iloc[:, i])
        area.append(num)
        # axes.axvline(x=ave, color=newcolors[i, :])
        
    ### 子图平均值和面积
    ax1 = sub_ax(fig, [0.1, 0.75, 0.3, 0.15], mean, 1)
    ax1.set_title("Corr Average", loc="left")
    ax2 = sub_ax(fig, [0.1, 0.55, 0.3, 0.15], area, 2)
    ax2.set_title("Significant Area Percent", loc="left")
    
    
    # ax2 = sub_ax([0.1, 0.55, 0.3, 0.15])
    
    #######
    # sub_ax = fig.add_axes([0.1, 0.75, 0.3, 0.3])
    # sub_ax = sub_plot()
    
    #######
    # sub_ax = fig.add_subplot(2, 1, figsize=(4.5, 3), dpi=500, sharex=True)
    # x = np.arange(1, 13)
    # sub_ax[0].bar(x, mean, color=newcolors)
    
        
    
    sns.kdeplot(data=data, palette=newcolors, ax=axes)
    axes.tick_params(labelsize=15)
    axes.set_xlim(-1.1, 1.1)
    axes.set_xticks(np.arange(-1, 1.1, 0.2))
    axes.set_ylim(0, 0.3)
    axes.set_yticks(np.arange(0, 0.31, 0.025))
    axes.set_xlabel("Correlation coefficient", fontsize=20)
    axes.set_ylabel("Density", fontsize=20)
    
    
    plt.suptitle(f'{title}', fontsize=25)
    plt.savefig(rf'E:/SPEI_base/python MG/JPG/响应时间分段/{title}.jpg',
                bbox_inches='tight')
    plt.show()


# %%
e_MG = read_nc()
read_nc3()


def origin_df():
    lON, lAT = np.meshgrid(lon, lat)
    lcc_re, lon_re, lat_re = lcc.reshape(
        30*50), lON.reshape(30*50), lAT.reshape(30*50)
    dic = dict(LCC=lcc_re, LAT=lat_re, LON=lon_re)
    df = pd.DataFrame(dic)

    return df


df = origin_df()

# %%
'''
scale = [str(x).zfill(2) for x in range(1, 13)]
month = np.arange(4, 11, 1)  # Apr-Oct 生长季
df2 = pd.read_excel('E:/ERA5/每月天数.xlsx')
mn_str = df2['月份']

for mn in month[2:5]:
    print(f"month {mn} is in programming!")
    e_mn1 = e_MG[mn-1, 1:19, :, :]
    e_mn2 = e_MG[mn-1, 19:38, :, :]
    df = origin_df()
    for s in scale:
        spei_s = read_nc2(s)
        spei_mn1 = spei_s[mn-1, 1:19, :, :]
        spei_mn2 = spei_s[mn-1, 19:38, :, :]
        r, p, _, df = corr(e_mn1, spei_mn1, df, s)
        # r, p, _, df = corr(e_mn2, spei_mn2, df, s)
        print(f"scale spei{s} is done!")

    print(f"Next is sns plot-----")
    df_g = df[df["LCC"] == 130]
    sns_plot(df_g, title=f"ET-SPEI CORR 82-99 {mn_str[mn-1]}")
    # sns_plot(df_g, title=f"ET-SPEI CORR 00-18 {mn_str[mn-1]}")
'''

# %% 夏季
scale = [str(x).zfill(2) for x in range(1, 13)]
e_MG_s = e_MG[5:8, :].mean(axis=0)  # 夏季
e_mn1 = e_MG_s[1:19, :, :]
e_mn2 = e_MG_s[19:38, :, :]
df = origin_df()
for s in scale:
    spei_s = read_nc2(s)
    spei_s = spei_s[5:8, :].mean(axis=0)  # 夏季
    spei_mn1 = spei_s[1:19, :, :]
    spei_mn2 = spei_s[19:38, :, :]
    # r, p, _, df = corr(e_mn1, spei_mn1, df, s)
    r, p, _, df = corr(e_mn2, spei_mn2, df, s)
    print(f"scale spei{s} is done!")

print(f"Next is sns plot-----")
df_g = df[df["LCC"] == 130]
# sns_plot(df_g, title=f"ET 82-99 Summer")
sns_plot(df_g, title=f"ET 00-18 Summer")


