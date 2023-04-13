# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:44:19 2023

@author: MaYutong
"""

import cftime
import netCDF4 as nc
import numpy as np
import pandas as pd
import xlsxwriter

# %%


def read_lcc():
    global lcc, lat, lon
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys()) 显示所有变量
        # print(f.variables['lccs_class']) 详细显示某种变量信息
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])


read_lcc()

# %%


def read_nc():
    global spei, t, lat, lon
    inpath = (rf"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        spei = (f.variables['spei'][960:])
        time = (f.variables['time'][960:])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        t = nc.num2date(time, 'days since 1900-01-01 00:00:0.0').data


# %%
def index(sy, sm, ey, em):
    tt = t.tolist()
    if sm == 2 and em == 2:
        i1 = tt.index(cftime.DatetimeGregorian(
            sy, sm, 15, 0, 0, 0, 0, has_year_zero=False))
        i2 = tt.index(cftime.DatetimeGregorian(
            ey, em, 15, 0, 0, 0, 0, has_year_zero=False))
    elif sm == 2:
        i1 = tt.index(cftime.DatetimeGregorian(
            sy, sm, 15, 0, 0, 0, 0, has_year_zero=False))
        i2 = tt.index(cftime.DatetimeGregorian(
            ey, em, 16, 0, 0, 0, 0, has_year_zero=False))
    elif em == 2:
        i1 = tt.index(cftime.DatetimeGregorian(
            sy, sm, 16, 0, 0, 0, 0, has_year_zero=False))
        i2 = tt.index(cftime.DatetimeGregorian(
            ey, em, 15, 0, 0, 0, 0, has_year_zero=False))
    else:
        i1 = tt.index(cftime.DatetimeGregorian(
            sy, sm, 16, 0, 0, 0, 0, has_year_zero=False))
        i2 = tt.index(cftime.DatetimeGregorian(
            ey, em, 16, 0, 0, 0, 0, has_year_zero=False))

    #print(i1, i2)
    return i1, i2

# %% mask数组


def mask(x, data):
    lcc2 = np.array(lcc)

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((480, 30, 50))
    spei_ma = np.ma.array(spei_ma)

    for l in range(480):
        a = data[l, :, :]
        a = np.ma.masked_array(a, mask=lcc2, fill_value=-999)
        spei_ma[l, :, :] = a.filled()

    return spei_ma


# %%
level = dict(dry1="x <= -2",
             dry2="x>-2 and x<=-1.5",
             dry3="x>-1.5 and x<=-1")


def drought_severity():
    bin = [-5, -2, -1.5, -1]
    labels = ["Severe", "Moderate", "Mild"]
    ind = 0
    for a, b, c, d in zip(df['SY'], df['SM'], df['EY'], df['EM']):
        ind1, ind2 = index(a, b, c, d)
        spei_g = mask(130, spei)
        spei_g = spei_g[ind1:ind2+1, :, :]

        # 删掉mask的值
        spei_ds = spei_g.reshape(1, 30*50*spei_g.shape[0])
        spei_ds = np.squeeze(spei_ds)

        sp = 383*spei_g.shape[0]
        ds = pd.cut(spei_ds, bin, right=True, labels=labels)
        ds_count = ds.value_counts()
        ds_count['sum'] = ds_count.sum()
        ds_per = ds.value_counts()/sp
        ds_per['sum'] = ds_per.sum()
        ds_once = pd.concat([ds_count, ds_per])
        print(ds_once, "\n")
        if ind == 0:
            ds_all = ds_once
        else:
            ds_all = pd.concat([ds_all, ds_once], axis=1)
        ind = ind+1
    return ds_all


# %%
read_nc()
df = pd.read_excel("E:/SPEI_base/python MG/MG Grassland干旱统计.xlsx")
ds_all = drought_severity()
ds_all = np.transpose(ds_all)
ds_all.sort_index(axis=1, inplace=True)
ds_all.index = range(20)


# %%

writer = pd.ExcelWriter(
    "E:/SPEI_base/python MG/MG Grassland干旱统计.xlsx", mode='a', engine='xlsxwriter')
ds_all.to_excel(writer, sheet_name="drought severity", index=False)
writer.save()
