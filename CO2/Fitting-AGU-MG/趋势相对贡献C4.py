# %%
import cmaps
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.stats import linregress
import xarray as xr

import ReadDataGlobal as rd
import Senplot as sp
import CartopyPlot as cp

# %%
# gpp, sif, pre, pet, tmp, vpd, sr, co2, spei
gpp, _, _, _, tmp, _, _, co2, spei \
    = rd.data_exact(1982, 2018, 1, 12)


def read_nc(yr, var):
    inpath = rf"/mnt/e/GLASS-GPP/8-day MG 0.5X0.5 (phenology) /dbl_derive_{yr}.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        var = var.upper()
        data = (f.variables[f'{var}'][:]).data
        data[data == -9999] = np.nan
    return data


def data(var):
    var_all = []
    for yr in range(1982, 2019):
        var1 = read_nc(yr, var)
        var_all.append(var1)

    var_all = np.array(var_all)

    var_m = np.nanmean(var_all, axis=0)

    return var_m, var_all


_, sos = data("sop2")
# %% 提取每个格点生长季
_ = np.load("/mnt/e/CO2/复现-AGU/TemAbove0.npz")
sm, em = _["sm"], _["em"]

var1 = ["gpp"]
var0 = ["spei", "tmp", "co2"]

# %%


def sif_xarray(band1):
    mn = np.arange(37)
    lat = np.linspace(-89.75, 89.75, 360)
    lon = np.linspace(-179.75, 179.75, 720)
    sif = xr.DataArray(band1, dims=['mn', 'y', 'x'], coords=[mn, lat, lon])

    sif_MG = sif.loc[:, 40:55, 100:125]
    return np.array(sif_MG)


def gs1(ind, data):
    data_gs = np.empty((37, 360, 720))
    for r in range(360):
        for c in range(720):
            sm1, em1 = int(sm[r, c]), int(em[r, c])+1
            data_gs[:, r, c] = np.nanmean(data[:, sm1:em1, r, c], axis=ind)

    data_MG = sif_xarray(data_gs)

    return data_MG


def gs0(ind, data):
    data_gs = np.empty((37, 360, 720))
    for r in range(360):
        for c in range(720):
            sm1, em1 = int(sm[r, c]), int(em[r, c])+1
            data_gs[:, r, c] = np.nanmean(data[sm1:em1, :, r, c], axis=ind)

    data_MG = sif_xarray(data_gs)

    return data_MG


for x in var1:
    exec(f"{x}_gs = gs1(1, {x})")

for x in var0:
    print(x)
    exec(f"{x}_gs = gs0(0, {x})")

del sm, em


# %%


def trend(data):
    s, p = np.zeros((30, 50)), np.zeros((30, 50))
    t = np.arange(37)
    for r in range(30):
        if r % 30 == 0:
            print(f"{r} is done!")
        for c in range(50):
            a = data[:, r, c]
            if np.isnan(a).any():
                s[r, c], p[r, c] = np.nan, np.nan
            else:
                s[r, c], _, _, p[r, c], _ = linregress(t, a)
    
    return s*10, p

# %%


def read_nc():
    inpath = ('./FittingGlobalSensitivityC4.nc')
    with nc.Dataset(inpath, mode='r') as f:

        # print(f.variables.keys(), "\n")
        # print(f.variables['fit_all'], "\n")

        spei = (f.variables['spei_sen'][:]).data  # units: 无
        tmp = (f.variables['tmp_sen'][:]).data  # units: Celsius degree
        co2 = (f.variables['co2_sen'][:]).data  # units: ppm()
        co2 = co2*10
        sos = (f.variables['sos_sen'][:]).data

        return spei, tmp, co2, sos

# %%

def read_plot(data1, data2, var):
    
    print(var)

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(-2.5, 2.51, 0.25)
    
    # if var == "spei":
    #     levels = np.arange(-1.5, 1.51, 0.15)
    # elif var == "tmp":
    #     levels = np.arange(-0.2, 0.21, 0.02)
    # elif var == "co2":
    #     levels = np.arange(-2.5, 2.51, 0.25)
    # elif var == "sos":
    #     levels = np.arange(-0.25, 0.251, 0.025)
    
    var = var.upper()
    cmap = cmaps.MPL_BrBG
    title = f"GPP change by {var}"
    cp.plot(data1, data2, levels, cmap, title)


# %%
if __name__ == "__main__":
    sen_spei, sen_tmp, sen_co2, sen_sos = read_nc()
    var0.append("sos")
    sos_gs = sos.copy()
    for x in var0:
        exec(f"s_{x}, _ = trend({x}_gs)")
        exec(f"con_{x} = s_{x}*sen_{x}")

    np.savez("Contribution", spei=con_spei, tmp=con_tmp, co2=con_co2, sos=con_sos)

# %%
for x in var0:
    exec(f"read_plot(con_{x}, x, x)")
# %%
