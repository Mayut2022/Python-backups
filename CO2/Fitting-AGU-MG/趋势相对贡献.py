# %%

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.stats import linregress

import ReadDataGlobal as rd
import Senplot as sp
# %%
# gpp, sif, pre, pet, tmp, vpd, sr, co2, spei
gpp, _, pre, pet, tmp, _, _, co2, spei \
    = rd.data_exact(1982, 2018, 1, 12)


# %% 提取每个格点生长季
_ = np.load("/mnt/e/CO2/复现-AGU/TemAbove0.npz")
sm, em = _["sm"], _["em"]

var1 = ["gpp"]
var0 = ["spei", "tmp", "co2"]

# %%


def gs1(ind, data):
    data_gs = np.empty((37, 360, 720))
    for r in range(360):
        for c in range(720):
            sm1, em1 = int(sm[r, c]), int(em[r, c])+1
            data_gs[:, r, c] = np.nanmean(data[:, sm1:em1, r, c], axis=ind)

    return data_gs


def gs0(ind, data):
    data_gs = np.empty((37, 360, 720))
    for r in range(360):
        for c in range(720):
            sm1, em1 = int(sm[r, c]), int(em[r, c])+1
            data_gs[:, r, c] = np.nanmean(data[sm1:em1, :, r, c], axis=ind)

    return data_gs


for x in var1:
    exec(f"{x}_gs = gs1(1, {x})")

for x in var0:
    print(x)
    exec(f"{x}_gs = gs0(0, {x})")

del sm, em

# %%


def trend(data):
    s, p = np.zeros((360, 720)), np.zeros((360, 720))
    t = np.arange(37)
    for r in range(360):
        if r % 30 == 0:
            print(f"{r} is done!")
        for c in range(720):
            a = data[:, r, c]
            if np.isnan(a).any():
                s[r, c], p[r, c] = np.nan, np.nan
            else:
                s[r, c], _, _, p[r, c], _ = linregress(t, a)
    s[lcc == 0] = np.nan
    p[lcc == 0] = np.nan
    return s*10, p

# %%


def read_nc():
    inpath = ('./FittingGlobalSensitivityC1.nc')
    with nc.Dataset(inpath, mode='r') as f:

        # print(f.variables.keys(), "\n")
        # print(f.variables['fit_all'], "\n")

        spei = (f.variables['spei_sen'][:]).data  # units: hPa
        tmp = (f.variables['tmp_sen'][:]).data  # units: Celsius degree
        co2 = (f.variables['co2_sen'][:]).data  # units: ppm()
        co2 = co2*10

        return spei, tmp, co2

# %%


def read_plot(data1, var):
    print(var)

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    labeltitle = f"gC m-2 \n per decade"
    if var == "spei":
        levels = [-0.5, 0, 0.5]
        labels = [">0.5", "0 - 0.5", "-0.5-0", "<-0.5"]
        counts = sp.cut(data1, -0.5, 0.5)
    elif var == "tmp":
        levels = [-0.05, 0, 0.05]
        labels = [">0.05", "0 - 0.05", "-0.05-0", "<-0.05"]
        counts = sp.cut(data1, -0.05, 0.05)
    elif var == "co2":
        levels = [-1, 0, 1]
        labels = [">1", "0 - 1", "-1-0", "<-1"]
        counts = sp.cut(data1, -1, 1)

    lat = np.linspace(-89.75, 89.75, 360)
    lon = np.linspace(-179.75, 179.75, 720)

    colors = ["#104C73", "#0085AB", "#37A800", "#A90000"]
    title = f"GPP change by {var}"
    sp.CreatMap(data1, lon, lat, levels, colors,
                labels, labeltitle, counts, title)


# %%
if __name__ == "__main__":
    lcc = np.load("/mnt/e/MODIS/MCD12C1/MCD12C1_vegnonveg_0.5X0.5.npy")
    sen_spei, sen_tmp, sen_co2 = read_nc()
    for x in var0:
        exec(f"s_{x}, _ = trend({x}_gs)")
        exec(f"con_{x} = s_{x}*sen_{x}")
    

    np.savez("Contribution", spei=con_spei, tmp=con_tmp, co2=con_co2)

# %%
for x in var0:
    exec(f"read_plot(con_{x}, x)")
# %%
