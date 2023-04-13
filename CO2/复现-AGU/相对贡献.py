# %%

import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.stats import linregress

import FittingReadDataGlobal as frd
import Senplot as sp

# plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体


# %%
frd.data_time(2001, 2018, 1, 12)
sif, pre, tmp, vpd, sr, co2 = frd.data_exact(2001, 2018, 1, 12)

_ = np.load("/mnt/e/CO2/复现-AGU/TemAbove0.npz")
sm, em = _["sm"], _["em"]

var1 = ["co2", "sif"]
var0 = ["pre", "tmp", "vpd", "sr"]

def gs1(ind, data):
    data_gs = np.empty((18, 360, 720))
    for r in range(360):
        for c in range(720):
            sm1, em1 = int(sm[r, c]), int(em[r, c])+1
            data_gs[:, r, c] = np.nanmean(data[:, sm1:em1, r, c], axis=ind)

    return data_gs

def gs0(ind, data):
    data_gs = np.empty((18, 360, 720))
    for r in range(360):
        for c in range(720):
            sm1, em1 = int(sm[r, c]), int(em[r, c])+1
            data_gs[:, r, c] = np.nanmean(data[sm1:em1, :, r, c], axis=ind)

    return data_gs


def trend(data):
    s, p = np.zeros((360, 720)), np.zeros((360, 720))
    t = np.arange(18)
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
def plot_sif(data1, var):
    print(var)

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    labeltitle = f"W m-2 um-1 sr-1 \n per decade"
    if var == "GOSIF":
        levels = [-0.01, 0, 0.01]
        labels = [">0.01", "0 - 0.01", "-0.01-0", "<-0.01"]
        counts = sp.cut(data1, -0.01, 0.01)

    lat = np.linspace(-89.75, 89.75, 360)
    lon = np.linspace(-179.75, 179.75, 720)

    colors = ["#104C73", "#38A700","#FFFF00", "#A90000"]
    title = f"{var} Trend"
    sp.CreatMap(data1, lon, lat, levels, colors[::-1],
                labels, labeltitle, counts, title)



lcc = np.load("/mnt/e/MODIS/MCD12C1/MCD12C1_vegnonveg_0.5X0.5.npy")

for x in var1:
    exec(f"{x}_gs = gs1(1, {x})")

for x in var0:
    print(x)
    exec(f"{x}_gs = gs0(0, {x})")

# s, p = trend(sif_gs)
# plot_sif(s, "GOSIF")

# %%
def read_nc():
    inpath = ('/mnt/e/CO2/复现-AGU/FittingGlobalSensitivity.nc')
    with nc.Dataset(inpath, mode='r') as f:

        # print(f.variables.keys(), "\n")
        # print(f.variables['fit_all'], "\n")

        vpd = (f.variables['vpd_sen'][:]).data  # units: hPa
        tmp = (f.variables['tmp_sen'][:]).data  # units: Celsius degree
        co2 = (f.variables['co2_sen'][:]).data  # units: ppm()
        # co2 = co2

        return vpd, tmp, co2

# %%


def read_plot(data1, var):
    print(var)

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    labeltitle = f"W m-2 um-1 sr-1 \n per decade"
    if var == "vpd":
        levels = [-0.002, 0, 0.002]
        labels = [">0.002", "0 - 0.002", "-0.002-0", "<-0.002"]
        counts = sp.cut(data1, -0.002, 0.002)
    elif var == "tmp":
        levels = [-0.002, 0, 0.002]
        labels = [">0.002", "0 - 0.002", "-0.002-0", "<-0.002"]
        counts = sp.cut(data1, -0.002, 0.002)
    elif var == "co2":
        levels = [-0.01, 0, 0.01]
        labels = [">0.01", "0 - 0.01", "-0.01-0", "<-0.01"]
        counts = sp.cut(data1, -0.01, 0.01)

    lat = np.linspace(-89.75, 89.75, 360)
    lon = np.linspace(-179.75, 179.75, 720)

    colors = ["#104C73", "#0085AB", "#37A800", "#A90000"]
    title = f"GOSIF change by {var}"
    sp.CreatMap(data1, lon, lat, levels, colors,
                labels, labeltitle, counts, title)




# %%
if __name__ == "__main__":
    sen_vpd, sen_tmp, sen_co2 = read_nc()
    var = ["vpd", "tmp", "co2"]
    for x in var:
        exec(f"s_{x}, _ = trend({x}_gs)")
        exec(f"con_{x} = s_{x}*sen_{x}")
        # exec(f"read_plot(con_{x}, x)")

    np.savez("Contribution", vpd=con_vpd, tmp=con_tmp, co2=con_co2)


# %%
