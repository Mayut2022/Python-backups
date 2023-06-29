
# %%
import cmaps

import netCDF4 as nc
import numpy as np
import pandas as pd

import Senplot as sp
# %%


def read_nc():
    inpath = ('./FittingGlobalSensitivityC1.nc')
    with nc.Dataset(inpath, mode='r') as f:

        # print(f.variables.keys(), "\n")
        # print(f.variables['fit_all'], "\n")

        spei = (f.variables['spei_sen'][:]).data  # units: 无
        tmp = (f.variables['tmp_sen'][:]).data  # units: Celsius degree
        co2 = (f.variables['co2_sen'][:]).data  # units: ppm()
        co2 = co2*10

        return spei, tmp, co2


# %%
def read_plot(data1, var):
    print(var)

    data1[lcc == 0] = np.nan

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    if var == "spei":
        levels = [-1, 0, 1]
        labels = [">1", "0 - 1", "-1-0", "<-1"]
        labeltitle = f"gC m-2 \n per SPEI"
        counts = sp.cut(data1, -1, 1)
    elif var == "tmp":
        levels = [-0.1, 0, 0.1]
        labels = [">0.1", "0 - 0.1", "-0.1-0", "<-0.1"]
        labeltitle = f"gC m-2 \n per Cesius Degree"
        counts = sp.cut(data1, -0.1, 0.1)
    elif var == "co2":
        levels = [-0.05, 0, 0.05]
        labels = [">0.05", "0 - 0.05", "-0.05-0", "<-0.05"]
        labeltitle = f"gC m-2 \n per 10 ppm"
        counts = sp.cut(data1, -0.05, 0.05)

    lat = np.linspace(-89.75, 89.75, 360)
    lon = np.linspace(-179.75, 179.75, 720)

    colors = ["#104C73", "#0085AB", "#37A800", "#A90000"]
    title = f"GPP sensitivity to {var}"
    sp.CreatMap(data1, lon, lat, levels, colors, labels, labeltitle, counts, title)


# %%
if __name__ == "__main__":
    spei, tmp, co2 = read_nc()
    lcc = np.load("/mnt/e/MODIS/MCD12C1/MCD12C1_vegnonveg_0.5X0.5.npy")
    var = ["spei", "tmp", "co2"]
    for x in var:
        xx = eval(f"{x}")
        read_plot(xx, x)

# %%
