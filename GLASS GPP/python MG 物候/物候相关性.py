# %%
import cmaps
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import linregress
import xarray as xr

import CorrPlotDemo as cpd

# %%
lat_RG = np.linspace(40.25, 54.75, 30)
lon_RG = np.linspace(100.25, 124.75, 50)


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


# %%
def read_nc2(inpath):
    global gpp, lat2, lon2
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        gpp = (f.variables['GPP'][:])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])

    return gpp


def sif_xarray(band1):
    mn = np.arange(12)
    sif = xr.DataArray(band1, dims=['mn', 'y', 'x'], coords=[mn, lat2, lon2])

    sif_MG = sif.loc[:, 40:55, 100:125]
    return np.array(sif_MG)


def exact_data1():
    for yr in range(1982, 2019):
        inpath = rf"/mnt/e/GLASS-GPP/Month RG/GLASS_GPP_RG_SPEI_0.5X0.5_{yr}.nc"
        data = read_nc2(inpath)

        data_MG = sif_xarray(data)
        data_MG = data_MG.reshape(1, 12, 30, 50)
        if yr == 1982:
            data_all = data_MG
        else:
            data_all = np.vstack((data_all, data_MG))

    return data_all

# %%


def corr(data1, data2):
    s, p = np.zeros((30, 50)), np.zeros((30, 50))

    for r in range(30):
        if r % 30 == 0:
            print(f"{r} is done!")
        for c in range(50):
            a = data1[:, r, c]
            b = data2[:, r, c]
            if np.isnan(a).any() or np.isnan(b).any():
                s[r, c], p[r, c] = np.nan, np.nan
            else:
                s[r, c], p[r, c] = pearsonr(a, b)

    return s, p

# %%


def plot(data1, data2, var):
    print(var.upper())
    print('5 percentile is: ' , np.nanpercentile(data1, 5))
    print('95 percentile is: ' , np.nanpercentile(data1, 95), "\n")
    
    # levels = np.arange(0, 1.01, 0.05)
    # cmap = cmaps.MPL_OrRd[:97]
    levels = np.arange(-1, 1.01, 0.05)
    if var=="sop2":
        cmap = "BrBG_r"
    else:
        cmap = "BrBG"
    cpd.CreatMap(data1, data2, lon_RG, lat_RG, levels, cmap, title=f"{var}-GPP(Summer) CORR".upper())
    
    


# %%
gpp = exact_data1()
gpp_gs = gpp[:, 3:10, :, :].mean(axis=1)
gpp_s = gpp[:, 5:8, :, :].mean(axis=1)
gpp_spr = gpp[:, 3:5, :, :].mean(axis=1)

# %%
var = ["sop1", "sop2", "gsl2"]
for x in var[1:2]:
    exec(f"_,  {x}_all = data(x)")
    exec(f"r, p = corr(gpp_s, {x}_all)")
    exec(f"plot(r, p, x)")

# %%
