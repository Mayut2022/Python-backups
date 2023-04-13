# Preseason drought controls on patterns of spring phenology in grasslands of the Mongolian Plateau
# 82-15 NDVI SG+?(方法没看) SOS Spatial pattern (a), frequency distribution (b), temporal (c), and significant trend (d)
# %%
import cmaps
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

import CartopyPlot as cp

# %%
lat_RG = np.linspace(40.25, 54.75, 30)
lon_RG = np.linspace(100.25, 124.75, 50)


def read_nc(yr, var):
    inpath = rf"E:/GLASS-GPP/8-day MG 0.5X0.5 (phenology) /dbl_derive_{yr}.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        var = var.upper()
        sop1 = (f.variables[f'{var}'][:]).data
        sop1[sop1 == -9999] = np.nan
    return sop1


def data(var):
    var_all = []
    for yr in range(1982, 2016):
        var1 = read_nc(yr, var)
        var_all.append(var1)

    var_all = np.array(var_all)

    var_m = np.nanmean(var_all, axis=0)

    return var_m, var_all


def read_lcc():
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat2 = (f.variables['lat'][:])
        lon2 = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
    
    return lcc

lcc = read_lcc()
# %%


def trend(data):
    s, p = np.zeros((30, 50)), np.zeros((30, 50))
    t = np.arange(34)
    for r in range(30):
        if r % 30 == 0:
            print(f"{r} is done!")
        for c in range(50):
            a = data[:, r, c]
            if np.isnan(a).any():
                s[r, c], p[r, c] = np.nan, np.nan
            else:
                s[r, c], _, _, p[r, c], _ = linregress(t, a)
    s = np.ma.array(s, mask=lcc==200)
    p = np.ma.array(p, mask=lcc==200)
    
    return s, p

# %%


def read_plot(data1, data2, var):
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(110, 146, 5)
    cmap = cmaps.MPL_YlGn
    title = f"Spatial Pattern {var}".title()
    cp.plot(data1, data2, levels, cmap, title)

#%%
def read_plot2(data1, data2, var):
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(-1, 1.1, 0.1)
    cmap = "bwr_r"
    # levels = np.arange(-1.5, 1.51, 0.1)
    # cmap = "BrBG"
    title = f"Trend {var}".title()
    cp.plot(data1, data2, levels, cmap, title)


# %%
if __name__ == "__main__":
    var = ["sop1", "sop2", "gsl2"]
    for x in var[:1]:
        exec(f"{x}_m, {x}_all = data(x)")
        # exec(f"read_plot({x}_m, {x}_m)")
        s, p = eval(f"trend({x}_all)")
        exec(f"read_plot2(s, p, x)")

    

# %%
