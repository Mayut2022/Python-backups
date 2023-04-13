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

def read_nc2():
    inpath = (r"/mnt/e/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lcc = (f.variables['lcc'][:])
        
    return lcc

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

#%% mask数组
def mask(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((37, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    
    for l in range(37):
        a = data[l, :, :]
        a = np.ma.masked_array(a, mask=lcc2)
        spei_ma[l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))
    
    return spei_ma_ave

#%%
def plot(data1, data2, title):
    print(np.nanmax(data1), np.nanmin(data1))
    print(np.nanmax(data2), np.nanmin(data2))
    
    fig = plt.figure(1, figsize=(10, 6), dpi=500)
    
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=None, hspace=0.12)
    ax = fig.add_subplot(111)
    yr = np.arange(1982, 2019)

    ax.scatter(yr, data1)
    ax.plot(yr, data1, label="SOS")
    ax.scatter(yr, data2)
    ax.plot(yr, data2, label="GSL")

    ax.set_ylim(110, 170)

    ax.tick_params(labelsize=15)
    
    ax.legend(loc = 'upper right', bbox_to_anchor=(1, 1), fontsize=15)
    
    plt.xlabel("years", fontsize=20)
    plt.ylabel("day of the year", fontsize=20)
    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'/mnt/e/GLASS-GPP/JPG MG 物候/{title}.jpg', 
                bbox_inches='tight')
    plt.show()

# %%
if __name__ == "__main__":
    lcc = read_nc2()
    var = ["sop2", "gsl2"]
    for x in var:
        exec(f"_,  {x}_all = data(x)")
        exec(f"{x}_ave = mask(130, {x}_all)")
        
    plot(sop2_ave, gsl2_ave, "82-18 SOS GSL")

# %%
