# %%

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

# %%


def read_nc():
    inpath = ('./FittingGlobalSensitivityC4.nc')
    with nc.Dataset(inpath, mode='r') as f:

        print(f.variables.keys(), "\n")
        # print(f.variables['fit_all'], "\n")

        spei = (f.variables['fit_spei00'][:]).data  
        tmp = (f.variables['fit_tmp00'][:]).data  
        co2 = (f.variables['fit_co200'][:]).data  
        sos = (f.variables['fit_sos00'][:]).data

        return spei, tmp, co2, sos

def read_nc2():
    inpath = (r"/mnt/e/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lcc = (f.variables['lcc'][:]).data

    return lcc

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
def plot(data1, data2, data3, data4, title):
    print(np.nanmax(data1), np.nanmin(data1))
    print(np.nanmax(data2), np.nanmin(data2))
    print(np.nanmax(data3), np.nanmin(data3))
    print(np.nanmax(data4), np.nanmin(data4))
    
    fig = plt.figure(1, figsize=(10, 6), dpi=500)
    
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=None, hspace=0.12)
    ax = fig.add_subplot(111)
    yr = np.arange(1982, 2019)

    ax.scatter(yr, data1)
    ax.plot(yr, data1, label="SPEI")
    ax.scatter(yr, data2)
    ax.plot(yr, data2, label="TMP", c="r")
    ax.scatter(yr, data3)
    ax.plot(yr, data3, label="CO2", c="black")
    ax.scatter(yr, data4)
    ax.plot(yr, data4, label="SOS", c="green")

    ax.set_ylim(60, 85)

    ax.tick_params(labelsize=15)
    
    ax.legend(loc = 'upper right', bbox_to_anchor=(1, 1), fontsize=15)
    
    plt.xlabel("years", fontsize=20)
    plt.ylabel("Simulating GPP (gC m-2)", fontsize=20)
    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'/mnt/e/CO2/JPG/JPG-AGU-MG/combin4/{title}.jpg', 
                bbox_inches='tight')
    plt.show()

# %%
lcc = read_nc2()
spei, tmp, co2, sos = read_nc()
var = ["spei", "tmp", "co2", "sos"]
for x in var:
    exec(f"{x}_ave = mask(130, {x})")

plot(spei_ave, tmp_ave, co2_ave, sos_ave, "82-18 SOS GSL")

# %%
