# %%
import netCDF4 as nc
import numpy as np

# %%


def read_nc():
    global lat, lon
    inpath = r"/mnt/e/CRU/TMP_DATA/TMP_CRU_MONTH_81_20.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())

        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        tmp = (f.variables['tmp'][:, 19:38, :, :]).data
        tmp[tmp == 9.96921e+36] = np.nan
    return tmp


# %%
# t = np.arange(1981, 2021)
tmp = read_nc()
tmp_ave = np.nanmean(tmp, axis=1)

# %%
sm, em = np.empty((360, 720)), np.empty((360, 720))
for r in range(360):
    for c in range(720):
        tem = tmp_ave[:, r, c]
        if np.isnan(tem).any():
            sm[r, c], em[r, c] = 0, 0
        else:
            ind = np.argwhere(tem > 0)
            if len(ind) == 0:
                sm[r, c], em[r, c] = 0, 0
            else:
                sm[r, c], em[r, c] = ind[0][0], ind[-1][0]
# %%
np.savez("./TemAbove0", sm=sm, em=em)
# %%
