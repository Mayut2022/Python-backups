# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:20:28 2023

可能部分年份植被变化较小，合成的话效果不是很明显
应将典型年份合成

@author: MaYutong
"""
import cmaps
import warnings

import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from scipy.stats import pearsonr
plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体

warnings.filterwarnings("ignore")


# %%
# yr = np.arange(1983, 2021)

# %%


def read_lcc():
    inpath = (r"E:/ESA/ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])

    return lcc


# %%


def read_nc():
    global lat, lon
    inpath = (r"E:/LAI4g/data_MG/LAI_82_20_MG_SPEI0.5x0.5.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lai'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lai = (f.variables['lai'][:, 4:9, :, :])
        lai_gs = np.nanmean(lai, axis=1)
        lai_diff = lai_gs[1:, ]-lai_gs[:-1, ]
    return lai_diff


def read_nc2():
    inpath = (r"E:/SPEI_base/data/spei03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        spei = (f.variables['spei'][972:, :, :])
        spei = spei.reshape(39, 12, 30, 50)
        spei_gs = np.nanmean(spei[:, 4:9, :, :], axis=1)
        spei_diff = spei_gs[1:, ]-spei_gs[:-1, ]
    return spei_diff


# %%
lcc = read_lcc()
lai = read_nc()
spei = read_nc2()
cr = lai/spei

lai1, lai2 = lai[:18, ], lai[18:, ]
spei1, spei2 = spei[:18, ], spei[18:, ]
cr1, cr2 = cr[:18, ], cr[18:, ]


# %%
def percentile(data):
    per = np.arange(5, 95.1, 5)
    for p in per:
        _ = np.nanpercentile(data.data, p)
        print(f"{p}分位数为：", _)

# %% Simultaneous change
# 统计同向变化
# def sim_change():
#     cr[cr > 0] = 1
#     cr[~(cr > 0)] = np.nan
#     cr1, cr2 = np.nansum(cr[:18, ], axis=0), np.nansum(cr[18:, ], axis=0)


#     var_all = ["cr1", "cr2"]

#     for var in var_all:
#         plt.figure(1, dpi=500)
#         exec(f'{var}[lcc!=130]=np.nan')
#         exec(f"plt.pcolormesh({var}, cmap='Reds', vmin=0, vmax=20)")
#         plt.title(var, fontsize=15)
#         plt.colorbar(shrink=0.75, orientation='horizontal')
#         plt.show()

# 空间图合成
# cr11 = cr1.copy()
# cr11[~(cr11 > 0)] = np.nan
# cr11_ave = np.nanmean(cr11, axis=0)
# cr11_ave[~(lcc == 130)] = np.nan

# percentile(cr11_ave)
# plt.figure(1, dpi=500)
# cs = plt.pcolor(cr11_ave, cmap='Reds', vmin=0, vmax=2)
# plt.colorbar(cs, shrink=0.75, orientation='horizontal')

# %% Reverse change
# 统计非同向变化
# def re_change():
#     cr[cr < 0] = -1
#     cr[~(cr < 0)] = np.nan
#     cr = abs(cr)
#     cr1, cr2 = np.nansum(cr[:18, ], axis=0), np.nansum(cr[18:, ], axis=0)

#     var_all = ["cr1", "cr2"]

#     lcc2 = lcc.copy()
#     lcc2[~(lcc2==130)]=1
#     lcc2[lcc2==130]=np.nan

#     for var in var_all:
#         plt.figure(1, dpi=500)
#         # exec(f'{var}[lcc!=130]=np.nan')
#         # exec(f"print(np.nanmax({var}))")
#         cmap = plt.get_cmap('tab20c')
#         newcolors = cmap(np.linspace(0, 1, 20))
#         newcmap = ListedColormap(newcolors[::4])
#         exec(f"cs = plt.pcolor({var}, cmap=newcmap, vmin=2, vmax=12)")
#         plt.pcolor(lcc2, cmap='Reds')

#         plt.title(var, fontsize=15)
#         plt.colorbar(cs, shrink=0.75, orientation='horizontal')
#         plt.show()

# 空间图合成
# cr22 = cr2.copy()
# cr22[~(cr22 < 0)] = np.nan
# cr22_ave = np.nanmean(cr22, axis=0)
# cr22_ave[~(lcc == 130)] = np.nan

# percentile(cr22_ave)
# plt.figure(2, dpi=500)
# cs = plt.pcolor(cr22_ave, cmap='Blues_r', vmin=-2, vmax=0)
# plt.colorbar(cs, shrink=0.75, orientation='horizontal')

# %%
def corr(d1, d2):
    sp = d1.shape
    spi, spj = sp[1], sp[2]
    r, p = np.zeros((spi, spj)), np.zeros((spi, spj))
    for i in range(spi):
        for j in range(spj):
            a = d1[:, i, j]
            b = d2[:, i, j]
            if np.isnan(a).any() or np.isnan(b).any():
                r[i, j], p[i, j] = np.nan, np.nan
            else:
                r[i, j], p[i, j] = pearsonr(a, b)
    return r, p


# %% 剔除非干旱主导因素格点，转折前后分开研究
r2, p2 = corr(lai2, spei2)
p2[~(lcc == 130)] = np.nan
p2[~(p2 <= 0.05)] = np.nan

# plt.figure(3, dpi=500)
# # level = [0, 0.05, np.nanmax(p2)]
# level = [0, 0.05]
# cmap = cmaps.precip_diff_1lev
# norm = mpl.colors.BoundaryNorm(level, cmap.N)
# cs = plt.pcolor(p2, cmap=cmap, norm=norm)
# plt.colorbar(cs, shrink=0.75, orientation='horizontal')
# plt.title("NonDrought Limited")
# plt.show()


# %%
def plot(data, level, cmap, title):
    plt.figure(1, dpi=500)
    print(title)
    data[~(lcc == 130)] = np.nan
    percentile(data)
    # level = [0, 0.05, np.nanmax(p2)]
    # level = [0, 0.05]
    norm = mpl.colors.BoundaryNorm(level, cmap.N)
    cs = plt.pcolor(data, cmap=cmap, norm=norm)
    plt.colorbar(cs, shrink=0.75, orientation='horizontal')
    plt.title(f"{title}")
    plt.show()


# %% 剔除干旱主导格点的非同向变化年份，得到一个判断是否为对称响应的标准：sym_thre
cr2[cr2 < 0] = np.nan
spei2[cr2 < 0] = np.nan
sym_thre = np.nanstd(cr2, axis=0)

# 首先判断是否为对称变化
cr2_dry = cr2.copy()
cr2_dry[~(spei2 < 0)] = np.nan
cr2_dry = np.nanmean(cr2_dry, axis=0)

plot(cr2_dry, np.arange(0, 1.1, 0.05), cmaps.MPL_Reds, "Dry")

cr2_wet = cr2.copy()
cr2_wet[~(spei2 > 0)] = np.nan
cr2_wet = np.nanmean(cr2_wet, axis=0)
plot(cr2_wet, np.arange(0, 1.1, 0.05), cmaps.MPL_Blues, "Wet")


# %%
cr2_com = np.zeros((30, 50))
ind = cr2_wet>cr2_dry
cr2_com[ind]=1
cr2_com[~ind]=-1
plot(cr2_com, [-1, 0, 1], cmaps.WhBlReWh_r[41:62], "Asymmetric Sample Test")
# %%
# cr2_diff = abs(cr2_dry-cr2_wet)
# cr2_asy = cr2_diff.copy()
# cr2_asy[~(cr2_diff>sym_thre)] = np.nan
# plot(cr2_diff, [0, 0.1, 1], cmaps.MPL_Greys, "Asymmetry")
