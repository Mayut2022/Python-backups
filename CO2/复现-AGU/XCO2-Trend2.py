
# %%
import cmaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

from scipy.stats import linregress

# %%
print("Time series of annual 17 growing-season mean atmospheric CO2 \
    concentration derived from the CarbonTracker CT2019B dataset ")


def read():
    inpath = r"/mnt/e/CO2/CT2019B/molefractions/XCO2 Global/XCO2_00_18_SPEI0.5x0.5.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        data = (f.variables['co2'][:]).data
    return data, lat, lon


# %%
def trend(data):
    t = np.arange(1, 20, 1)

    s, r, p = np.zeros((360, 720)), np.zeros((360, 720)), np.zeros((360, 720))

    for i in range(len(lat)):
        for j in range(len(lon)):
            s[i, j], _, r[i, j], p[i, j], _ = linregress(t, data[:, i, j])

    return s, p


# %%


def CreatMap(data1, data2, lon, lat, levels, cmap, title):
    fig = plt.figure(figsize=(10, 6), dpi=1080)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection=proj)
    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9,
                        top=0.9, wspace=None, hspace=None)

    # 打开边框
    ax.spines['geo'].set_visible(True)

    # 设置cfeature自带底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), linewidth=1)
    # ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=1, zorder=2)

    # 设置shp文件
    shp_path1 = r'/mnt/e/SHP/gadm36_CHN_shp/gadm36_CHN_1.shp'
    shp_path2 = r'/mnt/e/SHP/world_shp/world_adm0_Project.shp'

    provinces = cfeature.ShapelyFeature(
        Reader(shp_path1).geometries(),
        ccrs.PlateCarree(),
        edgecolor='k',
        facecolor='none')

    world = cfeature.ShapelyFeature(
        Reader(shp_path2).geometries(),
        ccrs.PlateCarree(),
        edgecolor='k',
        facecolor='none')
    # ax.add_feature(provinces, linewidth=1.2, zorder=3)
    ax.add_feature(world, linewidth=1.2, zorder=3)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xticks(np.arange(-180, 181, 60))  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    region = [-180, 180, -90, 90]
    # region = [100.25, 124.25, 40.25, 54.75]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = np.arange(-90, 91, 30)
    xlocs = np.arange(-180, 181, 60)
    lb = ax.gridlines(draw_labels=True, ylocs=ylocs, xlocs=xlocs, x_inline=False, y_inline=False,
                      linewidth=1, color='lightgray', linestyle='--', alpha=0.8)  # alpha是透明度

    lb.top_labels = False
    lb.bottom_labels = False
    lb.right_labels = False
    lb.left_labels = False
    lb.xlabel_style = {'size': 15}  # 修改经纬度字体大小
    lb.ylabel_style = {'size': 15}

    # 绘制填色图等
    lon, lat = np.meshgrid(lon, lat)

    cs = ax.contourf(lon, lat, data1,
                     levels=levels, transform=ccrs.PlateCarree(),
                     cmap=cmap, extend="both", zorder=1)

    # cs2 = ax.contourf(lon, lat, data2,
    #                   levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
    #                   hatches=['..', None], colors="none", zorder=3)

    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    # cb.set_label('XCO2(ppm/year)', fontsize=15)
    cb.set_label('CRU TMP ABOVE ZERO', fontsize=15)

    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'/mnt/e/CO2/JPG/JPG-AGU/{title}.jpg', bbox_inches='tight')
    plt.show()

#%%
def gslength(data):
    data_gs = np.empty((19, 360, 720))
    for r in range(360):
        for c in range(720):
            sm1, em1 = int(sm[r, c]), int(em[r, c]+1)
            if sm1!=0 and em1!=0:
                data_gs[:, r, c] = np.nanmean(data[:, sm1:em1, r, c], axis=1)
            else:
                data_gs[:, r, c] = np.nanmean(data[:, :, r, c], axis=1)
    return data_gs

# %%
# 非植被区：0， 植被区：100
# lcc = np.load("/mnt/e/MODIS/MCD12C1/MCD12C1_vegnonveg_CT.npy")
_ = np.load("./TemAbove0.npz")
sm, em = _["sm"], _["em"]
# gs = em+1-sm

xco2, lat, lon = read()
xco2_gs = gslength(xco2)
s, p = trend(xco2_gs)


# %%


def read_plot(data1, data2):
    # data1[lcc == 0] = np.nan
    # data2[lcc == 0] = np.nan

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(0, 3.51, 0.01)
    cmap = cmaps.vegetation_ClarkU_r[34:220]
    title = "XCO2 Trend2 GS 00-18"
    # levels = np.arange(1, 13, 1)
    # cmap = "Set3"
    # title = "Growing Season Length"
    CreatMap(data1, data2, lon, lat, levels, cmap, title)


read_plot(s, p)

# %% mask数组


def mask(x, data):
    # 保留输入的x数值的部分
    lcc2 = lcc.copy()

    np.place(lcc2, lcc2 != x, 1)
    np.place(lcc2, lcc2 == x, 0)

    spei_ma = np.empty((19, 90, 120))
    spei_ma = np.ma.array(spei_ma)

    for i in range(19):
        a = data[i, :, :]
        a = np.ma.masked_array(a, mask=lcc2)
        spei_ma[i, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(1, 2))

    return spei_ma_ave.data


# print("GS Mean XCO2 Time Series")
# xco2_time = mask(100, xco2_gs)
# print("xco2 time series:", "/n", xco2_time)


# %%

