
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
import pymannkendall as MK
from scipy.stats import linregress

# %%
print("Time series of annual 17 growing-season mean atmospheric CO2 \
    concentration derived from the CarbonTracker CT2019B dataset ")


def read():
    inpath = r"E:/CO2/CT2019B/molefractions/XCO2 Global/XCO2_00_18.nc"
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        data = (f.variables['co2'][:]).data
    return data, lat, lon


# %%
def trend(data):
    t = np.arange(1, 20, 1)

    s, r, p = np.zeros((90, 120)), np.zeros((90, 120)), np.zeros((90, 120))

    for i in range(len(lat)):
        for j in range(len(lon)):
            s[i, j], _, r[i, j], p[i, j], _ = linregress(t, data[:, i, j])

    return s, p

# %%


def sen(data):
    trend = np.zeros((90, 120))
    mk = np.zeros((90, 120))
    for r in range(90):
        if r % 30 == 0:
            print("still alive!")
        for c in range(120):
            if lcc[r, c] == 0:
                pass
            else:
                res = MK.original_test(data[:, r, c], alpha=0.05)
                trend[r, c] = res[7]
                if res[0] == "increasing":
                    mk[r, c] = -1  # 提前
                elif res[0] == "decreasing":
                    mk[r, c] = 1  # 延后

    return trend, mk

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
    shp_path1 = r'E:/SHP/gadm36_CHN_shp/gadm36_CHN_1.shp'
    shp_path2 = r'E:/SHP/world_shp/world_adm0_Project.shp'

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
    cb.set_label('XCO2(ppm/year)', fontsize=15)

    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(rf'E:/CO2/JPG/JPG-AGU/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%
# 非植被区：0， 植被区：100
lcc = np.load("E:/MODIS/MCD12C1/MCD12C1_vegnonveg_CT.npy")
xco2, lat, lon = read()
xco2_gs = np.nanmean(xco2, axis=1)
s, p = trend(xco2_gs)
tr, _ = sen(xco2_gs)
# %%


def read_plot(data1, data2):
    data1[lcc == 0] = np.nan
    data2[lcc == 0] = np.nan
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(0, 3.51, 0.01)
    cmap = cmaps.vegetation_ClarkU_r[34:220]
    title = "XCO2 MK Trend GS 00-18"
    CreatMap(data1, data2, lon, lat, levels, cmap, title)


read_plot(tr, p)

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


print("GS Mean XCO2 Time Series")
xco2_time = mask(100, xco2_gs)
print("xco2 time series:", "/n", xco2_time)


# %%
