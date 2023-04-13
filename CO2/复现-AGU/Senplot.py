
# cp -f /mnt/e/CO2/复现-AGU/Senplot.py /home/ma_yu_t_2023/env-py310/lib/python3.10/site-packages/
# %%
import cmaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

import netCDF4 as nc
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['times new roman'] # 指定默认字体
# %%


def read_nc():
    inpath = ('/mnt/e/CO2/复现-AGU/FittingGlobalSensitivity.nc')
    with nc.Dataset(inpath, mode='r') as f:

        # print(f.variables.keys(), "\n")
        # print(f.variables['fit_all'], "\n")

        vpd = (f.variables['vpd_sen'][:]).data # units: hPa
        tmp = (f.variables['tmp_sen'][:]).data # units: Celsius degree
        co2 = (f.variables['co2_sen'][:]).data # units: ppm()
        co2 = co2*10

        return vpd, tmp, co2

#%%
def CreatMap(data1, lon, lat, levels, colors, labels, labeltitle, counts, title):
    fig = plt.figure(figsize=(10, 6), dpi=1080)
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection=proj)
    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9,
                        top=0.9, wspace=None, hspace=None)

    # 打开边框
    ax.spines['geo'].set_visible(True)

    # 设置cfeature自带底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
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
    ax.add_feature(world, linewidth=0.8, zorder=3)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    # yticks = np.arange(42.5, 55, 5)
    # xticks = np.arange(102.5, 130, 5)

    yticks = np.arange(-90, 91, 30)
    xticks = np.arange(-180, 181, 60)

    ax.set_yticks(yticks)
    ax.set_xticks(xticks)  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    # region = [100.25, 124.25, 40.25, 54.75]
    region = [-180, 180, -90, 90]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = yticks
    xlocs = xticks
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
                     colors=colors, extend="both", zorder=1)

    # 绘制矩形图例
    rectangles = [Rectangle((0, 0,), 1, 1, facecolor=x, edgecolor="k") for x in colors[::-1]]
    labels = labels
    ax.legend(rectangles, labels,
              bbox_to_anchor=(0.2, 0.4), fancybox=True, frameon=True, 
              fontsize=12, title=labeltitle)

    ## 柱状图统计
    ax_bar = bar_plot(fig, counts, colors)

    ## MG子图
    sub_ax = fig.add_axes([0.9, 0.4, 0.24, 0.2],
                          projection=ccrs.PlateCarree())
    sub_ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    sub_ax.set_extent([100, 125, 40, 55])
    sub_ax.add_feature(world, linewidth=0.8, zorder=2)
    plt.title('Mongolian plateau', fontsize=15)
    sub_ax.contourf(lon, lat, data1,
                    levels=levels, transform=ccrs.PlateCarree(),
                    colors=colors, extend="both", zorder=1)

    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(
        rf'/mnt/e/CO2/JPG/JPG-GLOBAL/{title}.jpg', bbox_inches='tight')
    plt.show()

# %%

def cut(data, min, max):
    data = data.reshape(1, 360*720)
    data = np.squeeze(data)
    data = pd.Series(data)
    data = data.dropna()

    bin = [-np.inf, min, 0, max, np.inf]

    cats = pd.cut(data, bin, right=False)
    counts = cats.value_counts()/len(data)
    counts.sort_index(ascending=False, inplace=True)
    print(counts)
    return counts

def cut2(data):
    data = data.reshape(1, 360*720)
    data = np.squeeze(data)
    data = pd.Series(data)
    data = data.dropna()

    bin = [-np.inf, 0.25, 0.50, 0.75, np.inf]

    cats = pd.cut(data, bin, right=False)
    counts = cats.value_counts()/len(data)
    counts.sort_index(ascending=False, inplace=True)
    print(counts)
    return counts

def bar_plot(fig, data, colors):
    ax = fig.add_axes([0.55, 0.23, 0.23, 0.15])
    data.plot.bar(ax=ax, \
        rot=0, width = 0.8, yticks=[0, 0.4, 0.8],\
        color=colors[::-1])

    ax.set_xticklabels(["", "", "", ""])
    ax.set_yticklabels(["0%", "40%", "80%"])
    ax.tick_params(labelsize=10)
    ax.set_ylabel("Percentage", fontsize=10)




#%%
def read_plot(data1, var):
    print(var)

    data1[lcc == 0] = np.nan

    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    if var == "vpd":
        levels = [-0.001, 0, 0.001]
        labels = [">0.001", "0 - 0.001", "-0.001-0", "<-0.001"]
        labeltitle = f"W m-2 um-1 sr-1 \n per hPa"
        counts = cut(data1, -0.001, 0.001)
    elif var == "tmp":
        levels = [-0.01, 0, 0.01]
        labels = [">0.01", "0 - 0.01", "-0.01-0", "<-0.01"]
        labeltitle = f"W m-2 um-1 sr-1 \n per Cesius Degree"
        counts = cut(data1, -0.01, 0.01)
    elif var == "co2":
        levels = [-0.005, 0, 0.005]
        labels = [">0.005", "0 - 0.005", "-0.005-0", "<-0.005"]
        labeltitle = f"W m-2 um-1 sr-1 \n per 10 ppm"
        counts = cut(data1, -0.005, 0.005)

    lat = np.linspace(-89.75, 89.75, 360)
    lon = np.linspace(-179.75, 179.75, 720)

    colors = ["#104C73", "#0085AB", "#37A800", "#A90000"]
    title = f"GOSIF sensitivity to {var}"
    # CreatMap(data1, lon, lat, levels, colors, labels, labeltitle, counts, title)

# %%
if __name__ == "__main__":
    vpd, tmp, co2 = read_nc()
    lcc = np.load("/mnt/e/MODIS/MCD12C1/MCD12C1_vegnonveg_0.5X0.5.npy")
    var = ["vpd", "tmp", "co2"]
    for x in var:
        xx = eval(f"{x}")
        read_plot(xx, x)
# %%
