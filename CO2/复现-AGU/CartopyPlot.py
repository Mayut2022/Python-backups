
# cp -f E:/CO2/复现-AGU/CartopyPlot.py /home/ma_yu_t_2023/env-py310/lib/python3.10/site-packages/
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

import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['times new roman'] # 指定默认字体
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
    ax.add_feature(provinces, linewidth=1.2, zorder=3)
    ax.add_feature(world, linewidth=1.2, zorder=3)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    yticks = np.arange(42.5, 55, 5)
    xticks = np.arange(102.5, 130, 5)

    # yticks = np.arange(-90, 90, 30)
    # xticks = np.arange(-180, 181, 60)

    ax.set_yticks(yticks)
    ax.set_xticks(xticks)  # 需要显示的纬度
    ax.tick_params(labelsize=15)

    # 区域
    region = [100.25, 124.25, 40.25, 54.75]
    # region = [-180, 180, -90, 90]
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
                     cmap=cmap, extend="both", zorder=1)

    # cs2 = ax.contourf(lon, lat, data2,
    #                   levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
    #                   hatches=['...', None], colors="none", zorder=2)

    cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    # cb.set_label('Significant Level: 95%', fontsize=15)
    # cb.set_label('Trend (days/yr)', fontsize=15)


    '''
    # select 矩形关键区区域
    RE = Rectangle((100, 40), 25, 15, linewidth=1.2, linestyle='-', zorder=3,
                   edgecolor='yellow', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(RE)

    # 画子区域蒙古高原，这一步是新建一个ax，设置投影
    sub_ax = fig.add_axes([0.9, 0.4, 0.24, 0.2],
                          projection=ccrs.PlateCarree())
    # 画海，陆地，河流，湖泊
    sub_ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)

    # 框区域
    sub_ax.set_extent([100, 125, 40, 55])
    sub_ax.add_feature(world, linewidth=0.8, zorder=2)
    plt.title('Mongolian plateau', fontsize=15)
    sub_ax.contourf(lon, lat, data1,
                    levels=levels, transform=ccrs.PlateCarree(),
                    cmap=cmap, extend="both", zorder=1)
    sub_ax.contourf(lon, lat, data2,
                    levels=[0, 0.05, 1], transform=ccrs.PlateCarree(),
                    hatches=['...', None], colors="none", zorder=2)
    '''

    plt.suptitle(f'{title}', fontsize=20)
    plt.savefig(
        rf'E:/CO2/python-AGU-MG/JPG/{title}.jpg', bbox_inches='tight')
    plt.show()


# %%
def plot(data1, data2, levels, cmap, title):
    lat = np.linspace(40.25, 54.75, 30)
    lon = np.linspace(100.25, 124.75, 50)

    CreatMap(data1, data2, lon, lat, levels, cmap, title)


def plot2(data1, data2, levels, cmap, title):
    lat = np.linspace(-89.75, 89.75, 360)
    lon = np.linspace(-179.75, 179.75, 720)

    # CreatMap(data1, data2, lon, lat, levels, cmap, title)


# %% 画图
if __name__ == "__main__":
    data1 = np.random.random((30, 50))
    data2 = np.random.random((30, 50))
    levels = np.arange(0, 0.81, 0.05)
    cmap = cmaps.MPL_OrRd[:81]
    title = "Cartopy Plot Test"
    # plot(data1, data2, levels, cmap, title)

# %%
def read_plot(data1, data2):
    data1[lcc == 0] = np.nan
    data2[lcc == 0] = np.nan
    print('5 percentile is: ', np.nanpercentile(data1, 5))
    print('95 percentile is: ', np.nanpercentile(data1, 95), "\n")

    levels = np.arange(0, 0.81, 0.05)
    cmap = cmaps.MPL_OrRd[:81]
    title = "Fit Rsqured"
    # cp.plot2(data1, data2, levels, cmap, title)
