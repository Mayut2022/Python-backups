# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:54:33 2022

@author: MaYutong
"""
from collections import Counter
import cmaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth

from eofs.standard import Eof
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle

import netCDF4 as nc

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import xlsxwriter

import seaborn as sns

from scipy.stats import linregress
from scipy.stats import pearsonr

#%%
def read_nc(inpath):
    global  lat, lon
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        
        r = (f.variables['Rmax'][:])
        r_s = (f.variables['Rmax_scale'][:])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        
        r_c = r.copy()
        r_c[r_c>=0]=1; r_c[r_c<0]=-1
        
    return r, r_s, r_c

#%%
def read_nc3():
    global lcc
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
        
#%%
def lcc_count(data):
    sp = data.shape
    data = data.reshape(1, sp[0]*sp[1]*sp[2])
    data = np.squeeze(data)
    
    lcc_c = Counter(data)
    lcc_c = pd.Series(lcc_c)
    lcc_c = lcc_c.sort_index()
    
    return lcc_c.iloc[:12]

#%% mask数组, mask掉除130以外的
def mask(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = ma.masked_array(data, mask=lcc2)

    return spei_ma


def mask2(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((28, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(28):
        a = data[i, :, :]
        a = ma.masked_array(a, mask=lcc2, fill_value=np.nan)
        spei_ma[i, :, :] = a.filled()

    
    return spei_ma

#%%
def make_map(ax):
    # 打开边框
    ax.spines['geo'].set_visible(True)

    # 设置cfeature自带底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=1, zorder=2)

    # 设置shp文件
    shp_path1 = r'E:\SHP\gadm36_CHN_shp\gadm36_CHN_1.shp'
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
    
    
    ax.set_yticks(list(range(42, 55, 3)))
    ax.set_xticks(list(np.arange(102.5, 126, 5)))  # 需要显示的纬度
    
    ax.tick_params(labelsize=15)

    # 区域
    
    region = [100.25, 124.75, 40.25, 54.75]
    ax.set_extent(region, crs=ccrs.PlateCarree())

    # 设置网格点
    ylocs = list(range(42, 55, 3))
    xlocs = list(np.arange(102.5, 126, 5))
    lb = ax.gridlines(draw_labels=True, ylocs=ylocs, xlocs=xlocs, x_inline=False, y_inline=False,
                      linewidth=1, color='lightgray', linestyle='--', alpha=0.8)  # alpha是透明度

    lb.top_labels = False
    lb.bottom_labels = False
    lb.right_labels = False
    lb.left_labels = False
    lb.xlabel_style = {'size': 15}  # 修改经纬度字体大小
    lb.ylabel_style = {'size': 15}
    
    return ax



def Creat_figure(data1, data2, data3, levels, cmap, lat, lon, title):
    fig=plt.figure(figsize=(18, 8), dpi=500)
    proj = ccrs.PlateCarree()

    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.9,
                        top=0.9, wspace=None, hspace=0.2)
    
    lon, lat = np.meshgrid(lon, lat)
    
    for i in range(1, 4):
        ax = fig.add_subplot(2, 3, i, projection=proj)
        ax = make_map(ax)
        ax.set_title(f"EOF0{i}", fontsize=15, loc="left")
        cs = ax.contourf(lon, lat, data1[i-1, :, :],
                         levels=levels, transform=ccrs.PlateCarree(),
                         cmap=cmap, extend="both", zorder=1)
    
        ax.tick_params(labelsize=15)

    t = np.arange(1986, 2014)
    for j in range(4, 7):
        ax = fig.add_subplot(2, 3, j)
        #ax.scatter(t, data2[:, j-4])
        
        #渐变色柱状图
        #归一化
        norm = plt.Normalize(-3, 3) #值的范围
        norm_values = norm(data2[:, j-4])
        map_vir = cm.get_cmap(name='bwr_r')
        colors = map_vir(norm_values)
        ax.bar(t, data2[:, j-4], color=colors, width=1)
        ax.set_title(f"PC0{j-3} ({data3[j-4]:.2f}%)", fontsize=15, loc="left")####
        
        ax.tick_params(labelsize=15)
        ax.set_ylim(-3.5, 3.5)
        ax.set_yticks(np.arange(-3, 3.1, 1))
        ax.grid(which="major", color="dimgray", linestyle='--', linewidth=0.3)
        
    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    cb = fig.colorbar(cs, cax=cbar_ax, orientation='vertical')
    #cb.set_label('Units: mm/month (CRU-GLEAM)', fontsize=15)
    cb.ax.tick_params(labelsize=15)
    ax.text(2025, -4, f"{test}", color="dimgray")
    
    plt.suptitle(f'{title}', fontsize=25)
    plt.savefig(rf'E:/SPEI_base/python MG/JPG/Rmax_Scale分析/{title}.jpg', bbox_inches = 'tight')
    plt.show()

#%%
'''EOF分解 要求（time，lat，lon）分布'''
def eof_cal(data):
    coslat = np.cos(np.deg2rad(lat))
    wgts = np.sqrt(coslat)[..., np.newaxis] 
    
    #计算纬度权重
    solver = Eof(data, weights = wgts)
    #创建EOF函数
    eof=solver.eofs(neofs=3)
    #获取前三个模态
    pc = solver.pcs(npcs=3, pcscaling=1)
    var = solver.varianceFraction()
    #获取对应的PC序列和解释方差
    #方差贡献
    eigen_Values   = solver.eigenvalues()
    percentContrib = eigen_Values * 100./np.sum(eigen_Values)
    #North 检验
    eigen = solver.eigenvalues(neigs=3)
    north = solver.northTest(neigs=3)
    print('特征值: ', solver.eigenvalues(neigs=3))
    print('标准误差: ', solver.northTest(neigs=3))
    
    print("第一特征值保留！")
    test = [1]
    if eigen[0]>(eigen[1]+north[0]):
        print("第二特征值过检！")
        test.append(2)
    
    if eigen[0]>(eigen[1]+north[0]) and eigen[1]>(eigen[2]+north[1]):
        print("第三特征值过检！")
        test.append(3)
        
    print("\n")
    return eof, pc, percentContrib, test    


#%%
def If_change():
    change = np.zeros((30, 50))
    for i in range(30):
        for j in range(50):
            a = np.unique(r_c[:, i, j])
            a = a[~np.isnan(a)]
            if len(a)==2:
                change[i, j]=1
            else:
                pass
    change_ma = mask(130, change)

#%%
read_nc3()
df2 = pd.read_excel('E:/ERA5/每月天数.xlsx')
mn_str = df2['月份']
#%%
#r_s_cou = lcc_count(r_s)
#r_s_mask_cou = lcc_count(r_s_mask)

#%% 画图
def read_plot(data1, data2, data3, title):
    print('5 percentile is: ' , np.nanpercentile(data1, 5))
    print('95 percentile is: ' , np.nanpercentile(data1, 95), "\n")
    
    print(np.nanmax(data2))
    print(np.nanmin(data2), "\n")
    
    levels = np.arange(-0.08, 0.081, 0.01)
    cmap = cmaps.MPL_BrBG
    Creat_figure(data1, data2, data3, levels, cmap, lat, lon, title)
    
#%%
month = np.arange(4, 11, 1)

for mn in month[4:5]:
    inpath = rf"E:/SPEI_base/python MG/response time data/GPP_SPEI_MG_month{mn}.nc"
    r, r_s, r_c = read_nc(inpath)
    r_mask, r_s_mask = mask2(130, r), mask2(130, r_s)
    eof, pc, percentContrib, test = eof_cal(r_mask)
    eof2, pc2, percentContrib2, test2 = eof_cal(r_s_mask)
    read_plot(eof.data, pc, percentContrib, title=f"MGGrass GPP Rmax EOF {mn_str[mn-1]}")
    read_plot(eof2.data, pc2, percentContrib2, title=f"MGGrass GPP Rmax Scale EOF {mn_str[mn-1]}")


