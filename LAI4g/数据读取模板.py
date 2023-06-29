# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:06:30 2023

@author: MaYutong
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas  as pd
import rasterio

# %%
# inpath = r'E:/LAI4g/new/1991_2000_TIFF/GIMMS_LAI4g_V1.0_19940702.tif' ##19940702
inpath = r'E:/LAI4g/2001_2010_TIFF/GIMMS_LAI4g_V1.0_20090701.tif'
with rasterio.open(inpath) as ds:
    print('该栅格数据的基本数据集信息（这些信息都是以数据集属性的形式表示的）：')
    print(f'数据格式：{ds.driver}')
    print(f'波段数目：{ds.count}')
    print(f'影像宽度：{ds.width}')
    print(f'影像高度：{ds.height}')
    print(f'地理范围：{ds.bounds}')
    print(f'反射变换参数（六参数模型）：\n {ds.transform}')
    print(f'投影定义：{ds.crs}')
    # 获取第一个波段数据，跟GDAL一样索引从1开始
    # 直接获得numpy.ndarray类型的二维数组表示，如果read()函数不加参数，则得到所有波段（第一个维度是波段）
    band1 = ds.read(1)
    print(f'第一波段的最大值：{band1.max()}')
    print(f'第一波段的最小值：{band1.min()}')
    print(f'第一波段的平均值：{band1.mean()}')
    # 根据地理坐标得到行列号
    x, y = (ds.bounds.left, ds.bounds.top)
    row, col = ds.index(x, y)  # 对应的行列号
    print(f'(投影坐标{x}, {y})对应的行列号是({row}, {col})')
    # 根据行列号得到地理坐标
    x, y = ds.xy(row, col)  # 中心点的坐标
    print(f'行列号({row}, {col})对应的中心投影坐标是({x}, {y})')
    # 那么如何得到对应点左上角的信息
    # x, y = (row, col) * ds.transform
    # print(f'行列号({row}, {col})对应的左上角投影坐标是({x}, {y})')

    print('-------------------------------------------------')
    x2, y2 = (ds.bounds.right, ds.bounds.bottom)
    row2, col2 = ds.index(x2, y2)  # 对应的行列号
    x2, y2 = ds.xy(row2-1, col2-1)  # 中心点的坐标
    print(f'行列号({row2}, {col2})对应的中心投影坐标是({x2}, {y2})')

# %%
# lat = np.linspace(y, y2, 2160)
# lon = np.linspace(x, x2, 4320)

# np.savez("latlon", lat=lat, lon=lon)

# %% 
data = band1*0.01
data[data==655.35]=np.nan

plt.figure(1, dpi=500)
plt.imshow(data, cmap='jet')
plt.colorbar(shrink=0.75)
# plt.title("")
plt.show()


#%%
df = pd.DataFrame(band1)
df.to_excel("GIMMS_LAI4g_V1.0_20090701.xlsx", index=False, columns=None)
