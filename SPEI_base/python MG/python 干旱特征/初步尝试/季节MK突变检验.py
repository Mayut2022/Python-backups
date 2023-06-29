# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:02:56 2023

@author: MaYutong
"""


import netCDF4 as nc
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr

#%%
        
def read_nc():
    global spei
    inpath = (r"E:/SPEI_base/data/spei03_MG_season.nc")
    with nc.Dataset(inpath) as f:
        #print(f.variables.keys())
        spei = (f.variables['spei'][[1, 2],80:, :,])

#%%
def read_nc2():
    global lcc, lat, lon
    # global a1, a2, o1, o2
    inpath = (r"E:\ESA\ESA_LUC_2000_SPEI03_MG.nc")
    with nc.Dataset(inpath) as f:
        # print(f.variables.keys())
        # print(f.variables['lccs_class'])
        lat = (f.variables['lat'][:])
        lon = (f.variables['lon'][:])
        lcc = (f.variables['lcc'][:])
        
#%% mask数组
def mask(x, data):
    lcc2 = np.array(lcc)
    
    np.place(lcc2, lcc2!=x, 1)
    np.place(lcc2, lcc2==x, 0)
    
    spei_ma = np.empty((2, 40, 30, 50))
    spei_ma = np.ma.array(spei_ma)
    
    for i in range(2):
        for l in range(40):
            a = data[i, l, :, :]
            a = ma.masked_array(a, mask=lcc2, fill_value=-999)
            spei_ma[i, l, :, :] = a

    spei_ma_ave = np.nanmean(spei_ma, axis=(2, 3))
    
    return spei_ma_ave

#%%
def mktest(inputdata):
    inputdata = np.array(inputdata)
    n=inputdata.shape[0]
    s              =  0
    Sk = np.zeros(n)
    UFk = np.zeros(n)
    for i in range(1,n):
        for j in range(i):
            if inputdata[i] > inputdata[j]:
                s = s+1
            else:
                s = s+0
        Sk[i] = s
        E = (i+1)*(i/4)
        Var = (i+1)*i*(2*(i+1)+5)/72
        UFk[i] = (Sk[i] - E)/np.sqrt(Var)
        
    Sk2 = np.zeros(n)
    UBk = np.zeros(n)
    s  =  0
    inputdataT = inputdata[::-1]
    for i in range(1,n):
        for j in range(i):
            if inputdataT[i] > inputdataT[j]:
                s = s+1
            else:
                s = s+0
        Sk2[i] = s
        E = (i+1)*(i/4)
        Var = (i+1)*i*(2*(i+1)+5)/72
        UBk[i] = -(Sk2[i] - E)/np.sqrt(Var)
    UBk2 = UBk[::-1]
    return UFk, UBk2

def tip(uf, ub):
    year = np.arange(1981, 2021, 1)
    sign = np.sign(uf[0]-ub[0])
    for yr, a, b in zip(year, uf, ub):
        if np.sign(a-b)==-sign:
            print(yr, a, b)
            sign = -sign
        
    
#%%
def plot(uf, ub, title):
    fig = plt.figure(figsize=(12, 6), dpi=500)
    fig.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.92, wspace=None, hspace=0.1)
    ax = fig.add_subplot(111)
    
    t = pd.date_range(f'1981', periods=40, freq="YS")
    ax.plot(t, uf, 'r', label='UFk')
    ax.plot(t, ub, 'b', label='UBk')
    ax.scatter(t, uf, color='r')
    ax.scatter(t, ub, color='b')
    
    ax.axhline(y=1.96, c="k", linestyle="--")
    ax.axhline(y=-1.96, c="k", linestyle="--")
    
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y'))
    
    ax.set_ylim(-3.8, 3.8)
    ax.set_yticks(np.arange(-3, 3.1, 1))
    #ax.set_ylabel("percent", fontsize=15)
    ax.tick_params(labelsize=15)
    
    plt.legend(loc="upper right", fontsize=15)
    plt.suptitle(f'{title}', fontsize=20)
    # plt.savefig(rf'E:/SPEI_base/JPG_NorthEast Asia region1/MK突变/{title}.jpg', bbox_inches='tight')
    plt.show()

#%%

read_nc()
read_nc2()

    
#%% season
season = ["Spring", "Summer", "Autumn", "Winter"]
spei_ave = mask(130, spei)
for i in range(2):
    uf2, ub2 = mktest(spei_ave[i, :]) #front, back
    tip(uf2, ub2)
    print("")
    print(uf2.max(), ub2.max(), uf2.min(), ub2.min())
    plot(uf2, ub2, f"MK test {season[i+1]}")

