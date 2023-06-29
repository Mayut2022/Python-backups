# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 21:09:30 2023



@author: MaYutong
"""


import warnings
from matplotlib import font_manager
import matplotlib.dates as mdate
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
import xarray as xr

plt.rcParams['font.sans-serif'] = ['times new roman']  # 指定默认字体
warnings.filterwarnings("ignore")
# %%

df = pd.read_excel("Sample2.xlsx")



# %%

df1 = df[df['YEAR'] < 1998]
df2 = df[(df['YEAR'] >= 1998)&(df['YEAR'] < 2001)]
df3 = df[df['YEAR'] >= 2001]



# %%
def plot(data, title):
    fig = plt.figure(1, figsize=(6, 5), dpi=500)
    
    ax = fig.add_subplot(111)
    sns.scatterplot(data=data, x="SPEI dn", y="LAI dn", s=10,
                    hue="YEAR", palette="Oranges", ax=ax)
    sns.scatterplot(data=data, x="SPEI up", y="LAI up", s=10,
                    hue="YEAR", palette="Greens", ax=ax)
    
    ax.tick_params(labelsize=15)
    ax.set_title(f"{title}", fontsize=15)
    ax.set_ylabel("LAI Change", fontsize=15)
    ax.set_xlabel("SPEI Change", fontsize=15)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xlim(-3.5, 3.5)
    ax.axvline(0, c='k', linestyle="-")
    ax.axhline(0, c='k', linestyle="-")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    
    # plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力2/JPG/{title}.jpg',
    #             bbox_inches='tight')
    plt.show()


#%%
plot(df1, title=r"Sample Scatter 82-97")
plot(df2, title=r"Sample Scatter 98-00")
plot(df3, title=r"Sample Scatter 01-20")