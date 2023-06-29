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

df1 = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力/LAI Sig Change.xlsx", sheet_name="Sig Up")
df2 = pd.read_excel(r"E:/LAI4g/python_MG/抵抗力和恢复力/LAI Sig Change.xlsx", sheet_name="Sig Down")


# %%
fig = plt.figure(1, figsize=(6, 6), dpi=500)

ax = fig.add_subplot(111)
sns.scatterplot(data=df1, x="SPEI Ori", y="LAI Anom",
                hue="Year", palette="Blues", ax=ax)
sns.scatterplot(data=df2, x="SPEI Ori", y="LAI Anom",
                hue="Year", palette="Reds", ax=ax)

ax.tick_params(labelsize=15)
ax.set_ylabel("LAI Anomaly", fontsize=15)
ax.set_xlabel("SPEI Oringinal", fontsize=15)
ax.set_ylim(-1.25, 1.25)
ax.set_xlim(-3.5, 3.5)
ax.axvline(0, c='b', linestyle="--")
ax.axhline(0, c='b', linestyle="--")
ax.set_title("Significant Change1", fontsize=15)
ax.legend(loc="lower right", ncol=2)

# plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力/JPG/Significant Change1.jpg',
#             bbox_inches='tight')


# %% 

# df1_1 = df1[df1['Year']<1999]
# df1_2 = df1[df1['Year']>=1999]

# df2_1 = df2[df2['Year']<1999]
# df2_2 = df2[df2['Year']>=1999]

# fig = plt.figure(2, figsize=(8, 8), dpi=500)

# ax = fig.add_subplot(111)
# sns.scatterplot(data=df1_1, x="SPEI De", y="LAI Anom",
#                 hue="Year", palette="Blues", ax=ax)
# sns.scatterplot(data=df2_1, x="SPEI De", y="LAI Anom",
#                 hue="Year", palette="Reds", ax=ax)

# ax.set_title("Significant Change1", fontsize=15)
# ax.legend(loc="lower right", ncol=2)

# plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力/JPG/Significant Change1.jpg',
#             bbox_inches='tight')


# %%
# fig = plt.figure(2, figsize=(8, 8), dpi=500)

# ax = fig.add_subplot(111)
# sns.jointplot(data=df1, x="SPEI De", y="LAI Anom", kind="hex",
#                 palette="Blues")
# # sns.jointplot(data=df2, x="SPEI De", y="LAI Anom", kind="hex",
# #                 color="Red")

# plt.title("Significant Change2 Jointplot", fontsize=15)
# ax.legend(loc="lower right", ncol=2)

# plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力/JPG/Significant Change1.jpg',
#             bbox_inches='tight')
