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

df1 = pd.read_excel("LAI Sig Change2.xlsx", sheet_name="Sig Up")
df2 = pd.read_excel("LAI Sig Change2.xlsx", sheet_name="Sig Down")


# %%
# fig = plt.figure(1, figsize=(8, 8), dpi=500)

# ax = fig.add_subplot(111)
# sns.scatterplot(data=df1, x="SPEI Diff", y="LAI Diff",
#                 hue="Year", palette="Greens", ax=ax)
# sns.scatterplot(data=df2, x="SPEI Diff", y="LAI Diff",
#                 hue="Year", palette="Oranges", ax=ax)

# ax.set_title("Significant Change2", fontsize=15)
# ax.legend(loc="lower right", ncol=2)

# plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力/JPG/Significant Change2.jpg',
#             bbox_inches='tight')


# %%

df1_1 = df1[df1['Year'] < 2000]
df1_2 = df1[df1['Year'] >= 2000]

df2_1 = df2[df2['Year'] < 2000]
df2_2 = df2[df2['Year'] >= 2000]

# fig = plt.figure(1, figsize=(8, 8), dpi=500)

# ax = fig.add_subplot(111)
# sns.scatterplot(data=df1_1, x="SPEI Diff", y="LAI Diff",
#                 hue="Year", palette="Greens", ax=ax)
# sns.scatterplot(data=df2_1, x="SPEI Diff", y="LAI Diff",
#                 hue="Year", palette="Oranges", ax=ax)

# ax.set_title("Significant Change2 82-98", fontsize=15)
# ax.legend(loc="lower right", ncol=2)

# plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力/JPG/Significant Change2.jpg',
#             bbox_inches='tight')


# %%
fig = plt.figure(1, figsize=(6, 5), dpi=500)

ax = fig.add_subplot(111)
sns.scatterplot(data=df1_2, x="SPEI Diff", y="LAI Diff", s=10,
                hue="Year", palette="Greens", ax=ax)
sns.scatterplot(data=df2_2, x="SPEI Diff", y="LAI Diff", s=10,
                hue="Year", palette="Oranges", ax=ax)

ax.tick_params(labelsize=15)
ax.set_title("Significant Change2 00-20", fontsize=15)
ax.set_ylabel("LAI Change", fontsize=15)
ax.set_xlabel("SPEI Change", fontsize=15)
ax.set_ylim(-1.25, 1.25)
ax.set_xlim(-3.5, 3.5)
ax.axvline(0, c='b', linestyle="--")
ax.axhline(0, c='b', linestyle="--")
ax.legend(loc="lower right", ncol=2)

plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力/JPG/Significant Change2.jpg',
            bbox_inches='tight')

# %%
# fig = plt.figure(2, figsize=(8, 8), dpi=500)

# ax = fig.add_subplot(111)
# sns.jointplot(data=df1, x="SPEI Diff", y="LAI Diff", kind="hex",
#                 palette="Blues")
# sns.jointplot(data=df2, x="SPEI Diff", y="LAI Diff", kind="hex",
#                 color="Red")

# plt.title("Significant Change2", fontsize=15)
# ax.legend(loc="lower right", ncol=2)

# plt.savefig(rf'E:/LAI4g/python_MG/抵抗力和恢复力/JPG/Significant Change1.jpg',
#             bbox_inches='tight')
