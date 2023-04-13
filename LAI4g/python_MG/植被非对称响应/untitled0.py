# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:58:14 2023

@author: MaYutong
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

penguins = sns.load_dataset("penguins")

fig = plt.figure(figsize=(8, 6), dpi=1080)

ax = fig.add_axes([0.1, 0.1, 0.6, 0.6])

color = ["#2878B5", "#F8AC8C"]
color2 = ["#1681FC", "#FD7F1E"]
color3 = ["gray", "k", "red", "blue"]
# Draw a nested barplot by species and sex
g = sns.barplot(ax=ax,
    data=penguins,
    x="species", y="body_mass_g", hue="sex", palette=color3, alpha=0, 
    edgecolor=color3, linewidth=2)

# g.set_axis_labels("", "Body mass (g)")
# g.legend.set_title("")
