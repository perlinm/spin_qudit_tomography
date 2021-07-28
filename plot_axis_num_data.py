#!/usr/bin/env python3

import os, glob
import numpy as np
import matplotlib.pyplot as plt

data_dir = "./data/"
fig_dir = "./figures/"

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

params = { "font.size" : 10,
           "text.usetex" : True }
plt.rcParams.update(params)

figsize = (3,2)

colors = [ "k", "tab:blue", "tab:orange", "tab:green" ]
dims = [ 10, 20, 40, 80 ]

# quantum error scales achievable with the method in newton1968measurability,
#   optimized over all polar angles
old_scales = { 10 : 5.205453968693979,
               20 : 8.689567681451933,
               40 : 14.26455397815116,
               80 : 23.36684910939962 }

##################################################

plt.figure(figsize = figsize)

for dim, color in zip(dims, colors):
    files = glob.glob(data_dir + f"axis_num_data_d{dim}.txt")
    assert(len(files) == 1)
    axes, _, scales = np.loadtxt(files[0], unpack = True)

    scales *= np.sqrt(axes)
    excess_axes = axes - (2*dim-1)
    plt.plot(excess_axes/dim, scales/scales[0],
             ".", label = f"$d={dim}$", color = color)

    old_scale = old_scales[dim] * np.sqrt(2*dim-1)
    plt.axhline(old_scale/scales[0], color = color, linewidth = 1, zorder = 0)

plt.gca().set_ylim(bottom = 0)
plt.gca().tick_params(right = True)

plt.xlabel(r"$p/d$")
plt.ylabel(r"$\tilde\beta(p)/\tilde\beta(0)$")
spacing_kwargs = dict( handlelength = 1, columnspacing = 1, labelspacing = 0.2 )
plt.legend(loc = "best", ncol = 2, **spacing_kwargs)
plt.tight_layout()
plt.savefig(fig_dir + "qudit_axes.pdf")
