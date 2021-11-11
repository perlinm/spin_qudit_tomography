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

dims = [ 10, 20, 40, 80 ]
colors = [ "k", "#4E79A7", "#F28E2B", "#E15759" ]
markers = [ "^", "s", "p", "o"]

##################################################

plt.figure(figsize = figsize)

for dim, color, marker in zip(dims, colors, markers):
    files = glob.glob(data_dir + f"axis_num_data_d{dim}.txt")
    assert(len(files) == 1)
    axes, _, scales = np.loadtxt(files[0], unpack = True)

    # plot normalized empirical error scales from randomized tomography protocol
    scales *= np.sqrt(axes) # get "measurement-adjusted" error scale
    excess_axes = axes - (2*dim-1)
    excess_axes, scales = excess_axes[::2], scales[::2] # plot less data for legibility
    plt.plot(excess_axes/dim, scales/scales[0], marker, markersize = 4,
             label = f"$d={dim}$", color = color)

    # plot optimal error scales from "old" tomography protocol in
    #   https://doi.org/10.1016/0003-4916(68)90035-3
    with open(data_dir + f"angle_scales_d{dim}.txt", "r") as file:
        for line in file:
            if "optimum" in line:
                old_scale = float(line.split()[-1])

    old_scale *= np.sqrt(2*dim-1) # get "measurement-adjusted" error scale
    plt.axhline(old_scale/scales[0], color = color, linewidth = 1, zorder = 0)

plt.gca().set_ylim(bottom = 0)
plt.gca().tick_params(right = True)

plt.xlabel(r"$p/d$")
plt.ylabel(r"$\tilde\beta(p)/\tilde\beta(0)$")
spacing_kwargs = dict( handlelength = 1, columnspacing = 1, labelspacing = 0.2 )
plt.legend(loc = "best", ncol = 2, **spacing_kwargs)
plt.tight_layout(pad = 0.1)
plt.savefig(fig_dir + "qudit_axes.pdf")
