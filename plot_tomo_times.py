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
size_fits = 20

##################################################

def max_dim(file):
    pieces = file.replace(".txt","").split("/")[-1].split("_")
    return pieces[-1].split("-")[-1]
def get_data(tag):
    files = sorted(glob.glob(data_dir + f"times_{tag}_d*.txt"), key = max_dim)
    data = np.vstack([ np.loadtxt(file) for file in files ])
    return data[:,0], data[:,1]

plt.figure(figsize = figsize)

plot_params = [ ( "CB", r"$\mathcal{S}_V$", "o", "k" ),
                ( "QB", r"$\epsilon_V$", ".", "tab:blue" ),
                ( "RE", r"$\mathcal{E}_V(\rho)$", ".", "tab:orange" ) ]

for tag, label, marker, color in plot_params:
    sizes, times = get_data(tag)
    plt.loglog(sizes, times, marker, color = color, label = label)

    size_lims = plt.gca().get_xlim()
    time_lims = plt.gca().get_ylim()
    fit_idx = slice(-size_fits,-1)
    fit_args = [ np.log(sizes[fit_idx]), np.log(times[fit_idx]) ]
    fit, cov = np.polyfit(*fit_args, deg = 1, cov = True)
    plt.loglog(size_lims, np.exp(fit[0]*np.log(size_lims) + fit[1]), "--", color = color)
    plt.gca().set_xlim(size_lims)
    plt.gca().set_ylim(time_lims)

plt.xlabel(r"qudit dimension $(d)$")
plt.ylabel(r"seconds $(t)$")
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig(fig_dir + "qudit_times.pdf")
