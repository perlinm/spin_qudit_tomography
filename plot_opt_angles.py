#!/usr/bin/env python3

import os, glob
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

data_dir = "./data/"
fig_dir = "./figures/"

if not os.path.isdir(fig_dir):
    os.makedirs(fig_dir)

params = { "font.size" : 10,
           "text.usetex" : True }
plt.rcParams.update(params)

##########################################################################################
# plot quantum error scale as a function of the polar angle for a few qudit dimensions

dims = [ 2, 3, 4, 10, 20, 40 ]

kwargs = dict( figsize = (7,4), sharex = True, sharey = True )
figure, axes = plt.subplots(2, 3, **kwargs)

for dim, axis in zip(dims, axes.ravel()):
    angles, scales = np.loadtxt(data_dir + f"angle_scales_d{dim}.txt", unpack = True)
    axis.semilogy(angles/(np.pi/2), scales/dim, "k")

    # add subplot label
    text = f"$d={dim}$"
    method_box = dict(boxstyle = "round", facecolor = "white", alpha = 1)
    axis.text(0.9, 0.9, text, transform = axis.transAxes, bbox = method_box,
              verticalalignment = "top", horizontalalignment = "right")

plt.xlim(0, 1)
plt.ylim(0.3, 2000)
for axis in axes[:,0]:
    axis.set_ylabel(r"$\epsilon_\theta/d$")
for axis in axes[-1,:]:
    axis.set_xlabel(r"$\frac{\theta}{\pi/2}$")

plt.tight_layout()
plt.savefig(fig_dir + "angle_sweep.pdf")

##########################################################################################
# plot optimal angle as function of the qudit dimension

files = glob.glob(data_dir + "angle_scales*")
dims = np.zeros(len(files))
opt_angles = np.zeros(len(files))
for idx, file in enumerate(files):
    dims[idx] = int(file.split("d")[-1].split(".")[0])
    with open(file,"r") as file_text:
        for line in file_text:
            if "optimum" in line:
                opt_angles[idx] = float(line.split()[-2])
                break

dims, opt_angles = zip(*sorted([ (dim,angle) for dim, angle in zip(dims,opt_angles) ]))
dims = np.array(dims)
opt_angles = np.array(opt_angles)
print(np.log10(dims))
print(np.log10(opt_angles))

def fit_func(dim, scalar):
    return np.pi/2 * (1 - 1/(scalar*dim))

opt_val, opt_var = scipy.optimize.curve_fit(fit_func, dims, opt_angles)
opt_val = opt_val[0]
opt_var = opt_var[0,0]
print("scaling param:", opt_val, "+/-", np.sqrt(opt_var))

plt.figure(figsize = (3,2))
plt.loglog(dims, 1 - opt_angles / (np.pi/2), "r.", label = "data")
plt.loglog(dims, 1/(opt_val*dims), "k--", label = "fit")

plt.xlim(right = 100)
plt.xlabel(r"$d$")
plt.ylabel(r"$1-\frac{\theta_{\mathrm{opt}}}{\pi/2}$")
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig(fig_dir + "opt_angles.pdf")
