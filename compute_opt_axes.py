#!/usr/bin/env python3

##########################################################################################
# For a given spin qudit dimension, this script first generates random measurement axes,
# then takes the best found measurment axes and tries to further optimize them by
# minimizing their associated quantum error scale.
# usage: python3 compute_opt_axes.py [dim]
##########################################################################################

import os, sys, time
import scipy.optimize
import numpy as np

genesis = time.time()

save_data = "save" in sys.argv
if save_data: sys.argv.remove("save")

# get qudit dimension from command-line inputs
dim = int(sys.argv[1])

axis_num = 3*dim # number of axes for which to save best known angles
sample_num = 10**4 # number of times we choose a random set of measurement axes

seed = 0 # set random number seed. change at your own peril
np.random.seed(seed)

# determine the directory and file for saving data
data_dir = "./data/"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
data_file = data_dir + f"best_axes_d{dim}.txt"

while os.path.isfile(data_file):
    print("file exists:", data_file)
    data_file += ".new"

##########################################################################################
# import qudit tomography methods

import qudit_tomo_methods as tomo

random_axes = tomo.random_axes
_, meas_scales = tomo.get_meas_methods(dim)
def error_scale(axes):
    axes.shape = (axis_num,2)
    _meas_scales = meas_scales(dim, axes)
    return tomo.quantum_error_scale(_meas_scales)

##########################################################################################
# simulate!

# initialize the smallest error scale and best axes we've seen
min_error_scale = np.inf
best_axes = None

# find good random axes by sampling
for sample in range(sample_num):
    _axes = random_axes(axis_num)
    _error_scale = error_scale(_axes)

    if _error_scale < min_error_scale:
        min_error_scale = _error_scale
        best_axes = _axes

print("done sampling")

# optimize best found axes
optimum = scipy.optimize.minimize(error_scale, best_axes.ravel())
best_axes = optimum.x.reshape((axis_num,2))
min_error_scale = optimum.fun

header  = f"best known angles for {axis_num} measurement axes" + "\n"
header += f"quantum error scale: {min_error_scale}" + "\n"
header +=  "polar angle, azimuthal angle"
np.savetxt(data_file, best_axes, header = header)
