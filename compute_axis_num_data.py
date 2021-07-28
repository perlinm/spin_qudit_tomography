#!/usr/bin/env python3

##########################################################################################
# For a given spin qudit dimension, this script generates random measurement axes and
# computes their associated quantum error scale.  Results are printed to a data file,
# together with the mean computation time for number of axes.
# usage: python3 compute_axis_num_data.py [dim]
##########################################################################################

import os, sys, time
import numpy as np

genesis = time.time()

save_data = "save" in sys.argv
if save_data: sys.argv.remove("save")

# get qudit dimension from command-line inputs
dim = int(sys.argv[1])

excess_scale = 5
num_points = 51
axis_nums = ( dim * np.linspace(0,excess_scale,num_points) ).astype(int) + 2*dim-1
axis_nums = np.unique(axis_nums)

sample_cap = 1000 # maximum number of times we choose a random set of measurement axes
time_cap = 300 # maximum time to run per qubit dimension, in seconds

seed = 0 # set random number seed. change at your own peril
np.random.seed(seed)

# determine the directory and file for saving data
data_dir = "./data/"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
data_file = data_dir + f"axis_num_data_d{dim}.txt"

while os.path.isfile(data_file):
    print("file exists:", data_file)
    data_file += ".new"

##########################################################################################
# import qudit tomography methods

import qudit_tomo_methods as tomo

random_axes = tomo.random_axes
_, meas_scales = tomo.get_meas_methods(dim)
def error_scale(axes):
    _meas_scales = meas_scales(dim, axes)
    return tomo.quantum_error_scale(_meas_scales)

##########################################################################################
# simulate!

if save_data:
    # initialize data file
    with open(data_file, "w") as file:
        file.write(f"# sample_cap: {sample_cap}\n")
        file.write(f"# time_cap: {time_cap} sec\n")
        file.write(f"# seed: {seed}\n")
        file.write(f"# axis_num, mean_time, error_scale\n")

for axis_num in axis_nums:

    # initialize the smallest error scale we've seen
    min_error_scale = np.inf

    # record serial runtime for the simulations, excluding the pre-computations above
    start = time.time()

    # run simulations with random axes
    for sample in range(sample_cap):
        _axes = random_axes(axis_num)
        _error_scale = error_scale(_axes)
        min_error_scale = min(min_error_scale, _error_scale)

        if time.time() - start > time_cap: break

    # determine the average simulation time, and save/print updates
    mean_time = ( time.time() - start ) / (sample+1)
    update = f"{axis_num} {mean_time} {min_error_scale}"

    print(update)
    if save_data:
        with open(data_file, "a") as file:
            file.write(update + "\n")
    sys.stdout.flush()

# print the total time it took to run everything
runtime = time.time() - genesis
runtime_text = f"total runtime: {runtime}"
print(runtime_text)
if save_data:
    with open(data_file, "a") as file:
        file.write(f"# {runtime_text}\n")
