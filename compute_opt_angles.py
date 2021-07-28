#!/usr/bin/env python3

##########################################################################################
# For given spin qudit dimensions, this script computes the quantum error scale associated
# with different choices of the polar angle that must be chosen to carry out the
# tomography protocol in https://doi.org/10.1016/0003-4916(68)90035-3
# usage: python3 compute_opt_angles.py [list of dimensions]
##########################################################################################

import os, sys, time
import scipy.optimize
import numpy as np

genesis = time.time()

save_data = "save" in sys.argv
if save_data: sys.argv.remove("save")

# qudit dimensions from command-line inputs
dims = np.sort(np.array(sys.argv[1:], dtype = int))

theta_vals = np.linspace(0, np.pi/2, 501)[1:-1] # polar angles

# determine the directory and file for saving data
data_dir = "./data/"
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

##########################################################################################
# import qudit tomography methods

import qudit_tomo_methods as tomo

random_axes = tomo.random_axes
meas_mat, meas_scales = tomo.get_meas_methods(max(dims))
error_scale = tomo.quantum_error_scale

print("precomputation finished")

##########################################################################################
# find optimal angles

def get_axes(theta, axis_num):
    phi_vals = np.linspace(0,2*np.pi,axis_num+1)[:-1]
    return np.array([np.ones(axis_num)*theta, phi_vals]).T

def theta_scale(theta, dim):
    if np.isclose(theta % (np.pi/2), 0): return 1e10
    axis_num = 2*dim-1
    axes = get_axes(theta, axis_num)
    _meas_scales = meas_scales(dim, axes)
    scale = error_scale(_meas_scales)
    if np.isnan(scale): return 1e10
    return scale

for dim in dims:
    print(dim)
    def _theta_scale(theta): return theta_scale(theta,dim)
    theta_scales = list(map(_theta_scale,theta_vals))

    opt_guess = theta_vals[np.argmin(theta_scales)]
    kwargs = dict( bounds = [(0,np.pi/2)] )
    optimum = scipy.optimize.minimize(_theta_scale, opt_guess, **kwargs)

    header = "angle, quantum_error_scale"
    footer = f"optimum: {optimum.x[0]} {optimum.fun}"
    kwargs = dict( header = header, footer = footer )
    data = np.array([theta_vals,theta_scales]).T
    np.savetxt(data_dir + f"angle_scales_d{dim}.txt", data, **kwargs)
