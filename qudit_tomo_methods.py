#!/usr/bin/env python3

import sys, time, functools
import scipy, scipy.linalg
import numpy as np

import wigner, py3nj

_status_updates = False

##########################################################################################
# general simulation methods

# get squared singular values of a matrix
def get_sqr_svdvals(matrix):
    # return scipy.linalg.svdvals(matrix)**2 # for some reason this is slower...
    if matrix.shape[0] < matrix.shape[1]:
        MM = matrix @ matrix.conj().T
    else:
        MM = matrix.conj().T @ matrix
    return scipy.linalg.eigvalsh(MM)

# get diagonal bands of a matrix
def diagonals(mat, band_min = None, band_max = None):
    rows, cols = mat.shape
    fill = np.zeros(((cols - 1), cols), dtype = mat.dtype)
    stacked = np.vstack((mat, fill, mat))
    major_stride, minor_stride = stacked.strides
    strides = major_stride, minor_stride * (cols + 1)
    shape = (rows + cols - 1, cols)
    reversed_diags = np.lib.stride_tricks.as_strided(stacked, shape, strides)
    diags = np.roll(np.flipud(reversed_diags), 1, axis = 0)
    if band_min == None: band_min = -cols
    if band_max == None: band_max = +rows
    bands = band_max - band_min
    return np.roll(diags, -band_min, axis = 0)[:bands+1,:]

# generate random axes on the sphere by uniform sampling
def random_axes(axis_num):
    axes = np.random.rand(axis_num,2)
    axes[:,0] = np.arccos(2*axes[:,0]-1) # polar angles
    axes[:,1] *= 2*np.pi # azimuthal angles
    return axes

# generate a random (probably unphysical) qudit "state",
# represented by a point within a `dim^2`-dimensional hypersphere
def random_state(dim):
    point = np.array([ np.random.normal() for _ in range(dim**2) ])
    point /= np.sqrt( np.random.exponential() + sum(point**2) )
    # organize components into (2L+1)-sized vectors of "degree" L < dim
    return { LL : point[ LL**2 : LL**2 + 2*LL+1 ] for LL in range(dim) }

# run a batch of independent jobs, possibly in parallel (not yet implemented)
def compute_batch(function, args):
    args_list = list(args)
    values = map(function, args_list)
    return { arg : value for arg, value in zip(args_list, values) }

##########################################################################################
# methods to compute a "measurement matrix" of all D^L_{0,m}(v) with a fixed degree L
# where: v = (alpha,beta) is a point on the sphere at azimuth/polar angles (alpha,beta)
#        D^L_{mn}(v) = <Lm| e^{-i beta S_y} * e^{-i alpha S_z} |Ln>
#        S_y and S_z are respectively spin-y and spin-z operators
#        |Lm> is a state of a spin-L particle with spin projection m onto the z axis
#
# the basic idea behind our methods is to decompose D^L_{0,m}(v) into:
#     <L,0| exp(+i pi/2 S_y) * e^{-i beta S_z} * exp(-i pi/2 S_y) * e^{-i alpha S_z}

# methods to pre-compute spin values and pi/2 rotation matrices exp(-i pi/2 S_y)
def _spin_vals(LL):
    return np.arange(-LL, LL+1)
def _rot_z_to_x(LL, status_updates = _status_updates):
    if status_updates:
        print("rot:", LL)
        sys.stdout.flush()
    MM = _spin_vals(LL)[:-1]
    diag_vals = np.sqrt(LL*(LL+1)-MM*(MM+1))
    S_m = np.diag(diag_vals, 1)
    S_y = ( S_m - S_m.T ) * 1j/2
    return scipy.linalg.expm(-1j * np.pi/2 * S_y)

# get methods to compute
# (1) a fixed-degree measurement matrix, and
# (2) all fixed-degree error scales (for many qudit dimensions)
def get_meas_methods(max_dim):
    # pre-compute spin values and pi/2 rotation matrices exp(-i pi/2 S_y)
    spin_vals = compute_batch(_spin_vals, range(max_dim-1,-1,-1))
    rot_z_to_x = compute_batch(_rot_z_to_x, range(max_dim-1,-1,-1))

    # save the vector `<L,0| exp(+i pi/2 S_y)` for all (relevant) L
    # every other entry in this vector is zero, so remove it preemptively
    rot_zero_vecs = [ rot_z_to_x[LL].T[LL,:][::2] for LL in range(max_dim) ]

    # collect all exp(-i pi/2 S_y), skipping every other row because we won't need it
    pulse_mats = [ rot_z_to_x[LL][::2,:] for LL in range(max_dim) ]

    del rot_z_to_x # delete data we no longer need to save memory

    # method that constructs a vector of D^L_{0,m}(v) for all |m| <= L
    def rot_mid(LL, axis):
        phases_beta = np.exp(-1j * axis[0] * spin_vals[LL][::2])
        phases_alph = np.exp(-1j * axis[1] * spin_vals[LL])
        return ( ( rot_zero_vecs[LL] * phases_beta ) @ pulse_mats[LL] ) * phases_alph

    # construct a fixed-degree measurement matrix
    def meas_mat(LL, axes):
        return np.array([ rot_mid(LL, axis) for axis in axes ])

    # get the classical fixed-degree error scale
    def meas_scale(LL, axes):
        sqr_svdvals = get_sqr_svdvals(meas_mat(LL, axes))
        inv_sum = sum([ 1/val for val in sqr_svdvals if not np.isclose(val,0) ])
        return np.sqrt(inv_sum)

    # get all fixed-degree error scales
    def meas_scales(dim, axes):
        _meas_scale = functools.partial(meas_scale, axes = axes)
        return compute_batch(_meas_scale, range(dim-1,-1,-1))

    return meas_mat, meas_scales

##########################################################################################
# methods to compute diagonal bands of an "inverted" matrix of structure constants

# matrix of wigner-3j coefficients
def wigner_3j_mat(ll, LL):
    matrix = np.zeros((2*ll+1,)*2)
    for mm in range(-ll,ll+1):
        start, end, vals = wigner.wigner_3j_m(ll, ll, LL, mm)
        matrix[mm+ll, int(start)+ll : int(end)+ll+1] = vals
    return matrix

# diagonal bands of an "inverted" matrix of wigner-3j factors
def inv_3j_band_mat(labels, status_updates = _status_updates):
    ll, LL = labels
    if status_updates and ll == LL:
        print("inv_3j:", LL)
        sys.stdout.flush()
    mat = wigner_3j_mat(ll, LL)
    signs = np.array([ (-1)**mm for mm in range(-ll,ll+1) ])
    inv_mat = signs[:,None] * np.flipud(mat)
    return diagonals(inv_mat, -LL, LL)

# prefactors that convert between wigner-3j coefficients and structure constants
def get_prefactors(labels):
    dim, ll, LL = [ np.array(vals) for vals in zip(*labels) ]
    wigner_6j_degrees = 2 * np.array([ ll, ll, LL ])
    wigner_6j_spins = np.array([ dim, dim, dim ]) - 1
    wigner_6j_factors = py3nj.wigner6j(*wigner_6j_degrees.astype(int), *wigner_6j_spins)
    prefactors = (-1)**(dim-1+LL) * (2*ll+1) * np.sqrt(2*LL+1) * wigner_6j_factors
    return { label : prefactor for label, prefactor in zip(labels, prefactors) }

def get_struct_method(max_dim):
    labels = [ ( dim, ll, LL )
               for dim in range(max_dim,-1,-1)
               for ll in range(dim-1,-1,-1)
               for LL in range(min(dim-1,2*ll),-1,-1) ]
    prefactors = get_prefactors(labels)

    labels = [ ( ll, LL )
               for ll in range(max_dim-1,-1,-1)
               for LL in range(min(max_dim-1,2*ll),-1,-1) ]
    inv_3j_bands = compute_batch(inv_3j_band_mat, labels)

    def inv_struct_bands(dim, ll, LL):
        return prefactors[dim,ll,LL] * inv_3j_bands[ll,LL]

    return inv_struct_bands

##########################################################################################
# methods to compute classical and quantum error scales

# compute the classical error scale
def classical_error_scale(meas_scales):
    return np.sqrt(sum( degree_scale**2 for degree_scale in meas_scales.values() ))

# squared "gamma" factors that appear in the quantum error scale
def sqr_gamma(dim, ll):
    ss = (dim-1)/2
    mm_min, mm_max, wigner_3j_vals = wigner.wigner_3j_m(ll, ss, ss, 0)
    signs = np.array([ (-1)**(2*ll+ss-mm) for mm in np.arange(mm_min, mm_max+1) ])
    signed_vals = signs * wigner_3j_vals
    half_span = ( max(signed_vals) - min(signed_vals) ) / 2
    return (2*ll+1) * half_span**2

# compute the quantum error scale
def quantum_error_scale(meas_scales):
    dim = len(meas_scales)
    return np.sqrt(sum( sqr_gamma(dim, ll) * degree_scale**2
                        for ll, degree_scale in meas_scales.items() ))

##########################################################################################
# methods to compute the root-mean-squared reconstruction error

# compute the fixed-degree noise matrix
def noise_mat(LL, meas_mat):
    mat = meas_mat(LL)
    vecs, vals, _ = scipy.linalg.svd(mat, full_matrices = False)
    diag_vals = np.sum( abs(vecs/vals[None,:])**2, axis = 1)
    return ( mat.conj().T * diag_vals[None,:] ) @ mat

# compute all noise matrices
def noise_mats(dim, meas_mat):
    _noise_mat = functools.partial(noise_mat, meas_mat = meas_mat)
    return compute_batch(_noise_mat, range(dim-1,-1,-1))

# compute diagonal bands of transposed noise matrices
def noise_band_mat(LL, noise_mats):
    return diagonals(noise_mats[LL].T)

# compute the fixed-degree components of the "chi state" in the "degree-order" basis
def degree_chi_state(LL, dim, noise_band_mats, inv_struct_bands):
    min_ll = (LL+1)//2
    chi_state = np.zeros(2*LL+1, dtype = complex)
    for ll in range(min_ll, dim):
        noise_band_mat = noise_band_mats[ll]
        shift = -noise_band_mat.shape[0]//2 + LL
        noise_band_mat_mid = np.roll(noise_band_mat, shift, axis = 0)[:2*LL+1,:]
        chi_state += ( noise_band_mat_mid * inv_struct_bands(dim,ll,LL) ).sum(axis = 1)
    return chi_state

# compute the full "chi state" in the "degree-order" basis
def chi_state(dim, noise_mats, inv_struct_bands):
    _noise_band_mat = functools.partial(noise_band_mat, noise_mats = noise_mats)
    noise_band_mats = compute_batch(_noise_band_mat, range(dim-1,-1,-1))
    kwargs = dict( dim = dim,
                   noise_band_mats = noise_band_mats,
                   inv_struct_bands = inv_struct_bands )
    _degree_chi_state = functools.partial(degree_chi_state, **kwargs)
    return compute_batch(_degree_chi_state, range(dim-1,-1,-1))

# compute contribution to squared reconstruction error from a single degree
def sqr_degree_error(LL, state, chi_state, noise_mats):
    pos = chi_state[LL].conj() @ state[LL]
    neg = state[LL].conj() @ ( noise_mats[LL] @ state[LL] )
    return pos - neg

# compute the reconstruction error for a state in the "degree-order" basis
def recon_error(state, meas_mat, inv_struct_bands):
    dim = max(state.keys()) + 1
    _noise_mats = noise_mats(dim, meas_mat)
    _chi_state = chi_state(dim, _noise_mats, inv_struct_bands)
    kwargs = dict( state = state, chi_state = _chi_state, noise_mats = _noise_mats )
    _sqr_degree_error = functools.partial(sqr_degree_error, **kwargs)
    sqr_degree_errors = compute_batch(_sqr_degree_error, range(dim-1,-1,-1))
    return np.sqrt(sum( sqr_error for sqr_error in sqr_degree_errors.values() ))
