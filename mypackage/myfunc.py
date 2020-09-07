import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import glob
from datetime import datetime

import statsmodels.api as sm
from scipy.stats import pearsonr

def find_duplicate(alist):
	nondup, dup = [], []
	for item in alist:
		if item not in nondup: nondup.append(item)
		else:dup.append(item)
	dup = np.unique(dup).tolist()
	return dup

def countna(x, axis=None):
	return np.sum(np.isnan(x), axis=axis)

def corr(x, y):
	return np.corrcoef(x, y)[0, 1]

def autocorr(x, lag=1):
	x, y = x[:-lag], x[lag:]
	indx = find_common_valid(x, y)
	rho, pvalue = pearsonr(x[indx], y[indx])
	return rho, pvalue

def lm(x, y):
	x = sm.add_constant(x)
	model = sm.OLS(y, x).fit()
	return model

def rgb_to_hex(rgb):
	# rgb must be a list or tuple, such as [100, 120, 80]
	hex_str = '#%02x%02x%02x' % tuple(rgb)
	return hex_str

def digitize(value, bins):

	if np.isscalar(value): value = np.array([value])

	value = np.array(value)
	value_d = value.copy() * np.nan

	bins = np.sort(bins); n_bin = len(bins)

	value_d[value <= bins[0]] = 0
	for i in range(n_bin):
		value_d[value > bins[i]] = i + 1

	if len(value_d) == 1: value_d = value_d[0]
	return value_d


def movfunc(ndarr, k, axis=-1, func=np.mean):
    indx_all_dim = [slice(0, j) for j in ndarr.shape]
    ndarr_list = []
    for i in range(k):
        eindx = ndarr.shape[axis] - k + i + 1 
        indx_all_dim[axis] = slice(i, eindx)
        ndarr_list.append(ndarr[indx_all_dim])
    ndarr_sm = func(np.stack(ndarr_list), axis=0)
    return ndarr_sm


def find_nn(x, xref):

	x = np.array([x]) if np.isscalar(x) else np.array(x)
	xref = np.array(xref)

	nrow, ncol = len(x), len(xref)
	dist = np.abs(np.tile(x[:, np.newaxis], [1, ncol]) - np.tile(xref[np.newaxis, :], [nrow, 1]))
	nn_indx = np.argmin(dist, axis=1)
	nn = xref[nn_indx]

	output = (nn[0], nn_indx[0]) if len(nn)==1 else (nn, nn_indx)    
	return output

def find_common_valid(*args):

	vars_len = np.array([len(item) for item in args])
	assert all(vars_len == vars_len[0])

	indx = np.ones(vars_len[0]).astype(bool)
	for item in args:
		indx2 = ~np.isnan(item) & ~np.isinf(item)
		indx = indx & indx2
	return indx
