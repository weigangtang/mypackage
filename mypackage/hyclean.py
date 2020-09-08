import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from scipy.interpolate import interp1d


def find_gaps(arr):
	arr = np.concatenate((np.array([1]), arr, np.array([1])))
	d = np.diff(np.isnan(arr).astype(int))
	sindx = np.where(d==1)[0]
	eindx = np.where(d==-1)[0]
	outarr = np.vstack([sindx, eindx]).transpose()
	return outarr
# extract gaps: [arr[item[0]:item[1]] for item in outarr]
# get length of gaps: np.diff(outarr, axis=1)


def fill_small_gaps(arr, mglen=10): 
	gapindx = find_gaps(arr) 
	gaplens = np.diff(gapindx, axis=1) 
	n_gap = gapindx.shape[0]

	outarr = quick_interp(arr)
	for [sindx, eindx] in gapindx: 
		if eindx-sindx > mglen:
			outarr[sindx:eindx] = np.nan
	return outarr

def quick_interp(arr, method='linear', rm_nan=False): 

	# rm_nan determine if remove NaNs at HEAD and TAIL

	sindx, eindx = 0, len(arr)-1
	while np.isnan(arr[sindx]): sindx += 1
	while np.isnan(arr[eindx]): eindx -= 1

	x = np.arange(eindx-sindx+1)
	y = arr[sindx:eindx+1]    

	interp_func = interp1d(x[~np.isnan(y)], y[~np.isnan(y)], kind=method)
	arr[sindx:eindx+1] = interp_func(x)

	if rm_nan: 
		return arr[sindx:eindx+1]
	else:
		return arr


def shorten_time_series(ts, winsize=1, valid_days_thr=1):

	assert winsize >= valid_days_thr

	n = len(ts)
	mat = np.zeros([n+winsize-1, winsize]).astype(bool)
	for i in range(winsize): mat[i:n+i, i] = ~np.isnan(ts.values)

	y1 = np.sum(mat[winsize-1:, :], axis=1)
	y2 = np.sum(mat[:n, :], axis=1)

	sindx, eindx = 0, n-1
	while (np.isnan(ts.values[sindx])) | (y1[sindx] < valid_days_thr): sindx += 1
	while (np.isnan(ts.values[eindx])) | (y2[eindx] < valid_days_thr): eindx -= 1

	return ts[sindx:eindx+1]


def convert_time_series(tf, ts, thr):

	# thr in unit of second
	thr = pd.Timedelta(10, 's')

	ts = ts.sort_index()

	n_step = len(tf)

	brk_ticks = list(range(0, n_step, 200))
	rem = n_step - brk_ticks[-1]
	if  rem < 100: brk_ticks[-1] += rem
	else: brk_ticks = brk_ticks + [n_step]

	out_ts = pd.Series(np.zeros(n_step)*np.nan, index=tf)
	for sindx, eindx in zip(brk_ticks[:-1], brk_ticks[1:]):

		tf2 = tf[sindx:eindx]
		ts2 = ts[(ts.index >= tf2[0]) & (ts.index <= tf2[-1])]

		xx, yy = np.meshgrid(ts2.index, tf2)
		d = np.abs(xx - yy, dtype='timedelta64[s]')

		vmat = np.vstack([ts2.values] * len(tf2))
		vmat[d >= thr] = np.nan
		out_ts[sindx:eindx] = np.nanmean(vmat, axis=1)

	return out_ts


def check_time_step(ts):
	is_evenly_spaced = np.sum(np.abs(np.diff(np.diff(ts.index)))).astype(int) == 0
	return is_evenly_spaced


def find_adh_max_gap(hys): 
	nrow = hys.shape[0]
	max_gaplen = np.zeros(nrow)
	for i in range(nrow):
		se_indx = find_gaps(hys[i, :])
		if len(se_indx) > 0:
			max_gaplen[i] = np.max(np.diff(se_indx, axis=1))
	return max_gaplen 


def remove_leap_days(ts):
	out_ts = ts[(ts.index.month != 2) | (ts.index.day != 29)]
	return out_ts

