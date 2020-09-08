import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib

from copy import deepcopy

import statsmodels.api as sm

from scipy.stats import pearsonr
from scipy.special import ndtri, ndtr

from myfunc import digitize, find_common_valid
from hyclean import check_time_step


def mk_test(t, x, eps=10**(-6), alpha=0.05, Ha='upordown'):
    """
    this codes copy from https://up-rs-esp.github.io/mkt/_modules/mkt.html#test
    don't forget reference

    Runs the Mann-Kendall test for trend in time series data.

    Parameters
    ----------
    t : 1D numpy.ndarray
        array of the time points of measurements
    x : 1D numpy.ndarray
        array containing the measurements corresponding to entries of 't'
    eps : scalar, float, greater than zero
        least count error of measurements which help determine ties in the data
    alpha : scalar, float, greater than zero
        significance level of the statistical test (Type I error)
    Ha : string, options include 'up', 'down', 'upordown'
        type of test: one-sided ('up' or 'down') or two-sided ('updown')

    Returns
    -------
    MK : string
        result of the statistical test indicating whether or not to accept hte
        alternative hypothesis 'Ha'
    m : scalar, float
        slope of the linear fit to the data
    c : scalar, float
        intercept of the linear fit to the data
    p : scalar, float, greater than zero
        p-value of the obtained Z-score statistic for the Mann-Kendall test

    Raises
    ------
    AssertionError : error
                    least count error of measurements 'eps' is not given
    AssertionError : error
                    significance level of test 'alpha' is not given
    AssertionError : error
                    alternative hypothesis 'Ha' is not given

    """
    # assert a least count for the measurements x
    assert eps, "Please provide least count error for measurements 'x'"
    assert alpha, "Please provide significance level 'alpha' for the test"
    assert Ha, "Please provide the alternative hypothesis 'Ha'"

    # estimate sign of all possible (n(n-1)) / 2 differences
    n = len(t)
    sgn = np.zeros((n, n), dtype="int")
    for i in range(n):
        tmp = x - x[i]
        tmp[np.where(np.fabs(tmp) <= eps)] = 0.
        sgn[i] = np.sign(tmp)

    # estimate mean of the sign of all possible differences
    S = sgn[np.triu_indices(n, k=1)].sum()

    # estimate variance of the sign of all possible differences
    # 1. Determine no. of tie groups 'p' and no. of ties in each group 'q'
    np.fill_diagonal(sgn, eps * 1E6)
    i, j = np.where(sgn == 0.)
    ties = np.unique(x[i])
    p = len(ties)
    q = np.zeros(len(ties), dtype="int")
    for k in range(p):
        idx =  np.where(np.fabs(x - ties[k]) < eps)[0]
        q[k] = len(idx)
    # 2. Determine the two terms in the variance calculation
    term1 = n * (n - 1) * (2 * n + 5)
    term2 = (q * (q - 1) * (2 * q + 5)).sum()
    # 3. estimate variance
    varS = float(term1 - term2) / 18.

    # Compute the Z-score based on above estimated mean and variance
    if S > eps:
        Zmk = (S - 1) / np.sqrt(varS)
    elif np.fabs(S) <= eps:
        Zmk = 0.
    elif S < -eps:
        Zmk = (S + 1) / np.sqrt(varS)

    # compute test based on given 'alpha' and alternative hypothesis
    # note: for all the following cases, the null hypothesis Ho is:
    # Ho := there is no monotonic trend
    # 
    # Ha := There is an upward monotonic trend
    if Ha == "up":
        Z_ = ndtri(1. - alpha)
        if Zmk >= Z_:
            MK = "accept Ha := upward trend"
        else:
            MK = "reject Ha := upward trend"
    # Ha := There is a downward monotonic trend
    elif Ha == "down":
        Z_ = ndtri(1. - alpha)
        if Zmk <= -Z_:
            MK = "accept Ha := downward trend"
        else:
            MK = "reject Ha := downward trend"
    # Ha := There is an upward OR downward monotonic trend
    elif Ha == "upordown":
        Z_ = ndtri(1. - alpha / 2.)
        if np.fabs(Zmk) >= Z_:
            MK = "accept Ha := upward OR downward trend"
        else:
            MK = "reject Ha := upward OR downward trend"

    # ----------
    # AS A BONUS
    # ----------
    # estimate the slope and intercept of the line
    m = np.corrcoef(t, x)[0, 1] * (np.std(x) / np.std(t))
    c = np.mean(x) - m * np.mean(t)

    # ----------
    # AS A BONUS
    # ----------
    # estimate the p-value for the obtained Z-score Zmk
    if S > eps:
        if Ha == "up":
            p = 1. - ndtr(Zmk)
        elif Ha == "down":
            p = ndtr(Zmk)
        elif Ha == "upordown":
            p = 0.5 * (1. - ndtr(Zmk))
    elif np.fabs(S) <= eps:
        p = 0.5
    elif S < -eps:
        if Ha == "up":
            p = 1. - ndtr(Zmk)
        elif Ha == "down":
            p = ndtr(Zmk)
        elif Ha == "upordown":
            p = 0.5 * (ndtr(Zmk))

    return MK, m, c, p


def digitize_pvalue(pvalue, pvalue_bin=[0.01, 0.05, 0.1]):
    
    n_bin = len(pvalue_bin)
    pvalue_d = digitize(pvalue, pvalue_bin)
    pvalue_d = n_bin - pvalue_d
    
    return pvalue_d


# THREE requirements must be satisfied:
# 1) no missing time step(s)
# 2) sufficient sample size (e.g. n > 10)
# 3) not constant (var > 0)
def satisfy_prewhiten_req(ts):
    no_missing_step = check_time_step(ts)
    sufficient_data = np.sum(find_common_valid(ts.values[1:], ts.values[:-1])) > 10
    non_zero_var = np.nanvar(ts.values) > 0
    return no_missing_step & sufficient_data & non_zero_var


def prewhiten_time_series(ts, lag=1):
    
    x, y = ts.values[:-lag], ts.values[lag:]
    indx = find_common_valid(x, y)
    
    rho, pvalue = pearsonr(x[indx], y[indx])
    
    ts_pwt = ts.copy()
    ts_pwt.values[lag:] = y - rho * x
    return ts_pwt


# Run Trend Test -------------------------------------------------------------
# ----------------------------------------------------------------------------

def run_mktest(x, y):
    
    assert len(x) == len(y)
    
    indx = find_common_valid(x, y)
    x, y = x[indx], y[indx]
    
    M, slp, intp, pvalue = mk_test(x, y)
    
    if np.nanvar(y)==0: slp, intp = 0, y[0]
    if np.isnan(pvalue): pvalue = 0.5
    
    n = np.sum(indx)
    pvalue_d = digitize_pvalue(pvalue) * np.sign(slp)
    
    return slp, intp, pvalue, pvalue_d, n

# Plot Functions -------------------------------------------------------------
# ----------------------------------------------------------------------------
alpha_transform = lambda n: np.sum(np.arange(np.abs(n)+1)) * 0.1
color_transform = lambda x: ['blue', 'white', 'red'][np.sign(x).astype(int)+1]

def plot_time_series_with_trend(ts, slp, intp, pvalue_d, ax=None):

    if not isinstance(ax, matplotlib.axes._subplots.Axes): ax = plt.gca()

    t, y0 = ts.index, ts.values

    y = t * slp + intp

    alpha = alpha_transform(np.abs(pvalue_d))
    color = color_transform(slp)

    ax.plot(t, y0, color='limegreen', marker='o', markersize=5, linestyle='None')
    ax.plot(t, y, color='gold')
    ax.patch.set_facecolor(color)
    ax.patch.set_alpha(alpha=alpha)

    return ax

# MK Test Outcome ------------------------------------------------------------
# ----------------------------------------------------------------------------

class tensorframe:
    
    def __init__(self, tensor, *dim_name_list):

        assert len(tensor.shape) == len(dim_name_list)

        ndim = len(tensor.shape)

        for i in range(ndim):
            assert isinstance(dim_name_list[i], list)
        
        for i in range(ndim):
            assert tensor.shape[i] == len(dim_name_list[i])
        
        for i in range(ndim):
            exec('self.name{}d = dim_name_list[i]'.format(i+1)) 
        
        self.ndim = ndim
        self.tensor = tensor
        
    def get_dim_name_list(self):
        dim_name_list = []
        for i in range(self.ndim):
            exec('dim_name_list.append(self.name{}d)'.format(i+1))
        return dim_name_list

    def to_frame(self, *attr_list):
        
        n_attr = len(attr_list)
        assert (self.ndim - n_attr) == 2

        dim_name_list = self.get_dim_name_list()
        
        tindx = [slice(n) for n in self.tensor.shape]
        for i in range(self.ndim):
            dim_name = dim_name_list[i]
            for j in range(n_attr):
                attr = attr_list[j]
                if attr in dim_name:
                    tindx[i] = dim_name.index(attr)
        i_frdim = np.where([not np.isscalar(item) for item in tindx])[0]
        assert len(i_frdim) == 2
        rname = dim_name_list[i_frdim[0]]
        cname = dim_name_list[i_frdim[1]]
        df = pd.DataFrame(self.tensor[tindx], index=rname, columns=cname)
        return df

    def to_series(self, *attr_list):

        n_attr = len(attr_list)
        assert (self.ndim-n_attr) == 1

        dim_name_list = self.get_dim_name_list()
        
        tindx = [slice(n) for n in self.tensor.shape]
        for i in range(self.ndim):
            dim_name = dim_name_list[i]
            for j in range(n_attr):
                attr = attr_list[j]
                if attr in dim_name:
                    tindx[i] = dim_name.index(attr)
        i_frdim = np.where([not np.isscalar(item) for item in tindx])[0]
        assert len(i_frdim) == 1
        rname = dim_name_list[i_frdim[0]]
        sr = pd.Series(self.tensor[tindx], index=rname)
        return sr

    def count_nan(self, axis):
        return np.sum(np.isnan(self.tensor), axis=axis)

    def subset(self, select):
        # select is dictionary
        # for example: {'2d': ['mean', 'median', '7dmin']}

        dim_name_list = self.get_dim_name_list()

        for key in select:
            i_dim = int(key[:-1]) - 1
            assert i_dim < self.ndim

        for key in select:
            i_dim = int(key[:-1]) - 1
            all_attr = dim_name_list[i_dim]
            sel_attr = select[key]
            assert set(sel_attr).issubset(set(all_attr))

        tensor = self.tensor
        for key in select:
            i_dim = int(key[:-1]) - 1
            all_attr = dim_name_list[i_dim]
            sel_attr = select[key]
            sel_indx = [all_attr.index(item) for item in sel_attr]
            dim_name_list[i_dim] = sel_attr
            tensor = tensor.take(sel_indx, axis=i_dim)
        return tensorframe(tensor, *dim_name_list)

    def append(self, new_tframe, axis):

        assert self.ndim == new_tframe.ndim

        dim_name_list_cur = self.get_dim_name_list()
        dim_name_list_add = new_tframe.get_dim_name_list()

        for i in range(self.ndim):
            if i == axis:
                duplicated_names = set(dim_name_list_cur[i]).intersection(dim_name_list_add[i])
                assert len(duplicated_names) == 0
            else:
                assert dim_name_list_cur[i] == dim_name_list_add[i]

        dim_name_list = []
        for i in range(self.ndim):
            if i == axis:
                dim_name_list.append(dim_name_list_cur[i] + dim_name_list_add[i])
            else:
                dim_name_list.append(dim_name_list_cur[i])
        tensor = np.concatenate([self.tensor, new_tframe.tensor], axis=axis)
        return tensorframe(tensor, *dim_name_list)
        

    # def append(self, names_new, tensor_new, axis):
    #     assert len(names_new) == tensor_new.shape[axis]
        
    #     dim_name_list = self.get_dim_name_list()
    #     dim_name_list[axis] += names_new

    #     tensor = np.concatenate([self.tensor, tensor_new], axis=axis)
    #     return tensorframe(tensor, *dim_name_list)

