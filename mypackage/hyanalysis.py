import numpy as np
import pandas as pd

import os, sys
if sys.platform == 'linux':
    os.chdir('/home/weigangtang/Dropbox/Python')
else: 
    os.chdir('/users/weigangtang/Dropbox/Python')
from funclib.hyclean import convert_time_series


def splityear(ts, brk_date='01-01'):

    years = np.unique(ts.index.year)

    date_frame = '{}-01-01'
    start_date = date_frame.format(years[0])
    end_date = date_frame.format(years[-1]+1)
    tf = pd.date_range(start_date, end_date, freq='D')[:-1]

    thr = np.timedelta64(1, 'D')
    ts2 = convert_time_series(tf, ts, thr)

    leap_day_index = (ts2.index.month == 2) & (ts2.index.day == 29)
    ts2 = ts2.drop(ts2.index[leap_day_index])

    assert len(ts2) % 365 == 0 
    mflow = ts2.values.reshape([-1, 365])

    colname = ts2.index[:365].strftime('%m-%d')
    df_mflow = pd.DataFrame(mflow, index=years, columns=colname)

    return df_mflow

 
def generate_valid_year_matrix(df_hys):
    
    hys = df_hys.values
    syid = list(df_hys.index)
    
    sid_col =df_hys.index.get_level_values(0).values
    sid = np.unique(sid_col)
    n_sid = len(sid)
    
    years_col = df_hys.index.get_level_values(1).values
    years = np.unique(years_col)
    years = np.arange(years[0], years[-1]+1)
    n_year = len(years)
    
    valid_year_mat = np.zeros([n_sid, n_year]).astype(bool)
    for i in range(n_sid):
        js = years_col[sid_col==sid[i]] - years[0] # an array rather than scalar
        valid_year_mat[i, js] = True
    
    df_vymat = pd.DataFrame(valid_year_mat, index=sid, columns=years)
    return df_vymat
