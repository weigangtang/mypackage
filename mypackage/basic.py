import numpy as np
import pandas as pd

import scipy.io as sio
import scipy.interpolate
import pickle
import xlsxwriter
from datetime import datetime
import shutil
import os


def sumtab(rhbn_flow, info_type=['dquality']):

    options = ['statistic', 'dquality', 'monthly_mean', 'monthly_missd']
    month_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # must be 2 level nested list
    def flat(nested_list): 
        return [item for sublist in nested_list for item in sublist]

    if not all([item in options for item in info_type]):
        print('info type not found!')
        pass
    else:

        col_names = []
        col_names.append(['Mean', 'STD', 'Flashiness'])
        col_names.append(['OBS Period', 'Valid Year', 'Valid Year 2', 'Num Gaps', 'Missing Days'])
        col_names.append(['Monthly Missing Days {}'.format(x) for x in month_abbr])
        col_names.append(['Monthly Mean {}'.format(x) for x in month_abbr])
        col_names_all =  flat(col_names)

        summary_tab = pd.DataFrame(columns=col_names_all)

        for sid in sorted(rhbn_flow):
            flow = rhbn_flow[sid]
            mflow = splityear(flow)
            
            flow_mean = np.nanmean(flow.values)
            flow_std = np.nanstd(flow.values)
            change_rate = np.nanmean(np.abs(np.diff(flow.values)))

            st_year = flow.index.year[0]
            ed_year = flow.index.year[-1]
            obs_dur = ed_year - st_year

            annual_missing_days = np.apply_along_axis(lambda x: sum(np.isnan(x)), 0, arr=mflow)
            num_valid_years = sum(annual_missing_days < 20)

            max_gap = find_annual_max_gap(mflow)
            num_valid_years_2 = sum(max_gap < 10)

            gapindx = find_gaps(flow.values)
            num_gaps = gapindx.shape[0]
            total_missing_days = sum(np.isnan(flow))

            monthly_groups = flow.groupby(flow.index.month)
            monthly_missing_days = monthly_groups.size() - monthly_groups.count()
            monthly_flow_mean = monthly_groups.mean()
            
            values = [flow_mean, flow_std, change_rate, obs_dur, num_valid_years, num_valid_years_2, num_gaps, total_missing_days]
            values = values + monthly_missing_days.tolist() + monthly_flow_mean.tolist()

            summary_row = pd.DataFrame([values], columns=col_names_all, index=[sid])
            summary_tab = summary_tab.append(summary_row)

        option_codes = [options.index(item) for item in info_type]
        selected_col_names = [col_names[i] for i in option_codes]
        selected_col_names = flat(selected_col_names)

        return summary_tab[selected_col_names]



def find_total_missd_in_period(rhbn_flow, start_date, end_date): 
    total_days = end_date.toordinal() - start_date.toordinal()
    rhbn_sid = []
    rhbn_missing_days = []
    for sid in rhbn_flow:
        flow = rhbn_flow[sid]
        rhbn_sid.append(sid)
        flow_seg = flow[(flow.Time >= start_date) & (flow.Time < end_date)] 
        missing_days = total_days - sum(~np.isnan(flow_seg.Flow)) 
        rhbn_missing_days.append(missing_days)  
    return pd.DataFrame.from_items([('sid', rhbn_sid), ('missd', rhbn_missing_days)])


def find_annual_missd_in_period(rhbn_flow, start_year, end_year, brk_date='01-01'):
    years = range(start_year, end_year)
    annual_missd_matrix = pd.DataFrame({}, index=years)
    for sid in rhbn_flow:
        flow = rhbn_flow[sid]
        mflow  = splityear(flow, brk_date)
        annual_missing_days = np.isnan(mflow).sum(0).to_frame()
        annual_missing_days.columns = [sid]
        annual_missd_matrix = annual_missd_matrix.join(annual_missing_days)   
    annual_missd_matrix[np.isnan(annual_missd_matrix)] = 365
    annual_missd_matrix = annual_missd_matrix.transpose()
    return annual_missd_matrix



def convert_mindx(year, mindx): 
    dates = (str(year) + '-' + mindx).tolist()
    dt = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
    return dt



def read_mat_cell_array(fpath, var): 
    
    rdata = sio.loadmat(fpath)
    carr = rdata[var]
    
    colname = [item[0] for item in carr[0]]
    content = []
    for row in carr[1:]:
        row = [item[0] for item in row]
        for i in range(0, len(row)): 
            if isinstance(row[i], np.ndarray): 
                row[i] = row[i][0]
        content.append(row)
        
    df = pd.DataFrame.from_records(content, columns=colname)
    return df
