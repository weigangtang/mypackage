# http://earthpy.org/flow.html

import pandas as pd
import numpy as np

from datetime import datetime

import ulmo

def importusgssite(siteno):
    sitename = {}
    sitename = ulmo.usgs.nwis.get_site_data(siteno, service="daily", period="all", methods='all')
    sitename = pd.DataFrame(sitename['00060:00003']['values'])
    sitename['dates'] = pd.to_datetime(pd.Series(sitename['datetime']))
    sitename.set_index(['dates'],inplace=True)
    sitename['Q'] = sitename['value'].astype(float)
    sitename['qcode'] = sitename['qualifiers']
    sitename = sitename.drop(['datetime','qualifiers','value'],axis=1)
    sitename = sitename.replace('-999999', np.nan)
    return sitename


def getusgssiteinfo(siteno):
    siteinfo = ulmo.usgs.nwis.get_site_data(siteno, service="daily", period="all", methods='all')
    siteinfo = pd.DataFrame(siteinfo['00060:00003']['site'])
    siteinfo['latitude'] = siteinfo.loc['latitude','location']
    siteinfo['longitude'] = siteinfo.loc['longitude','location']
    siteinfo['latitude'] = siteinfo['latitude'].astype(float)
    siteinfo['longitude'] = siteinfo['longitude'].astype(float)
    siteinfo = siteinfo.drop(['default_tz','dst_tz','srs','uses_dst','longitude'],axis=0)
    siteinfo = siteinfo.drop(['agency','timezone_info','location','state_code','network'],axis=1)
    return siteinfo

def read_hydat_flow_data(station_number, hydat_df): 

    # rdata = hydat_df[hydat_df['STATION_NUMBER']==station_number]
    # fc_indx = np.arange(11, 73, 2) # select the columns of flow values 

    rdata = hydat_df.loc[station_number, :]

    colnames = hydat_df.columns.tolist()
    scol = colnames.index('FLOW1')
    ecol = colnames.index('FLOW31')
    fc_indx = np.arange(scol, ecol+2, 2)

    if len(rdata) == 0:
        ts = pd.Series([])
    else:
        ts = []
        for index, row in rdata.iterrows(): 
            
            year, month = row['YEAR'], row['MONTH']
            n_day = row['NO_DAYS'] # number of days for the given month
            
            start_date = '{}-{}-01'.format(year, month)
            tf = pd.date_range(start=start_date, periods=n_day)
            
            indx = fc_indx[:n_day]
            flow = row[indx].values
            
            ts.append(pd.Series(flow, index=tf))

        ts = pd.concat(ts)
        ts = ts.sort_index().astype(float)
    
    return ts

