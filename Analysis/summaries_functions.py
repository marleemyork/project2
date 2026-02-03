'''
This script includes functions for summary analysis of fluxes around
heatwaves. The first calculates one lag, the second was made by chat to calculate 
for multiple lags.
'''
from datetime import datetime, timedelta
import pandas as pd

def calc_flux_avg(flux_name, heatwaves_df, flux_df, before_lag, after_lag):
    """
    Parameters
    ----------
    flux_name : STR
        DESCRIPTION. String name of the flux you want to calculate for
    heatwaves_df : pd.DataFrame
        DESCRIPTION. Includes all heatwaves including start_date, end_date
    flux_df : pd.DataFrame
        DESCRIPTION. Includes columns including date and "flux_name"
    before_lag : INT
        DESCRIPTION. Number of days prior to heatwave you want in before period
    after_lag : INT
        DESCRIPTION. Number of days after heatwave you want in after period

    Returns
    -------
    heatwaves_df : TYPE
        DESCRIPTION. Same dataframe with added columns for average flux over
        before, during, and after heatwave periods

    """
    
    # Find the dates of before and after heatwave periods we want to calculate for
    heatwaves_df['before_hw'] = heatwaves_df.start_dates - timedelta(days=before_lag)
    heatwaves_df['after_hw'] = heatwaves_df.end_dates + timedelta(days=after_lag)
    
    # Initialize list to store the period averages we calculate
    before_flux_avg = []
    during_flux_avg = []
    after_flux_avg = []
    before_flux_std = []
    during_flux_std = []
    after_flux_std = []
    
    # Now loop through each heatwave and calculate the period averages
    for k in range(0,len(heatwaves_df)):
        
        # Isolate start and end dates for each heatwave
        site = heatwaves_df.iloc[k].Site
        before = heatwaves_df.iloc[k].before_hw
        start = heatwaves_df.iloc[k].start_dates
        end = heatwaves_df.iloc[k].end_dates
        after = heatwaves_df.iloc[k].after_hw
        
        # Create a date range for each period
        before_dates = pd.date_range(before, start)
        during_dates = pd.date_range(start, end)
        after_dates = pd.date_range(end, after)
        
        # Calculate average and append to list
        before_flux_avg.append(flux_df[(flux_df.date.isin(before_dates)) & (flux_df.Site==site)][flux_name].mean())
        during_flux_avg.append(flux_df[(flux_df.date.isin(during_dates)) & (flux_df.Site==site)][flux_name].mean())
        after_flux_avg.append(flux_df[(flux_df.date.isin(after_dates)) & (flux_df.Site==site)][flux_name].mean())
        
        # Calculate std and append to list
        before_flux_std.append(flux_df[(flux_df.date.isin(before_dates)) & (flux_df.Site==site)][flux_name].std())
        during_flux_std.append(flux_df[(flux_df.date.isin(during_dates)) & (flux_df.Site==site)][flux_name].std())
        after_flux_std.append(flux_df[(flux_df.date.isin(after_dates)) & (flux_df.Site==site)][flux_name].std())
        
    # Assign these to the dataframe and return!
    heatwaves_df[flux_name + "_before_avg"] = before_flux_avg
    heatwaves_df[flux_name + "_during_avg"] = during_flux_avg
    heatwaves_df[flux_name + "_after_avg"] = after_flux_avg
    heatwaves_df[flux_name + "_before_std"] = before_flux_std
    heatwaves_df[flux_name + "_during_std"] = during_flux_std
    heatwaves_df[flux_name + "_after_std"] = after_flux_std
    
    return heatwaves_df

def calc_flux_avg_multi_lag(flux_name, heatwaves_df, flux_df, before_lags=None, after_lags=None):
    """
    Calculate mean and std of a flux before, during, and after heatwaves
    for multiple before/after lag windows.

    before_lags and after_lags should be lists like [5,10,15,20,25,30]
    """

    if before_lags is None:
        before_lags = []
    if after_lags is None:
        after_lags = []

    # DURING period is the same regardless of lag, so compute once
    during_flux_avg = []
    during_flux_std = []

    for k in range(len(heatwaves_df)):
        site = heatwaves_df.iloc[k].Site
        start = heatwaves_df.iloc[k].start_dates
        end   = heatwaves_df.iloc[k].end_dates

        during_dates = pd.date_range(start, end)

        vals = flux_df[(flux_df.date.isin(during_dates)) &
                       (flux_df.Site == site)][flux_name]

        during_flux_avg.append(vals.mean())
        during_flux_std.append(vals.std())

    heatwaves_df[f"{flux_name}_during_avg"] = during_flux_avg
    heatwaves_df[f"{flux_name}_during_std"] = during_flux_std

    # ----- BEFORE LAGS -----
    for lag in before_lags:
        before_flux_avg = []
        before_flux_std = []

        for k in range(len(heatwaves_df)):
            site  = heatwaves_df.iloc[k].Site
            start = heatwaves_df.iloc[k].start_dates
            before = start - timedelta(days=lag)

            before_dates = pd.date_range(before, start)

            vals = flux_df[(flux_df.date.isin(before_dates)) &
                           (flux_df.Site == site)][flux_name]

            before_flux_avg.append(vals.mean())
            before_flux_std.append(vals.std())

        heatwaves_df[f"{flux_name}_before_avg_{lag}"] = before_flux_avg
        heatwaves_df[f"{flux_name}_before_std_{lag}"] = before_flux_std

    # ----- AFTER LAGS -----
    for lag in after_lags:
        after_flux_avg = []
        after_flux_std = []

        for k in range(len(heatwaves_df)):
            site = heatwaves_df.iloc[k].Site
            end  = heatwaves_df.iloc[k].end_dates
            after = end + timedelta(days=lag)

            after_dates = pd.date_range(end, after)

            vals = flux_df[(flux_df.date.isin(after_dates)) &
                           (flux_df.Site == site)][flux_name]

            after_flux_avg.append(vals.mean())
            after_flux_std.append(vals.std())

        heatwaves_df[f"{flux_name}_after_avg_{lag}"] = after_flux_avg
        heatwaves_df[f"{flux_name}_after_std_{lag}"] = after_flux_std

    return heatwaves_df

