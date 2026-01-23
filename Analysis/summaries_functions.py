'''
This script includes functions for summary analysis of fluxes around
heatwaves.
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
    before_flux = []
    during_flux = []
    after_flux = []
    
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
        before_flux.append(flux_df[(flux_df.date.isin(before_dates)) & (flux_df.Site==site)][flux_name].mean())
        during_flux.append(flux_df[(flux_df.date.isin(during_dates)) & (flux_df.Site==site)][flux_name].mean())
        after_flux.append(flux_df[(flux_df.date.isin(after_dates)) & (flux_df.Site==site)][flux_name].mean())
        
    # Assign these to the dataframe and return!
    heatwaves_df[flux_name + "_before"] = before_flux
    heatwaves_df[flux_name + "_during"] = during_flux
    heatwaves_df[flux_name + "_after"] = after_flux
    
    return heatwaves_df