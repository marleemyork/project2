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

def DOY_climatology(df, var_name, smoothing_function="weighted_15"):
    
    '''
    EDITS: NEED TO REMOVE HEATWAVE DAYS FROM THIS AND SMOOTH THE VALUES BEFORE RETURNING
    
    Description: This function calculates an expected value for each DOY at each site
    for some variable. It will be used in cumulative deviation calculations. It
    excludes any heatwave days when calculating the expected value
    
    Parameters
    ----------
    df: pd.DataFrame
        DESCRIPTION. includes site, date, and daily values of variable var_name
    var_name : string
        DESCRIPTION. variable name in df that we are calculated expected value for
    smoothing_function: str
        DESCRIPTION. smoothing function you want to use for calculating expected flux value

    Returns
    -------
    new_df : TYPE
        DESCRIPTION. Same dataframe with added column with DOY climatology-based 
        expected value for var_name
    '''
    
    # Convert date to day of year format
    df["DOY"] = df.date.dt.dayofyear
    
    # Check if we have a heatwave indicator, if not tell us to add it
    if ("heatwave_indicator" not in df.columns):
        print("Please add a heatwave indictor to this df.")
        print("Use: df = add_heatwave_indicator(df,all_heatwaves_df)")
    else:
        print("You have the heatwave indicator column!")
    
    # Remove any colums with heatwaves
    nohw_df = df[df.heatwave_indicator==0]
    
    # Extract columns of df
    col_list = df.columns.tolist()
    col_list.append("DOY_"+var_name)
    
    # Create dataframe to store new df
    new_df = pd.DataFrame(columns=col_list)
    
    # Loop through each site
    for site in df.Site.unique():
        site_df = nohw_df[nohw_df.Site==site]
        
        # Calculate mean flux for each DOY
        expected_value = site_df.groupby('DOY')[var_name].mean().reset_index()
        expected_value.columns = ["DOY", "DOY_"+var_name]
        
        # Now we want to smooth this
        if (smoothing_function == "fourier"):
            expected_value["expected_"+var_name] = fourier_smooth_fft(expected_value["DOY_"+var_name], n_harmonics=3)
        elif (smoothing_function == "weighted_15"):
            expected_value["expected_"+var_name] = rolling_weighted_mean(pd.Series(expected_value["DOY_"+var_name]),window=15)
        else:
            print("You didn't provide a valid smoothing function. Try weighted_15 or fourier.")
        
        # Merge with site df
        site_df = pd.merge(site_df,expected_value,on="DOY",how="left")
        
        # Concat with other sites
        new_df = pd.concat([new_df,site_df])
        
    return new_df


def fourier_smooth_fft(y, n_harmonics=3):
    """
    This is the fourier smoothing in order to get expected flux for deviation
    calculations.
    Keep only the lowest n_harmonics seasonal harmonics (plus the mean).
    n_harmonics=1 keeps annual cycle only,
    2 keeps annual + semiannual, etc.
    """
    y = np.asarray(y, dtype=float)
    N = y.size

    # FFT
    Y = np.fft.rfft(y)

    # Zero out high frequencies (keep k=0..n_harmonics)
    Y_filtered = np.zeros_like(Y)
    Y_filtered[:n_harmonics + 1] = Y[:n_harmonics + 1]

    # Inverse FFT to get smoothed signal
    y_smooth = np.fft.irfft(Y_filtered, n=N)
    return y_smooth

def rolling_weighted_mean(series, window):
    """
    Centered triangular-weight rolling mean.
    """
    # Create symmetric triangular weights
    half = window // 2
    weights = np.arange(1, half + 2)
    if window % 2 == 0:
        weights = np.concatenate([weights, weights[::-1]])
    else:
        weights = np.concatenate([weights, weights[-2::-1]])

    weights = weights / weights.sum()

    return series.rolling(window, center=True, min_periods=1) \
                 .apply(lambda x: np.dot(x, weights[:len(x)]), raw=True)


def add_heatwave_indicator(df,heatwaves):
    """
    This function takes the daily flux dataframe and dataframe of all heatwaves
    (with start and end dates) and returns the flux dataframe with an indicator
    column of whether or not it is in a heatwave that day.
    """
    # Create a copy of the df
    new_df = df
    
    # Reduce columns
    heatwaves = heatwaves[["Site","start_dates","end_dates","top_heatwave"]]
    
    # Create column holding date range of heatwaves
    heatwaves["date"] = heatwaves.apply(
        lambda row: pd.date_range(row["start_dates"], row["end_dates"]),
        axis=1
        )
    
    # Explode the heatwave column
    heatwaves_expanded = heatwaves.explode("date").reset_index(drop=True)
    
    # Create indicator column for the heatwaves
    heatwaves_expanded["heatwave_indicator"] = [1] * len(heatwaves_expanded)
    
    # Left merge with the df
    new_df = pd.merge(new_df,heatwaves_expanded,on=['Site','date'],how="left")

    # Replacing missing heatwave indicators with 0
    new_df.loc[new_df.heatwave_indicator.isna(),"heatwave_indicator"] = 0
    
    return new_df