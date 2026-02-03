"""
This script includes initial regressions run on differences in carbon fluxes before
and after heatwaves by different site and heatwave factors.
"""

# Loading in data and packages
import os
os.chdir("/Users/marleeyork/Documents/project2/Analysis")
import summaries_functions
import summaries_functions
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Lets load in the data. For the BADM, I skip over sites that don't have the climate or MAT
# None of the skipped sites are part of the 54 sites I'm interested in.
all_heatwaves_df = pd.read_csv("/Users/marleeyork/Documents/project2/data/heatwaves/all_heatwaves_df.csv")
df = pd.read_csv("/Users/marleeyork/Documents/project2/data/cleaned/AMF_DD.csv")
badm = loadBADM(path="/Users/marleeyork/Documents/project2/data/BADM",skip=["/Users/marleeyork/Documents/project2/data/BADM/AMF_CA-Qc2_BIF_20250731.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-Cop_BIF_20240229.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-UiD_BIF_20251017.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-BMM_BIF_20221003.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-NGC_BIF_20231208.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-Snf_BIF_20250731.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-AR2_BIF_20231031.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-SdH_BIF_20241204.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-CAK_BIF_20250731.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-AR1_BIF_20231031.xlsx"
                                                                            ],
                column='VARIABLE',value='DATAVALUE',measure=['IGBP','CLIMATE_KOEPPEN','MAT'],file_type='xslx')

# Reduce badm to those sites we have triad heatwaves for 
badm = badm[badm.Site.isin(all_heatwaves_df.Site.unique())].reset_index()
all_heatwaves_df = pd.merge(all_heatwaves_df,badm[['Site','CLIMATE_KOEPPEN','MAT']],on='Site',how='left')

# Reduce df to the sites we are interested in 
df = df[df.Site.isin(all_heatwaves_df.Site.unique())]\
    
# Load in soil water content data: we have swc for all of these sites
swc = loadAMF("/Users/marleeyork/Documents/project2/data/AMFdataDD",measures=['TIMESTAMP','SWC_F_MDS_1'])
swc.columns = ['date','SWC_F_MDS_1','Site']
df = pd.merge(df,swc,on=['Site','date'],how='left')

# Cleaning up SWC values
df.loc[df.SWC_F_MDS_1==-9999,'SWC_F_MDS_1'] = np.nan
df.loc[df.SWC_F_MDS_1<0,'SWC_F_MDS_1'] = 0
df = df.loc[df.Site!='US-xHA',:]

# Visualizing swc so we can determine what we need to cleanup
# Ok this looks pretty good!
for site in df.Site.unique():
    print(f"Plotting site {site}")
    site_df = df[df.Site==site]
    site_hw = all_heatwaves_df[all_heatwaves_df.Site==site].reset_index()
    plt.subplots()
    plt.scatter(site_df.date,site_df.SWC_F_MDS_1,s=.5,alpha=.5)
    for i in range(0,len(site_hw)):
        hw = site_hw.iloc[i,:]
        hw_dates = pd.date_range(hw.start_dates,hw.end_dates)
        plt.scatter(site_df[site_df.date.isin(hw_dates)].date,site_df[site_df.date.isin(hw_dates)].SWC_F_MDS_1, c="r",s=.5)
    plt.axhline(0,linestyle="--",c="black",linewidth=1)
    plt.title(f"Site: {site}")
    plt.show()
    
    input("Press [enter] to continue...")

# Correcting datetime variables
all_heatwaves_df["start_dates"] = pd.to_datetime(all_heatwaves_df.start_dates)
all_heatwaves_df["end_dates"] = pd.to_datetime(all_heatwaves_df.end_dates)
df["date"] = pd.to_datetime(df.date)

# Change flux names
df = df.iloc[:,1:]
df.columns = ['date','TA_F','SW_IN_F','VPD_F','P_F','NEE','RECO','GPP','Site','IGBP']

# Also reading in the min/max/mean heatwaves
heatwave_df = pd.read_csv("/Users/marleeyork/Documents/project2/data/heatwaves/heatwaves_df.csv")

# Relabel so that the min/max/mean heatwaves can be used in the calc_flux_avg()
heatwave_df['start_dates'] = pd.to_datetime(heatwave_df.start_dates)
heatwave_df['end_dates'] = pd.to_datetime(heatwave_df.end_dates)
heatwave_df = heatwave_df.iloc[:,1:]
heatwave_df.columns = ['Site','top_heatwave','start_dates','end_dates','duration','QAQC_flag']

## I want to do a little cleaning on this.
# If the data is missing, replace with NA
df.loc[df.GPP==-9999,'GPP'] = pd.NA
df.loc[df.RECO==-9999,'RECO'] = pd.NA
df.loc[df.NEE==-9999,'NEE'] = pd.NA

# Now replace negative GPP with 0
df.loc[df.GPP<0, 'GPP'] = 0

# Turn classes into categorical variables for use in regression
all_heatwaves_df['IGBP'] = all_heatwaves_df['IGBP'].astype("category")
all_heatwaves_df['CLIMATE_KOEPPEN'] = all_heatwaves_df['CLIMATE_KOEPPEN'].astype('category')

# Calculating before, during, and after chunks of fluxes and explanatory variables
###############################################################################

# First we calculate the actual heatwave chunks
# Lets calculate the average flux for before, during, and after periods of every heatwave
all_heatwaves_df = calc_flux_avg_multi_lag(flux_name = "GPP", 
                                           heatwaves_df = all_heatwaves_df, 
                                           flux_df = df, 
                                           before_lags = [10], 
                                           after_lags = [5,10,15,20,25,30])

all_heatwaves_df = calc_flux_avg_multi_lag(flux_name = "RECO",
                                           heatwaves_df = all_heatwaves_df,
                                           flux_df = df,
                                           before_lags = [10],
                                           after_lags = [5,10,15,20,25,30])

all_heatwaves_df = calc_flux_avg_multi_lag(flux_name = "NEE",
                                           heatwaves_df = all_heatwaves_df,
                                           flux_df = df,
                                           before_lags = [10],
                                           after_lags = [5,10,15,20,25,30])

# Calculating precipitation averages to include in the regression
all_heatwaves_df = calc_flux_avg_multi_lag(flux_name = "P_F",
                                           heatwaves_df = all_heatwaves_df,
                                           flux_df = df,
                                           before_lags = [30,10],
                                           after_lags = [5,10,15,20,25,30])

# Calculating temperature averages to include as predictors in the regression
all_heatwaves_df = calc_flux_avg_multi_lag(flux_name = "TA_F",
                                           heatwaves_df = all_heatwaves_df,
                                           flux_df = df,
                                           before_lags = [30,10],
                                           after_lags = [5,10,15,20,25,30]
                                           )

# Calculating SWC averages to include as predictors in the regression
all_heatwaves_df = calc_flux_avg_multi_lag(flux_name = "SWC_F_MDS_1",
                                           heatwaves_df = all_heatwaves_df,
                                           flux_df = df,
                                           before_lags = [30,10],
                                           after_lags = [5,10,15,20,25,30]
                                           )


# Save this dataframe to csv
os.chdir("/Users/marleeyork/Documents/project2/data/heatwaves/")
all_heatwaves_df.to_csv("all_heatwaves_df.csv")

# Clean up any nan: this removes 36 heatwaves from our dataframe
all_heatwaves_df_clean = all_heatwaves_df.loc[~all_heatwaves_df.isna().any(axis=1),:]
all_heatwaves_df = all_heatwaves_df_clean

# GPP
# Starting with during - before analysis, since this doesn't have different lag sizes
# Calculate the difference between fluxes before and during the event
# BD = before - during, avg = mean metric, std = standard deviation metric, # = length of lag
all_heatwaves_df["GPP_BD_avg"] = all_heatwaves_df['GPP_during_avg'] - all_heatwaves_df['GPP_before_avg_10']
all_heatwaves_df["GPP_BA_avg_5"] = all_heatwaves_df["GPP_after_avg_5"] - all_heatwaves_df['GPP_before_avg_10']
all_heatwaves_df["GPP_BA_avg_10"] = all_heatwaves_df["GPP_after_avg_10"] - all_heatwaves_df['GPP_before_avg_10']
all_heatwaves_df["GPP_BA_avg_15"] = all_heatwaves_df["GPP_after_avg_15"] - all_heatwaves_df['GPP_before_avg_10']
all_heatwaves_df["GPP_BA_avg_20"] = all_heatwaves_df["GPP_after_avg_20"] - all_heatwaves_df['GPP_before_avg_10']
all_heatwaves_df["GPP_BA_avg_25"] = all_heatwaves_df["GPP_after_avg_25"] - all_heatwaves_df['GPP_before_avg_10']
all_heatwaves_df["GPP_BA_avg_30"] = all_heatwaves_df["GPP_after_avg_30"] - all_heatwaves_df['GPP_before_avg_10']


all_heatwaves_df["RECO_BD_avg"] = all_heatwaves_df['RECO_during_avg'] - all_heatwaves_df['RECO_before_avg_10']
all_heatwaves_df["RECO_BA_avg_5"] = all_heatwaves_df["RECO_after_avg_5"] - all_heatwaves_df['RECO_before_avg_10']
all_heatwaves_df["RECO_BA_avg_10"] = all_heatwaves_df["RECO_after_avg_10"] - all_heatwaves_df['RECO_before_avg_10']
all_heatwaves_df["RECO_BA_avg_15"] = all_heatwaves_df["RECO_after_avg_15"] - all_heatwaves_df['RECO_before_avg_10']
all_heatwaves_df["RECO_BA_avg_20"] = all_heatwaves_df["RECO_after_avg_20"] - all_heatwaves_df['RECO_before_avg_10']
all_heatwaves_df["RECO_BA_avg_25"] = all_heatwaves_df["RECO_after_avg_25"] - all_heatwaves_df['RECO_before_avg_10']
all_heatwaves_df["RECO_BA_avg_30"] = all_heatwaves_df["RECO_after_avg_30"] - all_heatwaves_df['RECO_before_avg_10']

all_heatwaves_df["NEE_BD_avg"] = all_heatwaves_df['NEE_during_avg'] - all_heatwaves_df['NEE_before_avg_10']
all_heatwaves_df["NEE_BA_avg_5"] = all_heatwaves_df["NEE_after_avg_5"] - all_heatwaves_df['NEE_before_avg_10']
all_heatwaves_df["NEE_BA_avg_10"] = all_heatwaves_df["NEE_after_avg_10"] - all_heatwaves_df['NEE_before_avg_10']
all_heatwaves_df["NEE_BA_avg_15"] = all_heatwaves_df["NEE_after_avg_15"] - all_heatwaves_df['NEE_before_avg_10']
all_heatwaves_df["NEE_BA_avg_20"] = all_heatwaves_df["NEE_after_avg_20"] - all_heatwaves_df['NEE_before_avg_10']
all_heatwaves_df["NEE_BA_avg_25"] = all_heatwaves_df["NEE_after_avg_25"] - all_heatwaves_df['NEE_before_avg_10']
all_heatwaves_df["NEE_BA_avg_30"] = all_heatwaves_df["NEE_after_avg_30"] - all_heatwaves_df['NEE_before_avg_10']

# Doing the same for standard deviations
all_heatwaves_df["GPP_BD_std"] = all_heatwaves_df['GPP_during_std'] - all_heatwaves_df['GPP_before_std_10']
all_heatwaves_df["GPP_BA_std_5"] = all_heatwaves_df["GPP_after_std_5"] - all_heatwaves_df['GPP_before_std_10']
all_heatwaves_df["GPP_BA_std_10"] = all_heatwaves_df["GPP_after_std_10"] - all_heatwaves_df['GPP_before_std_10']
all_heatwaves_df["GPP_BA_std_15"] = all_heatwaves_df["GPP_after_std_15"] - all_heatwaves_df['GPP_before_std_10']
all_heatwaves_df["GPP_BA_std_20"] = all_heatwaves_df["GPP_after_std_20"] - all_heatwaves_df['GPP_before_std_10']
all_heatwaves_df["GPP_BA_std_25"] = all_heatwaves_df["GPP_after_std_25"] - all_heatwaves_df['GPP_before_std_10']
all_heatwaves_df["GPP_BA_std_30"] = all_heatwaves_df["GPP_after_std_30"] - all_heatwaves_df['GPP_before_std_10']


all_heatwaves_df["RECO_BD_std"] = all_heatwaves_df['RECO_during_std'] - all_heatwaves_df['RECO_before_std_10']
all_heatwaves_df["RECO_BA_std_5"] = all_heatwaves_df["RECO_after_std_5"] - all_heatwaves_df['RECO_before_std_10']
all_heatwaves_df["RECO_BA_std_10"] = all_heatwaves_df["RECO_after_std_10"] - all_heatwaves_df['RECO_before_std_10']
all_heatwaves_df["RECO_BA_std_15"] = all_heatwaves_df["RECO_after_std_15"] - all_heatwaves_df['RECO_before_std_10']
all_heatwaves_df["RECO_BA_std_20"] = all_heatwaves_df["RECO_after_std_20"] - all_heatwaves_df['RECO_before_std_10']
all_heatwaves_df["RECO_BA_std_25"] = all_heatwaves_df["RECO_after_std_25"] - all_heatwaves_df['RECO_before_std_10']
all_heatwaves_df["RECO_BA_std_30"] = all_heatwaves_df["RECO_after_std_30"] - all_heatwaves_df['RECO_before_std_10']

all_heatwaves_df["NEE_BD_std"] = all_heatwaves_df['NEE_during_std'] - all_heatwaves_df['NEE_before_std_10']
all_heatwaves_df["NEE_BA_std_5"] = all_heatwaves_df["NEE_after_std_5"] - all_heatwaves_df['NEE_before_std_10']
all_heatwaves_df["NEE_BA_std_10"] = all_heatwaves_df["NEE_after_std_10"] - all_heatwaves_df['NEE_before_std_10']
all_heatwaves_df["NEE_BA_std_15"] = all_heatwaves_df["NEE_after_std_15"] - all_heatwaves_df['NEE_before_std_10']
all_heatwaves_df["NEE_BA_std_20"] = all_heatwaves_df["NEE_after_std_20"] - all_heatwaves_df['NEE_before_std_10']
all_heatwaves_df["NEE_BA_std_25"] = all_heatwaves_df["NEE_after_std_25"] - all_heatwaves_df['NEE_before_std_10']
all_heatwaves_df["NEE_BA_std_30"] = all_heatwaves_df["NEE_after_std_30"] - all_heatwaves_df['NEE_before_std_10']

# Now I want to add in a variable that looks at deviance of each during period from site MAT 
# My hope is that this indicates seasonality!
all_heatwaves_df['MAT'] = all_heatwaves_df['MAT'].astype('float')
all_heatwaves_df["dev_from_MAT"] = all_heatwaves_df['MAT'] - all_heatwaves_df['TA_F_during_avg']

# Lets start by looking at distributions of some of our variables we are interested in
sns.kdeplot(all_heatwaves_df[["GPP_BD_avg","GPP_BA_avg_5","GPP_BA_avg_10","GPP_BA_avg_15","GPP_BA_avg_20",
                              "GPP_BA_avg_25","GPP_BA_avg_30"]])

sns.kdeplot(all_heatwaves_df[["RECO_BD_avg","RECO_BA_avg_5","RECO_BA_avg_10","RECO_BA_avg_15","RECO_BA_avg_20",
                              "RECO_BA_avg_25","RECO_BA_avg_30"]])

sns.kdeplot(all_heatwaves_df[["NEE_BD_avg","NEE_BA_avg_5","NEE_BA_avg_10","NEE_BA_avg_15","NEE_BA_avg_20",
                              "NEE_BA_avg_25","NEE_BA_avg_30"]])

sns.kdeplot(all_heatwaves_df[["GPP_BD_std","GPP_BA_std_5","GPP_BA_std_10","GPP_BA_std_15","GPP_BA_std_20",
                              "GPP_BA_std_25","GPP_BA_std_30"]])

sns.kdeplot(all_heatwaves_df[["RECO_BD_std","RECO_BA_std_5","RECO_BA_std_10","RECO_BA_std_15","RECO_BA_std_20",
                              "RECO_BA_std_25","RECO_BA_std_30"]])

sns.kdeplot(all_heatwaves_df[["NEE_BD_std","NEE_BA_std_5","NEE_BA_std_10","NEE_BA_std_15","NEE_BA_std_20",
                              "NEE_BA_std_25","NEE_BA_std_30"]])


# Duration has this poisson distribution
sns.kdeplot(all_heatwaves_df.duration)
plt.show()

fig, ax = plt.subplots(1,1,figsize=(12,12))
plt.scatter(all_heatwaves_df.duration,all_heatwaves_df.GPP_BD_avg,alpha=.3)
plt.show()

###############################################################################
##               Multipe Linear Regressions of Flux Differences              ##
###############################################################################
# Regression for Before-During GPP average difference #########################
GPP_avg_lin_mod = smf.ols(
    formula="GPP_BD_avg ~ duration*Mean_perc + duration*Max_perc + P_F_before_avg_30 + P_F_during_avg*duration + TA_F_before_avg_10 + TA_F_during_avg + dev_from_MAT + SWC_F_MDS_1_before_avg_10 + SWC_F_MDS_1_during_std + SWC_F_MDS_1_during_avg + C(IGBP)",
    data=all_heatwaves_df
).fit()

# Print model summaries
print(GPP_avg_lin_mod.summary())

# Extract coefficients
coef_table = GPP_avg_lin_mod.summary2().tables[1]
print(coef_table)

# Partial effects plot, holding everything else constant
fig, ax = plt.subplots(figsize=(6,6))

ax.scatter(
    all_heatwaves_df["Max_perc"],
    all_heatwaves_df["GPP_BD_avg"],
    alpha=0.3,
    label="Observed"
)

# Sort for clean line
df_sorted = all_heatwaves_df.sort_values("Max_perc")

ax.plot(
    df_sorted["Max_perc"],
    model.predict(df_sorted),
    linewidth=2,
    label="Model prediction"
)

ax.set_xlabel("Max_perc")
ax.set_ylabel("ΔGPP (Before–During)")
ax.legend()
plt.show()

# Q-Q Plot
sm.qqplot(model.resid, line="45")
plt.show()

# Regression for Before-During GPP std difference #############################
model = smf.ols(
    formula="GPP_BD_std ~ duration*Mean_perc + duration*Max_perc + P_F_before_avg_30 + P_F_during_avg*duration + TA_F_before_avg_10 + TA_F_during_avg + SWC_F_MDS_1_before_avg_10 + SWC_F_MDS_1_during_std + SWC_F_MDS_1_during_avg+ dev_from_MAT + C(IGBP)",
    data=all_heatwaves_df
).fit()

# Print model summaries
print(model.summary())

# Extract coefficients
coef_table = model.summary2().tables[1]
print(coef_table)

# Q-Q Plot
sm.qqplot(model.resid, line="45")
plt.show()

# Regression for Before-During Reco avg difference ############################
model = smf.ols(
    formula="RECO_BD_avg ~ duration*Mean_perc + duration*Max_perc + P_F_before_avg_30 + P_F_during_avg*duration + TA_F_before_avg_10 + TA_F_during_avg + SWC_F_MDS_1_before_avg_10 + SWC_F_MDS_1_during_std + SWC_F_MDS_1_during_avg+ dev_from_MAT + C(IGBP)",
    data=all_heatwaves_df
).fit()

# Print model summaries
print(model.summary())

# Extract coefficients
coef_table = model.summary2().tables[1]
print(coef_table)

# Q-Q Plot
sm.qqplot(model.resid, line="45")
plt.show()

# Regression for Before-During Reco std difference ############################
model = smf.ols(
    formula="RECO_BD_std ~ duration*Mean_perc + duration*Max_perc + P_F_before_avg_30 + P_F_during_avg*duration + TA_F_before_avg_10 + TA_F_during_avg + SWC_F_MDS_1_before_avg_10 + SWC_F_MDS_1_during_std + SWC_F_MDS_1_during_avg+ dev_from_MAT + C(IGBP)",
    data=all_heatwaves_df
).fit()

# Print model summaries
print(model.summary())

# Extract coefficients
coef_table = model.summary2().tables[1]
print(coef_table)

# Q-Q Plot
sm.qqplot(model.resid, line="45")
plt.show()

###############################################################################
##                   Quantile Regression of Flux Differences                 ##
###############################################################################
# I am going to start with the averages since those have the most intense tails
## Quantile regression for GPP be
mod = smf.quantreg("GPP_BD_avg ~ duration*Mean_perc + duration*Max_perc + P_F_before_avg_30 + P_F_during_avg*duration + TA_F_before_avg_10 + TA_F_during_avg + SWC_F_MDS_1_before_avg_10 + SWC_F_MDS_1_during_std + SWC_F_MDS_1_during_avg+ dev_from_MAT + C(IGBP)", 
                   data = all_heatwaves_df) 
quantiles = [0.1, 0.5, 0.9] 
fits = {q: mod.fit(q=q) for q in quantiles}

# Print summary of each model
print(fits[0.1].summary())
print(fits[0.5].summary())
print(fits[0.9].summary())

# Quantile regression of above
mod = smf.quantreg(
    """
    GPP_BD_avg ~ 
        duration + I(duration**2) +
        Mean_perc + I(Mean_perc**2) +
        Max_perc + I(Max_perc**2) +
        duration:Mean_perc + duration:Max_perc +
        P_F_before_avg_30 +
        P_F_during_avg + duration:P_F_during_avg +
        TA_F_before_avg_10 +
        TA_F_during_avg +
        dev_from_MAT +
        C(IGBP)
    """,
    data=all_heatwaves_df
)

quantiles = [0.1, 0.5, 0.9] 
fits = {q: mod.fit(q=q) for q in quantiles}

# Print summary of each model
print(fits[0.1].summary())
print(fits[0.5].summary())
print(fits[0.9].summary())


# Quantile regression for GPP std
mod = smf.quantreg("GPP_BD_std ~ duration*Mean_perc + duration*Max_perc + P_F_before_avg_30 + P_F_during_avg*duration + TA_F_before_avg_10 + TA_F_during_avg + SWC_F_MDS_1_before_avg_10 + SWC_F_MDS_1_during_std + SWC_F_MDS_1_during_avg+ dev_from_MAT + C(IGBP)", 
                   data = all_heatwaves_df) 
quantiles = [0.1, 0.5, 0.9] 
fits = {q: mod.fit(q=q) for q in quantiles}

# Print summary of each model
print(fits[0.1].summary())
print(fits[0.5].summary())
print(fits[0.9].summary())

# Now for RECO average before-during
mod = smf.quantreg("RECO_BD_avg ~ duration*Mean_perc + duration*Max_perc + P_F_before_avg_30 + P_F_during_avg*duration + TA_F_before_avg_10 + TA_F_during_avg + SWC_F_MDS_1_before_avg_10 + SWC_F_MDS_1_during_std + SWC_F_MDS_1_during_avg+ dev_from_MAT + C(IGBP)", 
                   data = all_heatwaves_df) 
quantiles = [0.1, 0.5, 0.9] 
fits = {q: mod.fit(q=q) for q in quantiles}

# Print summary of each model
print(fits[0.1].summary())
print(fits[0.5].summary())
print(fits[0.9].summary())

# Now for RECO std before-during
mod = smf.quantreg("RECO_BD_avg ~ duration*Mean_perc + duration*Max_perc + P_F_before_avg_30 + P_F_during_avg*duration + TA_F_before_avg_10 + TA_F_during_avg + SWC_F_MDS_1_before_avg_10 + SWC_F_MDS_1_during_std + SWC_F_MDS_1_during_avg+ dev_from_MAT + C(IGBP)", 
                   data = all_heatwaves_df) 
quantiles = [0.1, 0.5, 0.9] 
fits = {q: mod.fit(q=q) for q in quantiles}

# Print summary of each model
print(fits[0.1].summary())
print(fits[0.5].summary())
print(fits[0.9].summary())

###############################################################################
##               Mixed Linear Effect Models for Flux Differences             ##
###############################################################################

# Lets plot each average on a timeseries colored by site since site will be our 
plt.subplots()
for site in all_heatwaves_df.Site.unique():
    site_hw = all_heatwaves_df[all_heatwaves_df.Site==site]
    plt.plot(site_hw.start_dates,
                site_hw.RECO_BD_avg,
                label=site)
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Lets do the same for standard deviation
plt.subplots()
for site in all_heatwaves_df.Site.unique():
    site_hw = all_heatwaves_df[all_heatwaves_df.Site==site]
    plt.plot(site_hw.start_dates,
                site_hw.RECO_BA_std_30,
                label=site)
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Ok lets try to fit a mixed linear effects model for average difference
GPP_avg_ME_mod = smf.mixedlm(
    """
    GPP_BD_avg ~
        duration*Mean_perc +
        duration*Max_perc +
        P_F_before_avg_30 +
        P_F_during_avg*duration +
        TA_F_before_avg_10 +
        TA_F_during_avg +
        dev_from_MAT +
        C(IGBP)
    """,
    data=all_heatwaves_df,
    groups=all_heatwaves_df["Site"]  
)

GPP_avg_ME_fit = GPP_avg_ME_mod.fit(method="lbfgs")
print(GPP_avg_ME_fit.summary())

# Comparing metric between mixed effects and linear
yhat_ols = GPP_avg_lin_mod.predict(all_heatwaves_df)
yhat_mixed = GPP_avg_ME_fit.predict(all_heatwaves_df)

y = all_heatwaves_df["GPP_BD_avg"]
mse_ols = mean_squared_error(y, yhat_ols)
mse_mixed = mean_squared_error(y, yhat_mixed)

rmse_ols = mean_squared_error(y, yhat_ols, squared=False)
rmse_mixed = mean_squared_error(y, yhat_mixed, squared=False)

mse_ols, mse_mixed, rmse_ols, rmse_mixed

# Lets visualize this plot: This is really bad
plt.scatter(y, yhat_mixed, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--")
plt.xlabel("Observed GPP_BD_avg")
plt.ylabel("Predicted GPP_BD_avg")
plt.title("Mixed-effects model: observed vs predicted")
plt.tight_layout()
plt.show()

# Lets try centering by site
df_hw = all_heatwaves_df.copy()
df_hw["yhat"] = fit.predict(df_hw)

df_hw["GPP_centered"] = df_hw["GPP_BD_avg"] - df_hw.groupby("Site")["GPP_BD_avg"].transform("mean")
df_hw["yhat_centered"] = df_hw["yhat"] - df_hw.groupby("Site")["yhat"].transform("mean")

plt.scatter(df_hw["GPP_centered"], df_hw["yhat_centered"], alpha=0.5)
plt.axline((0, 0), slope=1, linestyle="--")
plt.xlabel("Observed (site-centered)")
plt.ylabel("Predicted (site-centered)")
plt.title("Within-site fit")
plt.tight_layout()
plt.show()

np.corrcoef(df_hw['GPP_centered'],df_hw['yhat_centered'])
###############################################################################
##               Exploratory Plotting to Survey Variable Impact              ##
###############################################################################

# More exploratory plotting to determine the effects of different variables
# Linear regression is just a rough way to go for this
# turn IGBP_y into categorical codes

# Plotting flux changes across MAT deviation
# Making IGBP color map
igbp = all_heatwaves_df["IGBP"].astype("category")
cats = igbp.cat.categories

cmap = plt.get_cmap("tab10")

color_map = {cat: cmap(i) for i, cat in enumerate(cats)}

# convert to per rows colors
colors = igbp.astype(str).map(color_map)

# Plotting
fig, ax = plt.subplots(2,2)
ax = ax.flatten()
ax[0].scatter(all_heatwaves_df.dev_from_MAT, all_heatwaves_df.GPP_BD_avg,s=.5,c=colors)
ax[0].set_title("GPP Avg")
ax[1].scatter(all_heatwaves_df.dev_from_MAT, all_heatwaves_df.GPP_BD_std,s=.5,c=colors)
ax[1].set_title("GPP std")
ax[2].scatter(all_heatwaves_df.dev_from_MAT, all_heatwaves_df.RECO_BD_avg,s=.5,c=colors)
ax[2].set_title("RECO Avg")
ax[3].scatter(all_heatwaves_df.dev_from_MAT, all_heatwaves_df.RECO_BD_std,s=.5,c=colors)
ax[3].set_title("RECO std")
plt.tight_layout()
plt.show()

# Plotting regressions performed separately on duration by each IGBP
# Duration impact on GPP_BD_avg by IGBP
plt.figure()

for cat, group in all_heatwaves_df.groupby('IGBP'):
    plt.scatter(group['duration'], group['GPP_BD_avg'], label=cat, alpha=.5)
    reg = LinearRegression().fit(group[['duration']],group['GPP_BD_avg'])
    y_pred = reg.predict(group[['duration']])
    plt.plot(group.duration,y_pred)

plt.title("Regressions of during-before avg GPP by duration on each IGBP")
plt.xlabel('Duration')
plt.ylabel('GPP_BD_avg')
plt.legend(title='IGBP', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Duration impact on RECO_BD_avg by IGBP
plt.figure()

for cat, group in all_heatwaves_df.groupby('IGBP'):
    plt.scatter(group['duration'], group['RECO_BD_avg'], label=cat, alpha=.5)
    reg = LinearRegression().fit(group[['duration']],group['RECO_BD_avg'])
    y_pred = reg.predict(group[['duration']])
    plt.plot(group.duration,y_pred)

plt.title("Regressions of during-before avg RECO by duration on each IGBP")
plt.xlabel('Duration')
plt.ylabel('RECO_BD_avg')
plt.legend(title='IGBP', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Duration impact on GPP_BS_std by IGBP
plt.figure()

for cat, group in all_heatwaves_df.groupby('IGBP'):
    plt.scatter(group['duration'], group['GPP_BA_std_30'], label=cat, alpha=.5)
    reg = LinearRegression().fit(group[['duration']],group['GPP_BA_std_30'])
    y_pred = reg.predict(group[['duration']])
    plt.plot(group.duration,y_pred)

plt.title("Regressions of during-before std_30 GPP by duration on each IGBP")
plt.xlabel('Duration')
plt.ylabel('GPP_BA_std_30')
plt.legend(title='IGBP', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Duration impact on RECO_BS_std by IGBP
plt.figure()

for cat, group in all_heatwaves_df.groupby('IGBP'):
    plt.scatter(group['duration'], group['RECO_BA_std_30'], label=cat, alpha=.5)
    reg = LinearRegression().fit(group[['duration']],group['RECO_BA_std_30'])
    y_pred = reg.predict(group[['duration']])
    plt.plot(group.duration,y_pred)

plt.title("Regressions of during-before std_30 RECO by duration on each IGBP")
plt.xlabel('Duration')
plt.ylabel('RECO_BA_std_30')
plt.legend(title='IGBP', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Do I have variables for all these heatwave periods?
# I think Im interested in temperature, precipitation, and swc for each of these
for site in all_heatwaves_df.Site.unique():
    site_hw = all_heatwaves_df[all_heatwaves_df.Site==site]
    site_hw["date"] = site_hw.apply(
        lambda r: pd.date_range(r.start_dates, r.end_dates, freq="D"),
        axis=1
        )
    hw_periods = site_hw.explode("date", ignore_index=True)

    site_df = df[df.Site==site]
    fig, ax = plt.subplots(2,1)
    ax = ax.flatten()
    
    ax[0].scatter(site_df.date,site_df.TA_F,s=.5,alpha=.5)
    ax[0].scatter(hw_periods.date,site_df[site_df.date.isin(hw_periods.date)].TA_F,c='red',s=.5)
    ax[0].set_title(f"TA_F for Site: {site}")
    
    ax[1].scatter(site_df.date,site_df.P_F,s=.5,alpha=.5)
    ax[1].scatter(hw_periods.date,site_df[site_df.date.isin(hw_periods.date)].P_F,c='red',s=.5)
    ax[1].set_title(f"P_F for Site: {site}")
    
    plt.tight_layout()
    plt.show()
    
    input("Press [enter] to continue...")

# Looking at moisture distribution of heatwaves
plt.subplots()
plt.scatter(all_heatwaves_df.P_F_during_avg,all_heatwaves_df.SWC_F_MDS_1_during_avg,alpha=.1)
plt.xlabel("Avg Precipitation During Heatwave")
plt.ylabel("Average SWC During Heatwave")
plt.title("Average Moisture Conditions During Heatwave")
plt.show()

plt.subplots()
plt.scatter(all_heatwaves_df.P_F_during_std,all_heatwaves_df.SWC_F_MDS_1_during_std,alpha=.1)
plt.xlabel("Precipitation Std During Heatwave")
plt.ylabel("SWC Std During Heatwave")
plt.title("Moisture Variability During Heatwave")
plt.show()
