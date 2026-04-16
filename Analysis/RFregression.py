"""
This script runs initial random forest regression to assess the importance of different variables in 
predicting flux deviance.
"""
import os
os.chdir("/Users/marleeyork/Documents/project2")
import auxiliary
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
import seaborn as sns
import random
import pandas as pd

random.seed(10)

# Load in reduced heatwave dataset for flux data
flux_heatwaves_df = pd.read_csv("/Users/marleeyork/Documents/project2/data/heatwaves/flux_heatwaves_df.csv")

# I am going to start by mo
# For the "during" regressions, we want to use prior and during inputs into the model
# Including prior and durign avg SWC, sum of precipitation, difference in temperature with respect to
# DOY temperature, duration, prior and during VPD, soil properties, and biome data
X = flux_heatwaves_df[["top_heatwave","IGBP","Site","Season",   # heatwave/site specific
                       "P_during_avg","P_during_std",           # precipitation during heatwave
                       "SWC_during_avg", "SWC_during_std",      # swc during the heatwaves
                       "VPD_during_avg","VPD_during_std",       # VPD during the heatwave
                       "TA_during_avg","TA_during_std",         # Temperature during the heatwave
                       "SW_during_avg","SW_during_std",         # SW during the heatwave
                      "SWC_before_avg_30","SWC_before_std_30",  # prior SWC
                      "SWC_before_avg_10","SWC_before_std_10",  
                      "P_before_avg_30","P_before_std_30",      # prior precipitation
                      "P_before_avg_10","P_before_std_10",      
                      "VPD_before_avg_30","VPD_before_std_30",  # prior VPD
                      "VPD_before_avg_10","VPD_before_std_10",  
                      "SW_before_avg_30","SW_before_avg_10",    # prior SW
                      "SW_before_std_30","SW_before_std_10",    
                      "TA_mean_dev","duration",                 # Heatwave intensity metrics
                      "clay","sand","silt",                     # Soil information
                      "GPP_dev_during_avg","RECO_dev_during_avg","NEE_dev_during_avg",    # y variables
                      "NEE_dev_avg_diff","NEE_during_avg",              # Predicting difference in NEE instead?
                      "MAT","MAP",                                      # Long term climate information
                      "LOCATION_LAT","LOCATION_LONG","LOCATION_ELEV",   # Site location information
                      "start_DOY","start_Year",
                      "expected_NEE_during_avg"                         # Expected NEE for this day of the year
                      ]]  

# Cleaning up of missing variables and datatypes
X = X[~X.isna()]
X.dtypes

# List out the categorical variables
cat_cols = ["top_heatwave","IGBP","Site","Season"]
for c in cat_cols:
    X[c] = X[c].astype("category")
    
# Convert flux deviations to float
convert_var = ["TA_mean_dev","GPP_dev_during_avg","RECO_dev_during_avg",
               "NEE_dev_during_avg","NEE_dev_avg_diff","NEE_during_avg",
               "expected_NEE_during_avg"]
X[convert_var] = X[convert_var].astype("float")
X.dtypes

# Isolate the y values as the mean deviation of fluxes during the heatwave
y_GPP_during = X["GPP_dev_during_avg"]
y_RECO_during = X["RECO_dev_during_avg"]
y_NEE_dev = X["NEE_dev_during_avg"]
y_NEE_diff = X["NEE_dev_avg_diff"]
y_NEE = X["NEE_during_avg"]

# Update the X matrix without the target variables
X = X.drop(columns=["GPP_dev_during_avg","RECO_dev_during_avg","NEE_dev_during_avg","NEE_dev_avg_diff","NEE_during_avg"])

# Prepping data for NEE model (removing anything with missing NEE deviation during)
# This model has no training/testing/validation split
X_NEE = X[~y_NEE.isna()]
y_NEE = y_NEE[~y_NEE.isna()]

X_GPP = X[~y_GPP_during.isna()]
y_GPP_during = y_GPP_during[~y_GPP_during.isna()]

X_RECO = X[~y_RECO_during.isna()]
y_RECO_during = y_RECO_during[~y_RECO_during.isna()]

# This is using Site in the model
small_sites = ['CA-NS3','US-xNG','US-BRG']
X_NEE_cv = X_NEE[~X_NEE.Site.isin(small_sites)]
y_NEE_dev_cv = y_NEE_during[~X_NEE.Site.isin(small_sites)]
y_NEE_cv = y_NEE[~X_NEE.Site.isin(small_sites)]
X_NEE_cv["Site"] = X_NEE_cv["Site"].cat.remove_unused_categories()

# Modeling raw NEE values: this is actually getting better
NEE8a = fit_RFregression_cv(
    X=X_NEE_cv.drop(columns=["expected_NEE_during_avg"]),
    y=y_NEE_cv,
    site_col="Site",
    cat_cols=["Site","top_heatwave","IGBP","Season"],
    include_site_as_predictor=True,
    n_splits=5,
    val_size=0.2,
    n_repeats=20,
    plot_importance=True,
    random_state=10
)

# Plotting observed vs predicted
plt.subplots()
plt.scatter(NEE8a["cv_observed"],NEE8a["cv_predicted"],alpha=.4)
plt.plot([-10,10],[-10,10],linestyle="dashed",color="red")
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.show()

### THESE ARE OLD MODELS ######################################################
# This is a random forest without training and testing split
NEE1 = fit_RFregression(
    X=X_NEE,
    y=y_NEE_during,
    cat_cols=["top_heatwave","IGBP","Site"],
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    stopping_rounds=50,
    n_repeats=20,
    plot_pred=True,
    plot_importance=True)

plot_importance(NEE1)

# This model has a training/testing/validation split, with equal sampling from
# each site
NEE2 = fit_RFregression_split(
    X=X_NEE,
    y=y_NEE_during,
    site_col="Site",
    cat_cols=["top_heatwave","Site","IGBP"],
    test_size=0.2,
    val_size=0.2,
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    stopping_rounds=50,
    n_repeats=20,
    plot_pred=True,
    plot_importance=True)

plot_importance(NEE2)

# This model has training/testing/validation split with site sampling AND CROSS VALIDATION
# For cross validation, we have to have at least 5 heatwaves at a site

# This is using Site in the model
small_sites = ['CA-NS3','US-xNG']
X_NEE_cv = X_NEE[~X_NEE.Site.isin(small_sites)]
y_NEE_during_cv = y_NEE_during[~X_NEE.Site.isin(small_sites)]
X_NEE_cv["Site"] = X_NEE_cv["Site"].cat.remove_unused_categories()

X_GPP_cv = X_GPP[~X_GPP.Site.isin(small_sites)]
y_GPP_during_cv = y_GPP_during[~X_GPP.Site.isin(small_sites)]
X_GPP_cv["Site"] = X_GPP_cv["Site"].cat.remove_unused_categories()

X_RECO_cv = X_RECO[~X_RECO.Site.isin(small_sites)]
y_RECO_during_cv = y_RECO_during[~X_RECO.Site.isin(small_sites)]
X_RECO_cv["Site"] = X_RECO_cv["Site"].cat.remove_unused_categories()

# Clustered heatmap of multicollinearity amongst these variables
# From this we can see that the VPD and TA are highly correlated
numeric_X = X_NEE_cv.select_dtypes(include="number")
numeric_X = numeric_X.dropna(axis=1,how="all")
corr = numeric_X.corr(method="spearman")

sns.clustermap(
    corr,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    figsize=(14,14),
    linewidths=0.5
    )

# Cross validated random forest including site. For some reason, this isn't including site
NEE3a = fit_RFregression_cv(
    X=X_NEE_cv,
    y=y_NEE_during_cv,
    site_col="Site",
    cat_cols=["Site","top_heatwave","IGBP","Season"],
    include_site_as_predictor=True,
    n_splits=5,
    val_size=0.2,
    n_repeats=20,
    plot_importance=True,
    random_state=10
)

NEE3b = fit_RFregression_cv(
    X=X_NEE_cv,
    y=y_NEE_during_cv,
    site_col="Site",
    cat_cols=["Site","top_heatwave","IGBP","Season"],
    n_splits=5,
    val_size=0.2,
    n_repeats=20,
    plot_importance=True,
    random_state=11
)
NEE3c = fit_RFregression_cv(
    X=X_NEE_cv,
    y=y_NEE_during_cv,
    site_col="Site",
    cat_cols=["Site","top_heatwave","IGBP","Season"],
    n_splits=5,
    val_size=0.2,
    n_repeats=20,
    plot_importance=True,
    random_state=12
)

GPP3a = fit_RFregression_cv(
    X=X_GPP_cv.drop(columns=["NEE_dev_before_avg_5","NEE_dev_before_std_5"]),
    y=y_GPP_during_cv,
    site_col="Site",
    cat_cols=["Site","top_heatwave","IGBP","Season"],
    include_site_as_predictor=True,
    n_splits=5,
    val_size=0.2,
    n_repeats=20,
    plot_importance=True,
    random_state=10
)

RECO3a = fit_RFregression_cv(
    X=X_RECO_cv.drop(columns=["NEE_dev_before_avg_5","NEE_dev_before_std_5"]),
    y=y_RECO_during_cv,
    site_col="Site",
    cat_cols=["Site","top_heatwave","IGBP","Season"],
    include_site_as_predictor=True,
    n_splits=5,
    val_size=0.2,
    n_repeats=20,
    plot_importance=True,
    random_state=10
)

# Plotting observed vs predicted
plt.subplots()
plt.scatter(NEE3a["cv_observed"],NEE3a["cv_predicted"])
plt.plot([-2,2],[-2,2],linestyle="dashed",color="red")
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.show()

# Removing Site as a predictor; continuing to sample by it
NEE4a = fit_RFregression_cv(
    X=X_NEE_cv,
    y=y_NEE_during_cv,
    site_col="Site",
    cat_cols=["top_heatwave","IGBP","Season"],
    n_splits=5,
    val_size=0.2,
    n_repeats=20,
    plot_importance=True,
    random_state=10
)

NEE4b = fit_RFregression_cv(
    X=X_NEE_cv,
    y=y_NEE_during_cv,
    site_col="Site",
    cat_cols=["top_heatwave","IGBP","Season"],
    n_splits=5,
    val_size=0.2,
    n_repeats=20,
    plot_importance=True,
    random_state=11
)
NEE4c = fit_RFregression_cv(
    X=X_NEE_cv,
    y=y_NEE_during_cv,
    site_col="Site",
    cat_cols=["top_heatwave","IGBP","Season"],
    n_splits=5,
    val_size=0.2,
    n_repeats=20,
    plot_importance=True,
    random_state=12
)

# Removing soil but keeping site
X_NEE_nosoil = X_NEE_cv.drop(columns=["silt","clay","sand"])

NEE5a = fit_RFregression_cv(
    X=X_NEE_nosoil,
    y=y_NEE_during_cv,
    site_col="Site",
    cat_cols=["Site","top_heatwave","IGBP","Season"],
    include_site_as_predictor=True,
    n_splits=5,
    val_size=0.2,
    n_repeats=20,
    plot_importance=True,
    random_state=10
)

NEE5b = fit_RFregression_cv(
    X=X_NEE_nosoil,
    y=y_NEE_during_cv,
    site_col="Site",
    cat_cols=["Site","top_heatwave","IGBP","Season"],
    n_splits=5,
    val_size=0.2,
    n_repeats=20,
    plot_importance=True,
    random_state=11
)
NEE5c = fit_RFregression_cv(
    X=X_NEE_nosoil,
    y=y_NEE_during_cv,
    site_col="Site",
    cat_cols=["Site","top_heatwave","IGBP","Season"],
    n_splits=5,
    val_size=0.2,
    n_repeats=20,
    plot_importance=True,
    random_state=12
)

# Lets model the difference between prior and during average NEE instead
# This includes NEE before the heatwave event
X_NEE_diff = X[~y_NEE_diff.isna()] 
y_NEE_diff = y_NEE_diff[~y_NEE_diff.isna()]

# Now we reduce for cross validation
small_sites = ['CA-NS3','US-xNG','US-Cst']
X_NEE_diff_cv = X_NEE_diff[~X_NEE_diff.Site.isin(small_sites)]
y_NEE_diff_cv = y_NEE_diff[~X_NEE_diff.Site.isin(small_sites)]
X_NEE_diff_cv["Site"] = X_NEE_diff_cv["Site"].cat.remove_unused_categories()


# Lets try to model now
NEE6a = fit_RFregression_cv(
    X=X_NEE_diff_cv,
    y=y_NEE_diff_cv,
    site_col="Site",
    cat_cols=["Site","top_heatwave","IGBP","Season"],
    include_site_as_predictor=True,
    n_splits=5,
    val_size=0.2,
    n_repeats=20,
    plot_importance=True,
    random_state=10
)

# Plotting observed vs predicted
plt.subplots()
plt.scatter(NEE6a["cv_observed"],NEE6a["cv_predicted"])
plt.plot([-2,2],[-2,2],linestyle="dashed",color="red")
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.show()


# Trying to predict NEE during heatwave with the prior NEE also
NEE7a = fit_RFregression_cv(
    X=X_NEE_cv,
    y=y_NEE_during_cv,
    site_col="Site",
    cat_cols=["Site","top_heatwave","IGBP","Season"],
    include_site_as_predictor=True,
    n_splits=5,
    val_size=0.2,
    n_repeats=20,
    plot_importance=True,
    random_state=10
)

# Plotting observed vs predicted
plt.subplots()
plt.scatter(NEE7a["cv_observed"],NEE7a["cv_predicted"])
plt.plot([-2,2],[-2,2],linestyle="dashed",color="red")
plt.xlabel("Observed")
plt.ylabel("Predicted")
plt.show()

