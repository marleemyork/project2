'''
This script runs initial analysis of heatwave impacts on carbon cycling.
Some things I'm interested in: comparison of GPP before, during, and after
heatwaves.
'''

import os
os.chdir("/Users/marleeyork/Documents/project2")
import auxiliary
os.chdir("/Users/marleeyork/Documents/project2/Analysis")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import visualization_functions
import seaborn as sns


###############################################################################
#   Working with deviance from expected flux
###############################################################################

# Plotting the difference between non-heatwave DOY average and our smoothed
# expected values
for site in df.Site.unique():
    
    site_df = df[df.Site==site]
    fig, ax = plt.subplots(3,1, figsize=(6,12))
    ax = ax.flatten()
    ax[0].plot(site_df.DOY,site_df.DOY_GPP,label="Daily Mean",c="blue")
    ax[0].plot(site_df.DOY,site_df.expected_GPP,label="Smoothed Expected Value",c="red")
    ax[0].legend()
    ax[0].set_ylabel("GPP")
    
    ax[1].plot(site_df.DOY,site_df.DOY_RECO,label="Daily Mean",c="blue")
    ax[1].plot(site_df.DOY,site_df.expected_RECO,label="Smoothed Expected Value",c="red")
    ax[1].set_xlabel("DOY")
    ax[1].set_ylabel("RECO")
    
    ax[2].plot(site_df.DOY,site_df.DOY_NEE,label="Daily Mean",c="blue")
    ax[2].plot(site_df.DOY,site_df.expected_NEE,label="Smoothed Expected Value",c="red")
    ax[2].set_xlabel("DOY")
    ax[2].set_ylabel("NEE")
    
    
    plt.title(f"{site}")
    plt.show()
    
    input("Press [enter] for next site..")
    
# Plotting deviation over time
plt.subplots()
plt.scatter(all_heatwaves_df[all_heatwaves_df.GPP_cum_dev < 250].start_dates,all_heatwaves_df[all_heatwaves_df.GPP_cum_dev < 250].GPP_cum_dev,s=.5)
plt.show()

plt.subplots()
plt.scatter(all_heatwaves_df[all_heatwaves_df.GPP_mean_dev < 250].start_dates,all_heatwaves_df[all_heatwaves_df.GPP_mean_dev < 250].GPP_mean_dev,s=.5)
plt.show()

plt.subplots()
plt.scatter(all_heatwaves_df.start_dates,all_heatwaves_df.RECO_cum_dev,s=.5)
plt.show()

plt.subplots()
plt.scatter(all_heatwaves_df.start_dates,all_heatwaves_df.RECO_mean_dev,s=.5)
plt.show()

# Plot density plots: maybe I could do a z-test?
sns.kdeplot(all_heatwaves_df.GPP_mean_dev)
sns.kdeplot(all_heatwaves_df.RECO_mean_dev)
sns.kdeplot(all_heatwaves_df.NEE_mean_dev)


# Looking at deivance across all heatwaves
sns.kdeplot(all_heatwaves_df.GPP_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df.GPP_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df.GPP_mean_dev - all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].GL|PP_mean_dev_10, label="Difference", linestyle="--", color="red")
plt.title("Mean GPP deviance: Prior to and during heatwave event")
plt.legend()
# plt.xlim(-5,15)
plt.show()

sns.kdeplot(all_heatwaves_df.RECO_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df.RECO_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df.RECO_mean_dev - all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].RECO_mean_dev_10, label="Difference", linestyle="--", color="red")
plt.title("Mean RECO deviance: Prior to and during heatwave event")
plt.xlim(-5,6)
plt.legend()
plt.show()

sns.kdeplot(all_heatwaves_df.NEE_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df.NEE_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df.NEE_mean_dev - all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].NEE_mean_dev_10, label="Difference", linestyle="--", color="red")
plt.title("Mean NEE deviance: Prior to and during heatwave event")
plt.xlim(-5,6)
plt.legend()
plt.show()

# Looking at differences in deviation distributions for triad heatwaves specifically
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].NEE_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].NEE_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].NEE_mean_dev - all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].NEE_mean_dev_10, label="Difference", linestyle="--", color="red")
plt.title("Triad Mean NEE deviance: Prior to and during heatwave event")
plt.legend()
plt.show()

sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].RECO_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].RECO_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].RECO_mean_dev - all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].RECO_mean_dev_10, label="Difference", linestyle="--", color="red")
plt.title("Triad Mean RECO deviance: Prior to and during heatwave event")
plt.legend()
plt.show()

sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].GPP_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].GPP_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].GPP_mean_dev - all_heatwaves_df[all_heatwaves_df.top_heatwave=="Triad"].RECO_mean_dev_10, label="Difference", linestyle="--", color="red")
plt.title("Triad Mean RECO deviance: Prior to and during heatwave event")
plt.legend()
plt.show()

# Looking at differences for night-intensified
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Night-intensified"].NEE_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Night-intensified"].NEE_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Night-intensified"].NEE_mean_dev_diff, label="Difference", linestyle="--", color="red")
plt.title("Night-intensified Mean NEE deviance: Prior to and during heatwave event")
plt.legend()
plt.show()

sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Night-intensified"].RECO_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Night-intensified"].RECO_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Night-intensified"].RECO_mean_dev_diff, label="Difference", linestyle="--", color="red")
plt.title("Night-intensified Mean RECO deviance: Prior to and during heatwave event")
plt.legend()
plt.show()

sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Night-intensified"].GPP_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Night-intensified"].GPP_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Night-intensified"].GPP_mean_dev_diff, label="Difference", linestyle="--", color="red")
plt.title("Night-Intensified Mean GPP deviance: Prior to and during heatwave event")
plt.legend()
plt.show()

# Looking at differences for overall heatwaves
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Overall"].NEE_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Overall"].NEE_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Overall"].NEE_mean_dev_diff, label="Difference", linestyle="--", color="red")
plt.title("Overall Mean NEE deviance: Prior to and during heatwave event")
plt.legend()
plt.show()

sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Overall"].RECO_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Overall"].RECO_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Overall"].RECO_mean_dev_diff, label="Difference", linestyle="--", color="red")
plt.title("Overall Mean RECO deviance: Prior to and during heatwave event")
plt.legend()
plt.show()

sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Overall"].GPP_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Overall"].GPP_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Overall"].GPP_mean_dev_diff, label="Difference", linestyle="--", color="red")
plt.title("Overall Mean GPP deviance: Prior to and during heatwave event")
plt.legend()
plt.show()

# Day time heatwaves only now
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Day"].NEE_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Day"].NEE_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Day"].NEE_mean_dev_diff, label="Difference", linestyle="--", color="red")
plt.title("Day Mean NEE deviance: Prior to and during heatwave event")
plt.legend()
plt.show()

sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Day"].RECO_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Day"].RECO_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Day"].RECO_mean_dev_diff, label="Difference", linestyle="--", color="red")
plt.title("Day Mean RECO deviance: Prior to and during heatwave event")
plt.legend()
plt.show()

sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Day"].GPP_mean_dev, label="During")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Day"].GPP_mean_dev_10, label="Before")
sns.kdeplot(all_heatwaves_df[all_heatwaves_df.top_heatwave=="Day"].GPP_mean_dev_diff, label="Difference", linestyle="--", color="red")
plt.title("Overall Mean GPP deviance: Prior to and during heatwave event")
plt.legend()
plt.show()


# Making some initial tables of this
all_heatwaves_df.groupby("top_heatwave")[['GPP_mean_dev_10','GPP_mean_dev','GPP_mean_dev_diff']].mean().reset_index().rename(columns={'GPP_mean_dev_10':'Prior','GPP_mean_dev':'During','GPP_mean_dev_diff':'Difference'})
all_heatwaves_df.groupby("top_heatwave")[["RECO_mean_dev_10","RECO_mean_dev","RECO_mean_dev_diff"]].mean().reset_index().rename(columns={'RECO_mean_dev_10':'Prior','RECO_mean_dev':'During','RECO_mean_dev_diff':'Difference'})
all_heatwaves_df.groupby("top_heatwave")[["NEE_mean_dev_10","NEE_mean_dev","NEE_mean_dev_diff"]].mean().reset_index().rename(columns={'NEE_mean_dev_10':'Prior','NEE_mean_dev':'During','NEE_mean_dev_diff':'Difference'})

# Performing t-tests
GPP_diff = np.array(all_heatwaves_df.GPP_mean_dev_diff)
t_stat, p_val = stats.ttest_1samp(GPP_diff,0)

###############################################################################
#   Visualizing a single heatwave
###############################################################################
# Lets look at heatwaves that are longer than 10 days
long_heatwaves = all_heatwaves_df[all_heatwaves_df.duration>10].index

for i in long_heatwaves:
    # Selecting a sample heatwave
    sample_hw = all_heatwaves_df.iloc[i]

    # Create a range of dates from 10 days prior to 30 days after
    before_hw = sample_hw.start_dates - timedelta(days=10)
    after_hw = sample_hw.end_dates + timedelta(days=30)
    sample_dates = pd.date_range(start=before_hw,end=after_hw)

    # Create a range of dates over only the heatwave
    hw_dates = pd.date_range(start=sample_hw.start_dates,end=sample_hw.end_dates)

    # Selecting data for that heatwave
    sample_df = df[(df.Site==sample_hw.Site) & (df.date.isin(sample_dates))]
    sample_df["this_hw"] = df["date"].apply(lambda v: 1 if (v in hw_dates) else 0)

    # Now lets visualize this
    plt.subplots()
    plt.plot(sample_df.date,sample_df.NEE_VUT_REF)
    plt.plot(sample_df.date,-sample_df.GPP_NT_VUT_REF,c="green")
    plt.plot(sample_df.date,sample_df.RECO_NT_VUT_REF,c="red")
    plt.axvline(x=sample_hw.start_dates, linestyle='--',c="black")
    plt.axvline(x=sample_hw.end_dates,linestyle='--',c="black")
    plt.xticks(rotation=45)
    plt.title(sample_hw.Site)
    plt.show()
    
    input("Press [enter] to continue...")
    
###############################################################################
#   Lets calculate averages before and after heatwaves
###############################################################################
# Remember that this is across 76 sites
# Lets calculate the average flux for before, during, and after periods of every heatwave
all_heatwaves_df = calc_flux_avg(flux_name = "GPP", 
                                 heatwaves_df = all_heatwaves_df, 
                                 flux_df = df, 
                                 before_lag = 10, 
                                 after_lag = 20)

all_heatwaves_df = calc_flux_avg(flux_name = "RECO",
                                 heatwaves_df = all_heatwaves_df,
                                 flux_df = df,
                                 before_lag = 10,
                                 after_lag = 20)

all_heatwaves_df = calc_flux_avg(flux_name = "NEE",
                                 heatwaves_df = all_heatwaves_df,
                                 flux_df = df,
                                 before_lag = 10,
                                 after_lag = 20)

# Averages before, during, and after heatwaves
all_heatwaves_df.GPP_before_avg.mean()
all_heatwaves_df.GPP_during_avg.mean()
all_heatwaves_df.GPP_after_avg.mean()
all_heatwaves_df.RECO_before_avg.mean()
all_heatwaves_df.RECO_during_avg.mean()
all_heatwaves_df.RECO_after_avg.mean()
all_heatwaves_df.NEE_before_avg.mean()
all_heatwaves_df.NEE_during_avg.mean()
all_heatwaves_df.NEE_after_avg.mean()

# Paired t-test of before and after mean fluxes before, during, and after heatwaves
t_BA_GPP, p_BA_GPP = stats.ttest_rel(all_heatwaves_df.GPP_before_avg, 
                                          all_heatwaves_df.GPP_after_avg, 
                                          nan_policy="omit")
t_BD_GPP, p_BD_GPP = stats.ttest_rel(all_heatwaves_df.GPP_before_avg, 
                                          all_heatwaves_df.GPP_during_avg, 
                                          nan_policy="omit")

t_BA_RECO, p_BA_RECO = stats.ttest_rel(all_heatwaves_df.RECO_before_avg, 
                                          all_heatwaves_df.RECO_after_avg, 
                                          nan_policy="omit")
t_BD_RECO, p_BD_RECO = stats.ttest_rel(all_heatwaves_df.RECO_before_avg, 
                                          all_heatwaves_df.RECO_during_avg, 
                                          nan_policy="omit")

t_BA_NEE, p_BA_NEE = stats.ttest_rel(all_heatwaves_df.NEE_before_avg, 
                                          all_heatwaves_df.NEE_after_avg, 
                                          nan_policy="omit")
t_BD_NEE, p_BD_NEE = stats.ttest_rel(all_heatwaves_df.NEE_before_avg, 
                                          all_heatwaves_df.NEE_during_avg, 
                                          nan_policy="omit")

# Now grouping by heatwave type
GPP_means = all_heatwaves_df.groupby('top_heatwave')[['GPP_before_avg','GPP_during_avg','GPP_after_avg']].mean().reset_index()
RECO_means = all_heatwaves_df.groupby('top_heatwave')[['RECO_before_avg','RECO_during_avg','RECO_after_avg']].mean().reset_index()
NEE_means = all_heatwaves_df.groupby('top_heatwave')[['NEE_before_avg','NEE_during_avg','NEE_after_avg']].mean().reset_index()

# Trying this with multiple tests for each flux
tests_GPP_avg = [
    ("GPP_before_avg_10", "GPP_during_avg",  "BD_GPP"),
    ("GPP_before_avg_10", "GPP_after_avg_5", "BA_5_GPP"),
    ("GPP_before_avg_10", "GPP_after_avg_10",  "BA_10_GPP"),
    ("GPP_before_avg_10", "GPP_after_avg_15",  "BA_15_GPP"),
    ("GPP_before_avg_10", "GPP_after_avg_20",  "BA_20_GPP"),
    ("GPP_before_avg_10", "GPP_after_avg_25",  "BA_25_GPP"),
    ("GPP_before_avg_10", "GPP_after_avg_30",  "BA_30_GPP")
]

tests_GPP_std = [
    ("GPP_before_std_10", "GPP_during_std",  "BD_GPP"),
    ("GPP_before_std_10", "GPP_after_std_5", "BA_5_GPP"),
    ("GPP_before_std_10", "GPP_after_std_10",  "BA_10_GPP"),
    ("GPP_before_std_10", "GPP_after_std_15",  "BA_15_GPP"),
    ("GPP_before_std_10", "GPP_after_std_20",  "BA_20_GPP"),
    ("GPP_before_std_10", "GPP_after_std_25",  "BA_25_GPP"),
    ("GPP_before_std_10", "GPP_after_std_30",  "BA_30_GPP") 
]

tests_RECO_avg = [
    ("RECO_before_avg_10", "RECO_during_avg",  "BD_RECO"),
    ("RECO_before_avg_10", "RECO_after_avg_5", "BA_5_RECO"),
    ("RECO_before_avg_10", "RECO_after_avg_10",  "BA_10_RECO"),
    ("RECO_before_avg_10", "RECO_after_avg_15",  "BA_15_RECO"),
    ("RECO_before_avg_10", "RECO_after_avg_20",  "BA_20_RECO"),
    ("RECO_before_avg_10", "RECO_after_avg_25",  "BA_25_RECO"),
    ("RECO_before_avg_10", "RECO_after_avg_30",  "BA_30_RECO")
]

tests_RECO_std = [
    ("RECO_before_std_10", "RECO_during_std",  "BD_RECO"),
    ("RECO_before_std_10", "RECO_after_std_5", "BA_5_RECO"),
    ("RECO_before_std_10", "RECO_after_std_10",  "BA_10_RECO"),
    ("RECO_before_std_10", "RECO_after_std_15",  "BA_15_RECO"),
    ("RECO_before_std_10", "RECO_after_std_20",  "BA_20_RECO"),
    ("RECO_before_std_10", "RECO_after_std_25",  "BA_25_RECO"),
    ("RECO_before_std_10", "RECO_after_std_30",  "BA_30_RECO") 
]

tests_NEE_avg = [
    ("NEE_before_avg_10", "NEE_during_avg",  "BD_NEE"),
    ("NEE_before_avg_10", "NEE_after_avg_5", "BA_5_NEE"),
    ("NEE_before_avg_10", "NEE_after_avg_10",  "BA_10_NEE"),
    ("NEE_before_avg_10", "NEE_after_avg_15",  "BA_15_NEE"),
    ("NEE_before_avg_10", "NEE_after_avg_20",  "BA_20_NEE"),
    ("NEE_before_avg_10", "NEE_after_avg_25",  "BA_25_NEE"),
    ("NEE_before_avg_10", "NEE_after_avg_30",  "BA_30_NEE")
]

tests_NEE_std = [
    ("NEE_before_std_10", "NEE_during_std",  "BD_NEE"),
    ("NEE_before_std_10", "NEE_after_std_5", "BA_5_NEE"),
    ("NEE_before_std_10", "NEE_after_std_10",  "BA_10_NEE"),
    ("NEE_before_std_10", "NEE_after_std_15",  "BA_15_NEE"),
    ("NEE_before_std_10", "NEE_after_std_20",  "BA_20_NEE"),
    ("NEE_before_std_10", "NEE_after_std_25",  "BA_25_NEE"),
    ("NEE_before_std_10", "NEE_after_std_30",  "BA_30_NEE") 
]

def run_tests(g, tests):
    out = {}
    for a, b, tag in tests:
        t, p = stats.ttest_rel(g[a], g[b], nan_policy="omit")
        out[f"t_{tag}"] = t
        out[f"p_{tag}"] = p
    return pd.Series(out)

GPP_avg_ttests = (
    all_heatwaves_df
      .groupby("top_heatwave")
      .apply(run_tests, tests_GPP_avg)
      .reset_index()
)
GPP_avg_ttests.to_clipboard(index=False)

GPP_std_ttests = (
    all_heatwaves_df
      .groupby("top_heatwave")
      .apply(run_tests, tests_GPP_std)
      .reset_index()
)
GPP_std_ttests.to_clipboard(index=False)

RECO_avg_ttests = (
    all_heatwaves_df
      .groupby("top_heatwave")
      .apply(run_tests, tests_RECO_avg)
      .reset_index()
)
RECO_avg_ttests.to_clipboard(index=False)

RECO_std_ttests = (
    all_heatwaves_df
      .groupby("top_heatwave")
      .apply(run_tests, tests_RECO_std)
      .reset_index()
)
RECO_std_ttests.to_clipboard(index=False)

NEE_avg_ttests = (
    all_heatwaves_df
      .groupby("top_heatwave")
      .apply(run_tests, tests_NEE_avg)
      .reset_index()
)
NEE_avg_ttests.to_clipboard(index=False)

NEE_std_ttests = (
    all_heatwaves_df
      .groupby("top_heatwave")
      .apply(run_tests, tests_NEE_std)
      .reset_index()
)
NEE_std_ttests.to_clipboard(index=False)

# Now calculating the paired t-tests for these

ttest_BA_GPP = (
    all_heatwaves_df
      .groupby('top_heatwave')
      .apply(lambda g: pd.Series(
          stats.ttest_rel(g['GPP_before_avg'], g['GPP_after_avg'], nan_policy='omit'),
          index=['t_BA_GPP', 'p_BA_GPP']
      ))
      .reset_index()
)

ttest_BD_GPP =  (
    all_heatwaves_df
      .groupby('top_heatwave')
      .apply(lambda g: pd.Series(
          stats.ttest_rel(g['GPP_before_avg'], g['GPP_during_avg'], nan_policy='omit'),
          index=['t_BD_GPP', 'p_BD_GPP']
      ))
      .reset_index()
)

ttest_BA_RECO = (
    all_heatwaves_df
      .groupby('top_heatwave')
      .apply(lambda g: pd.Series(
          stats.ttest_rel(g['RECO_before_avg'], g['RECO_after_avg'], nan_policy='omit'),
          index=['t_BA_RECO', 'p_BA_RECO']
      ))
      .reset_index()
)

ttest_BD_RECO =  (
    all_heatwaves_df
      .groupby('top_heatwave')
      .apply(lambda g: pd.Series(
          stats.ttest_rel(g['RECO_before_avg'], g['RECO_during_avg'], nan_policy='omit'),
          index=['t_BD_RECO', 'p_BD_RECO']
      ))
      .reset_index()
)

ttest_BA_NEE = (
    all_heatwaves_df
      .groupby('top_heatwave')
      .apply(lambda g: pd.Series(
          stats.ttest_rel(g['NEE_before_avg'], g['NEE_after_avg'], nan_policy='omit'),
          index=['t_BA_NEE', 'p_BA_NEE']
      ))
      .reset_index()
)

ttest_BD_NEE =  (
    all_heatwaves_df
      .groupby('top_heatwave')
      .apply(lambda g: pd.Series(
          stats.ttest_rel(g['NEE_before_avg'], g['NEE_during_avg'], nan_policy='omit'),
          index=['t_BD_NEE', 'p_BD_NEE']
      ))
      .reset_index()
)


# Merge these datasets
ttest_GPP = pd.merge(ttest_BA_GPP,ttest_BD_GPP,on='top_heatwave',how='inner')
GPP_means = pd.merge(GPP_means,ttest_GPP,on='top_heatwave',how='inner')
GPP_means.to_clipboard(index=False)

ttest_RECO = pd.merge(ttest_BA_RECO,ttest_BD_RECO,on='top_heatwave',how='inner')
RECO_means = pd.merge(RECO_means,ttest_RECO,on='top_heatwave',how='inner')
RECO_means.to_clipboard(index=False)

ttest_NEE = pd.merge(ttest_BA_NEE,ttest_BD_NEE,on='top_heatwave',how='inner')
NEE_means = pd.merge(NEE_means,ttest_NEE,on='top_heatwave',how='inner')
NEE_means.to_clipboard(index=False)

###############################################################################
#   Lets calculate averages of variance metrics before and after heatwaves
###############################################################################

# Averages of std before, during, and after heatwaves
all_heatwaves_df.GPP_before_std.mean()
all_heatwaves_df.GPP_during_std.mean()
all_heatwaves_df.GPP_after_std.mean()
all_heatwaves_df.RECO_before_std.mean()
all_heatwaves_df.RECO_during_std.mean()
all_heatwaves_df.RECO_after_std.mean()
all_heatwaves_df.NEE_before_std.mean()
all_heatwaves_df.NEE_during_std.mean()
all_heatwaves_df.NEE_after_std.mean()

# Paired t-test of before and after mean fluxes before, during, and after heatwaves
t_BA_GPP, p_BA_GPP = stats.ttest_rel(all_heatwaves_df.GPP_before_std, 
                                          all_heatwaves_df.GPP_after_std, 
                                          nan_policy="omit")
t_BD_GPP, p_BD_GPP = stats.ttest_rel(all_heatwaves_df.GPP_before_std, 
                                          all_heatwaves_df.GPP_during_std, 
                                          nan_policy="omit")

t_BA_RECO_5, p_BA_RECO_5 = stats.ttest_rel(all_heatwaves_df.RECO_before_std_10, 
                                          all_heatwaves_df.RECO_after_std_5, 
                                          nan_policy="omit")
t_BD_RECO, p_BD_RECO = stats.ttest_rel(all_heatwaves_df.RECO_before_std, 
                                          all_heatwaves_df.RECO_during_std, 
                                          nan_policy="omit")

t_BA_NEE_5, p_BA_NEE_5 = stats.ttest_rel(all_heatwaves_df.NEE_before_std_5, 
                                          all_heatwaves_df.NEE_after_std_5, 
                                          nan_policy="omit")
t_BD_NEE_5, p_BD_NEE_5 = stats.ttest_rel(all_heatwaves_df.NEE_before_std_5, 
                                          all_heatwaves_df.NEE_during_std_5, 
                                          nan_policy="omit")

# Now grouping by heatwave type
GPP_std = all_heatwaves_df.groupby('top_heatwave')[['GPP_before_std','GPP_during_std','GPP_after_std']].mean().reset_index()
RECO_std = all_heatwaves_df.groupby('top_heatwave')[['RECO_before_std','RECO_during_std','RECO_after_avg']].mean().reset_index()
NEE_std = all_heatwaves_df.groupby('top_heatwave')[['NEE_before_std','NEE_during_std','NEE_after_avg']].mean().reset_index()

# Now calculating the paired t-tests for these

ttest_BA_GPP = (
    all_heatwaves_df
      .groupby('top_heatwave')
      .apply(lambda g: pd.Series(
          stats.ttest_rel(g['GPP_before_std'], g['GPP_after_std'], nan_policy='omit'),
          index=['t_BA_GPP', 'p_BA_GPP']
      ))
      .reset_index()
)

ttest_BD_GPP =  (
    all_heatwaves_df
      .groupby('top_heatwave')
      .apply(lambda g: pd.Series(
          stats.ttest_rel(g['GPP_before_std'], g['GPP_during_std'], nan_policy='omit'),
          index=['t_BD_GPP', 'p_BD_GPP']
      ))
      .reset_index()
)

ttest_BA_RECO = (
    all_heatwaves_df
      .groupby('top_heatwave')
      .apply(lambda g: pd.Series(
          stats.ttest_rel(g['RECO_before_std'], g['RECO_after_std'], nan_policy='omit'),
          index=['t_BA_RECO', 'p_BA_RECO']
      ))
      .reset_index()
)

ttest_BD_RECO =  (
    all_heatwaves_df
      .groupby('top_heatwave')
      .apply(lambda g: pd.Series(
          stats.ttest_rel(g['RECO_before_std'], g['RECO_during_std'], nan_policy='omit'),
          index=['t_BD_RECO', 'p_BD_RECO']
      ))
      .reset_index()
)

ttest_BA_NEE = (
    all_heatwaves_df
      .groupby('top_heatwave')
      .apply(lambda g: pd.Series(
          stats.ttest_rel(g['NEE_before_std'], g['NEE_after_std'], nan_policy='omit'),
          index=['t_BA_NEE', 'p_BA_NEE']
      ))
      .reset_index()
)

ttest_BD_NEE =  (
    all_heatwaves_df
      .groupby('top_heatwave')
      .apply(lambda g: pd.Series(
          stats.ttest_rel(g['NEE_before_std'], g['NEE_during_std'], nan_policy='omit'),
          index=['t_BD_NEE', 'p_BD_NEE']
      ))
      .reset_index()
)


# Merge these datasets
ttest_GPP = pd.merge(ttest_BA_GPP,ttest_BD_GPP,on='top_heatwave',how='inner')
GPP_std = pd.merge(GPP_std,ttest_GPP,on='top_heatwave',how='inner')
GPP_std.to_clipboard(index=False)

ttest_RECO = pd.merge(ttest_BA_RECO,ttest_BD_RECO,on='top_heatwave',how='inner')
RECO_std = pd.merge(RECO_std,ttest_RECO,on='top_heatwave',how='inner')
RECO_std.to_clipboard(index=False)

ttest_NEE = pd.merge(ttest_BA_NEE,ttest_BD_NEE,on='top_heatwave',how='inner')
NEE_std = pd.merge(NEE_std,ttest_NEE,on='top_heatwave',how='inner')
NEE_std.to_clipboard(index=False)



###############################################################################
#   Boxplots off flux periods across all types of heatwaves
###############################################################################

# Avg GPP difference, night intensified heatwaves
night_int = all_heatwaves_df[all_heatwaves_df.top_heatwave=="Night-intensified"]
multi_boxplots(night_int,value_cols=['GPP_before_avg_10','GPP_during_avg','GPP_after_avg_10',
                                     'GPP_after_avg_15','GPP_after_avg_20','GPP_after_avg_25',
                                     'GPP_after_avg_30'],category_col=[],title="GPP Avg for Night-Intensified")

# Avg GPP difference, before-during for day, overall, and triad heatwaves
GPP_diff_df = all_heatwaves_df[all_heatwaves_df.top_heatwave.isin(["Day","Overall","Triad"])]
multi_boxplots_grouped(df=GPP_diff_df,
                       value_cols=["GPP_before_avg_10","GPP_during_avg"],
                       category_col="top_heatwave",
                       title="Before-During GPP Avg",
                       figsize=(6,6)
                       )

# Avg Reco difference, before-during for all heatwave types
multi_boxplots_grouped(df=all_heatwaves_df,
                       value_cols=["RECO_before_avg_10","RECO_during_avg"],
                       category_col="top_heatwave",
                       title="Before-During RECO Avg",
                       figsize=(12,6)
                       )

# Average Reco difference, Night-intensified across timescales
night_int = all_heatwaves_df[all_heatwaves_df.top_heatwave=="Night-intensified"]
multi_boxplots(night_int,value_cols=['RECO_before_avg_10','RECO_during_avg','RECO_after_avg_10',
                                     'RECO_after_avg_15','RECO_after_avg_20','RECO_after_avg_25',
                                     'RECO_after_avg_30'],category_col=[],title="RECO Avg for Night-Intensified")

# Average Reco difference, Overall across timescales
overall_int = all_heatwaves_df[all_heatwaves_df.top_heatwave=="Overall"]
multi_boxplots(overall_int,value_cols=['RECO_before_avg_10','RECO_during_avg','RECO_after_avg_10',
                                     'RECO_after_avg_15','RECO_after_avg_20','RECO_after_avg_25',
                                     'RECO_after_avg_30'],category_col=[],title="RECO Avg for Overall")

# STD GPP difference, before-during for all heatwave types
multi_boxplots_grouped(df=all_heatwaves_df,
                       value_cols=["GPP_before_avg_10","GPP_during_avg"],
                       category_col="top_heatwave",
                       title="Before-During GPP std",
                       figsize=(12,6)
                       )

multi_boxplots_grouped(df=all_heatwaves_df,
                       value_cols=["RECO_before_avg_10","RECO_during_avg"],
                       category_col="top_heatwave",
                       title="Before-During RECO std",
                       figsize=(12,6)
                       )

multi_boxplots_grouped(df=all_heatwaves_df,
                       value_cols=["NEE_before_avg_10","NEE_during_avg"],
                       category_col="top_heatwave",
                       title="Before-During RECO std",
                       figsize=(12,6)
                       )

# STD RECO difference, Day across timescales
day = all_heatwaves_df[all_heatwaves_df.top_heatwave=="Day"]
day_long = day[day.duration >=10]
multi_boxplots(day,value_cols=['RECO_before_std_10','RECO_during_std','RECO_after_std_10',
                                   'RECO_after_std_15','RECO_after_std_20','RECO_after_std_25',
                                   'RECO_after_std_30'],category_col=[],title="RECO Std for Day Across Timescales",
               showfliers=False)

# STD RECO difference, Day-intensified timescales
day_int = all_heatwaves_df[all_heatwaves_df.top_heatwave=="Day-intensified"]
multi_boxplots(day_int,value_cols=['RECO_before_std_10','RECO_during_std','RECO_after_std_10',
                                   'RECO_after_std_15','RECO_after_std_20','RECO_after_std_25',
                                   'RECO_after_std_30'],category_col=[],title="RECO Std for Day Across Timescales",
               showfliers=False)

# STD RECO difference, Day-Night Spike timescales
day_night = all_heatwaves_df[all_heatwaves_df.top_heatwave=="Day-Night Spike"]
multi_boxplots(day_night,value_cols=['RECO_before_std_10','RECO_during_std','RECO_after_std_10',
                                   'RECO_after_std_15','RECO_after_std_20','RECO_after_std_25',
                                   'RECO_after_std_30'],category_col=[],title="RECO Std for Day Across Timescales",
               showfliers=False)

# STD GPP during heatwave by duration, across all heatwaves
multi_boxplots_grouped(df=all_heatwaves_df[all_heatwaves_df.duration < 18],
                       value_cols=["GPP_before_std_10","GPP_during_std","GPP_after_std_10","GPP_after_std_30"],
                       category_col="duration",
                       title="GPP STD Difference by Duration",
                       figsize=(12,6))


# STD RECO during heatwave by duration, across all heatwaves
multi_boxplots_grouped(df=all_heatwaves_df[all_heatwaves_df.duration<18],
                       value_cols=["RECO_before_std_10","RECO_during_std","RECO_after_std_10","RECO_after_std_30"],
                       category_col="duration",
                       title="GPP STD Difference by Duration",
                       figsize=(12,4),
                       showfliers=False)

multi_boxplots_grouped(df=all_heatwaves_df[all_heatwaves_df.duration<18],
                       value_cols=["RECO_before_std_10","RECO_during_std","RECO_after_std_10","RECO_after_std_30"],
                       category_col="duration",
                       title="GPP STD Difference by Duration",
                       figsize=(12,4),
                       showfliers=True)


all_heatwaves_df["during_perc_GPP_change"] = all_heatwaves_df["GPP_BD_avg"] / all_heatwaves_df["GPP_during_avg"]
all_heatwaves_df["during_perc_RECO_change"] = all_heatwaves_df["RECO_BD_avg"] / all_heatwaves_df["RECO_during_avg"]
all_heatwaves_df["during_perc_NEE_change"] = all_heatwaves_df["NEE_BD_avg"] / all_heatwaves_df["NEE_during_avg"]

all_heatwaves_df.groupby("top_heatwave")[["during_perc_GPP_change","during_perc_RECO_change","during_perc_NEE_change"]].mean()


# STARTING WITH RECO ##########################################################
categories = all_heatwaves_df["top_heatwave"].unique()

# Assign a color to each heatwave type
colors = plt.cm.Set2(np.arange(len(categories)))
type_to_color = dict(zip(categories, colors))

fig, ax = plt.subplots(figsize=(12, 6))

box_data = []
labels = []

# Build boxplot data
for hw_type in categories:
    type_df = all_heatwaves_df[all_heatwaves_df["top_heatwave"] == hw_type]

    for period in ["RECO_before_avg", "RECO_during_avg", "RECO_after_avg"]:
        values = type_df[type_df[period] > 0][period]
        box_data.append(values)
        labels.append(f"{hw_type}-{period.split('_')[1]}")

# Create boxplot
positions = np.arange(1, len(box_data) + 1)
bp = ax.boxplot(box_data, positions=positions, patch_artist=True)

ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45, ha="right")


# Color boxes by heatwave type (every 3 boxes belong to the same type)
for i, box in enumerate(bp["boxes"]):
    hw_type = categories[i // 3]
    box.set_facecolor(type_to_color[hw_type])

# Legend
legend_handles = [
    mpatches.Patch(color=type_to_color[t], label=t)
    for t in categories
]
ax.legend(handles=legend_handles, title="Heatwave Type")

# Labels and formatting
ax.set_ylabel("RECO")
ax.set_title("RECO before, during, and after heatwaves\nColored by heatwave type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# NOW FOR GPP #################################################################
fig, ax = plt.subplots(figsize=(12, 6))

box_data = []
labels = []

# Build boxplot data
for hw_type in categories:
    type_df = all_heatwaves_df[all_heatwaves_df["top_heatwave"] == hw_type]

    for period in ["GPP_before_avg", "GPP_during_avg", "GPP_after_avg"]:
        values = type_df[type_df[period] > 0][period]
        box_data.append(values)
        labels.append(f"{hw_type}-{period.split('_')[1]}")

# Create boxplot
positions = np.arange(1, len(box_data) + 1)
bp = ax.boxplot(box_data, positions=positions, patch_artist=True)

ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45, ha="right")


# Color boxes by heatwave type (every 3 boxes belong to the same type)
for i, box in enumerate(bp["boxes"]):
    hw_type = categories[i // 3]
    box.set_facecolor(type_to_color[hw_type])

# Legend
legend_handles = [
    mpatches.Patch(color=type_to_color[t], label=t)
    for t in categories
]
ax.legend(handles=legend_handles, title="Heatwave Type")

# Labels and formatting
ax.set_ylabel("GPP")
ax.set_title("GPP before, during, and after heatwaves\nColored by heatwave type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# NOW FOR NEE #################################################################
fig, ax = plt.subplots(figsize=(12, 6))

box_data = []
labels = []

# Build boxplot data
for hw_type in categories:
    type_df = all_heatwaves_df[all_heatwaves_df["top_heatwave"] == hw_type]

    for period in ["NEE_before_avg", "NEE_during_avg", "NEE_after_avg"]:
        values = type_df[type_df[period] > 0][period]
        box_data.append(values)
        labels.append(f"{hw_type}-{period.split('_')[1]}")

# Create boxplot
positions = np.arange(1, len(box_data) + 1)
bp = ax.boxplot(box_data, positions=positions, patch_artist=True)

ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45, ha="right")


# Color boxes by heatwave type (every 3 boxes belong to the same type)
for i, box in enumerate(bp["boxes"]):
    hw_type = categories[i // 3]
    box.set_facecolor(type_to_color[hw_type])

# Legend
legend_handles = [
    mpatches.Patch(color=type_to_color[t], label=t)
    for t in categories
]
ax.legend(handles=legend_handles, title="Heatwave Type")

# Labels and formatting
ax.set_ylabel("NEE")
ax.set_title("NEE before, during, and after heatwaves\nColored by heatwave type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

###############################################################################
#   Boxplots off std of heatwave fluxes across all types of heatwaves
###############################################################################

# STARTING WITH RECO ##########################################################
categories = all_heatwaves_df["top_heatwave"].unique()

# Assign a color to each heatwave type
colors = plt.cm.Set2(np.arange(len(categories)))
type_to_color = dict(zip(categories, colors))

fig, ax = plt.subplots(figsize=(12, 6))

box_data = []
labels = []

# Build boxplot data
for hw_type in categories:
    type_df = all_heatwaves_df[all_heatwaves_df["top_heatwave"] == hw_type]

    for period in ["RECO_before_std", "RECO_during_std", "RECO_after_std"]:
        values = type_df[type_df[period] > 0][period]
        box_data.append(values)
        labels.append(f"{hw_type}-{period.split('_')[1]}")

# Create boxplot
positions = np.arange(1, len(box_data) + 1)
bp = ax.boxplot(box_data, positions=positions, patch_artist=True,showfliers=False)

ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45, ha="right")


# Color boxes by heatwave type (every 3 boxes belong to the same type)
for i, box in enumerate(bp["boxes"]):
    hw_type = categories[i // 3]
    box.set_facecolor(type_to_color[hw_type])

# Legend
legend_handles = [
    mpatches.Patch(color=type_to_color[t], label=t)
    for t in categories
]
ax.legend(handles=legend_handles, title="Heatwave Type")

# Labels and formatting
ax.set_ylabel("RECO")
ax.set_title("RECO before, during, and after heatwaves\nColored by heatwave type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# NOW FOR GPP #################################################################
fig, ax = plt.subplots(figsize=(12, 6))

box_data = []
labels = []

# Build boxplot data
for hw_type in categories:
    type_df = all_heatwaves_df[all_heatwaves_df["top_heatwave"] == hw_type]

    for period in ["GPP_before_std", "GPP_during_std", "GPP_after_std"]:
        values = type_df[type_df[period] > 0][period]
        box_data.append(values)
        labels.append(f"{hw_type}-{period.split('_')[1]}")

# Create boxplot
positions = np.arange(1, len(box_data) + 1)
bp = ax.boxplot(box_data, positions=positions, patch_artist=True,showfliers=False)

ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45, ha="right")


# Color boxes by heatwave type (every 3 boxes belong to the same type)
for i, box in enumerate(bp["boxes"]):
    hw_type = categories[i // 3]
    box.set_facecolor(type_to_color[hw_type])

# Legend
legend_handles = [
    mpatches.Patch(color=type_to_color[t], label=t)
    for t in categories
]
ax.legend(handles=legend_handles, title="Heatwave Type")

# Labels and formatting
ax.set_ylabel("GPP")
ax.set_title("GPP before, during, and after heatwaves\nColored by heatwave type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# NOW FOR NEE #################################################################
fig, ax = plt.subplots(figsize=(12, 6))

box_data = []
labels = []

# Build boxplot data
for hw_type in categories:
    type_df = all_heatwaves_df[all_heatwaves_df["top_heatwave"] == hw_type]

    for period in ["NEE_before_std", "NEE_during_std", "NEE_after_std"]:
        values = type_df[type_df[period] > 0][period]
        box_data.append(values)
        labels.append(f"{hw_type}-{period.split('_')[1]}")

# Create boxplot
positions = np.arange(1, len(box_data) + 1)
bp = ax.boxplot(box_data, positions=positions, patch_artist=True,showfliers=False)

ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45, ha="right")


# Color boxes by heatwave type (every 3 boxes belong to the same type)
for i, box in enumerate(bp["boxes"]):
    hw_type = categories[i // 3]
    box.set_facecolor(type_to_color[hw_type])

# Legend
legend_handles = [
    mpatches.Patch(color=type_to_color[t], label=t)
    for t in categories
]
ax.legend(handles=legend_handles, title="Heatwave Type")

# Labels and formatting
ax.set_ylabel("NEE")
ax.set_title("NEE before, during, and after heatwaves\nColored by heatwave type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

###############################################################################
#  Looking at before and after averages for only min/max/mean heatwaves
###############################################################################
# I am going to calculate these averages for 10 days prior and 30 days after
heatwave_df = calc_flux_avg(flux_name = "GPP", 
                            heatwaves_df = heatwave_df, 
                            flux_df = df, 
                            before_lag = 10, 
                            after_lag = 20)

heatwave_df = calc_flux_avg(flux_name = "RECO",
                            heatwaves_df = heatwave_df,
                            flux_df = df,
                            before_lag = 10,
                            after_lag = 20)

heatwave_df = calc_flux_avg(flux_name = "NEE",
                            heatwaves_df = heatwave_df,
                            flux_df = df,
                            before_lag = 10,
                            after_lag = 20)

###############################################################################
#  Visualizing boxplots of the flux periods around min/max/mean heatwaves
###############################################################################

# STARTING WITH RECO ##########################################################
categories = heatwave_df["top_heatwave"].unique()

# Assign a color to each heatwave type
colors = plt.cm.Set2(np.arange(len(categories)))
type_to_color = dict(zip(categories, colors))

fig, ax = plt.subplots(figsize=(6, 6))

box_data = []
labels = []

# Build boxplot data
for hw_type in categories:
    type_df = heatwave_df[heatwave_df["top_heatwave"] == hw_type]

    for period in ["RECO_before", "RECO_during", "RECO_after"]:
        values = type_df[type_df[period] > 0][period]
        box_data.append(values)
        labels.append(f"{hw_type}-{period.split('_')[1]}")

# Create boxplot
positions = np.arange(1, len(box_data) + 1)
bp = ax.boxplot(box_data, positions=positions, patch_artist=True)

ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45, ha="right")


# Color boxes by heatwave type (every 3 boxes belong to the same type)
for i, box in enumerate(bp["boxes"]):
    hw_type = categories[i // 3]
    box.set_facecolor(type_to_color[hw_type])

# Legend
legend_handles = [
    mpatches.Patch(color=type_to_color[t], label=t)
    for t in categories
]
ax.legend(handles=legend_handles, title="Heatwave Type")

# Labels and formatting
ax.set_ylabel("RECO")
ax.set_title("RECO before, during, and after heatwaves\nColored by heatwave type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# NOW FOR GPP #################################################################
fig, ax = plt.subplots(figsize=(6, 6))

box_data = []
labels = []

# Build boxplot data
for hw_type in categories:
    type_df = heatwave_df[heatwave_df["top_heatwave"] == hw_type]

    for period in ["GPP_before", "GPP_during", "GPP_after"]:
        values = type_df[type_df[period] > 0][period]
        box_data.append(values)
        labels.append(f"{hw_type}-{period.split('_')[1]}")

# Create boxplot
positions = np.arange(1, len(box_data) + 1)
bp = ax.boxplot(box_data, positions=positions, patch_artist=True)

ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45, ha="right")


# Color boxes by heatwave type (every 3 boxes belong to the same type)
for i, box in enumerate(bp["boxes"]):
    hw_type = categories[i // 3]
    box.set_facecolor(type_to_color[hw_type])

# Legend
legend_handles = [
    mpatches.Patch(color=type_to_color[t], label=t)
    for t in categories
]
ax.legend(handles=legend_handles, title="Heatwave Type")

# Labels and formatting
ax.set_ylabel("GPP")
ax.set_title("GPP before, during, and after heatwaves\nColored by heatwave type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculating differences in deviance #########################################

# Mean deviation by heatwave type
from scipy import stats
import pandas as pd

GPP_ttest = ttest_by_cat(df=all_heatwaves_df,testing_var="GPP_mean_dev_diff",grouping_var="top_heatwave")
RECO_ttest = ttest_by_cat(df=all_heatwaves_df,testing_var="RECO_mean_dev_diff",grouping_var="top_heatwave")
NEE_ttest = ttest_by_cat(df=all_heatwaves_df,testing_var="NEE_mean_dev_diff",grouping_var="top_heatwave")

# Mean deviation by season
GPP_season_ttest = ttest_by_cat(df=all_heatwaves_df,testing_var="GPP_mean_dev_diff",grouping_var="Season")
RECO_season_ttest = ttest_by_cat(df=all_heatwaves_df,testing_var="RECO_mean_dev_diff",grouping_var="Season")
NEE_season_ttest = ttest_by_cat(df=all_heatwaves_df,testing_var="NEE_mean_dev_diff",grouping_var="Season")


# Density plots


