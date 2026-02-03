'''
This script runs initial analysis of heatwave impacts on carbon cycling.
Some things I'm interested in: comparison of GPP before, during, and after
heatwaves.
'''
import os
os.chdir("/Users/marleeyork/Documents/project2/Analysis")
import summaries_functions
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

# Lets load in the data
all_heatwaves_df = pd.read_csv("/Users/marleeyork/Documents/project2/data/heatwaves/all_heatwaves_df.csv")
df = pd.read_csv("/Users/marleeyork/Documents/project2/data/cleaned/AMF_DD.csv")

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

