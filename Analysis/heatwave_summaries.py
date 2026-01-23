'''
This script runs initial analysis of heatwave impacts on carbon cycling.
Some things I'm interested in: comparison of GPP before, during, and after
heatwaves.
'''
import summarues_functions
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timedelta

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
                                 after_lag = 10)

all_heatwaves_df = calc_flux_avg(flux_name = "RECO",
                                 heatwaves_df = all_heatwaves_df,
                                 flux_df = df,
                                 before_lag = 10,
                                 after_lag = 10)

all_heatwaves_df = calc_flux_avg(flux_name = "NEE",
                                 heatwaves_df = all_heatwaves_df,
                                 flux_df = df,
                                 before_lag = 10,
                                 after_lag = 10)

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
fig, ax = plt.subplots(figsize=(12, 6))

box_data = []
labels = []

# Build boxplot data
for hw_type in categories:
    type_df = all_heatwaves_df[all_heatwaves_df["top_heatwave"] == hw_type]

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

# NOW FOR NEE
fig, ax = plt.subplots(figsize=(12, 6))

box_data = []
labels = []

# Build boxplot data
for hw_type in categories:
    type_df = all_heatwaves_df[all_heatwaves_df["top_heatwave"] == hw_type]

    for period in ["NEE_before", "NEE_during", "NEE_after"]:
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
#  Looking at before and after averages for only min/max/mean heatwaves
###############################################################################

heatwave_df = calc_flux_avg(flux_name = "GPP", 
                            heatwaves_df = heatwave_df, 
                            flux_df = df, 
                            before_lag = 10, 
                            after_lag = 10)

heatwave_df = calc_flux_avg(flux_name = "RECO",
                            heatwaves_df = heatwave_df,
                            flux_df = df,
                            before_lag = 10,
                            after_lag = 10)

heatwave_df = calc_flux_avg(flux_name = "NEE",
                            heatwaves_df = heatwave_df,
                            flux_df = df,
                            before_lag = 10,
                            after_lag = 10)

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

