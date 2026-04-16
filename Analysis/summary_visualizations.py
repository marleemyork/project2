'''
This is for all visualizations related to summarizing the heatwaves.
Summaries will be made using all_heatwaves_df since they are not related to fluxes.
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import geopandas as gpd

# Importing heatwave events
all_heatwaves_df = pd.read_csv("/Users/marleeyork/Documents/project2/data/heatwaves/all_heatwaves_df.csv")
all_heatwaves_df["start_dates"] = pd.to_datetime(all_heatwaves_df.start_dates)
all_heatwaves_df["end_dates"] = pd.to_datetime(all_heatwaves_df.end_dates)
all_heatwaves_df = all_heatwaves_df.iloc[:,4:]
all_heatwaves_df.dtypes

# Load in badm data and merge with our heatwave events
badm = loadBADM(path="/Users/marleeyork/Documents/project2/data/BADM",skip=["/Users/marleeyork/Documents/project2/data/BADM/AMF_CA-Qc2_BIF_20250731.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-Cop_BIF_20240229.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-UiD_BIF_20251017.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-BMM_BIF_20221003.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-NGC_BIF_20231208.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-Snf_BIF_20250731.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-AR2_BIF_20231031.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-SdH_BIF_20241204.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-CAK_BIF_20250731.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-AR1_BIF_20231031.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-BMM_BIF_20221003.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-NGC_BIF_20231208.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_CA-Qc2_BIF_20250731.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-Snf_BIF_20250731.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-SdH_BIF_20241204.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-Fcr_BIF_20240401.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-Sta_BIF_20250731.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_US-AR1_BIF_20231031.xlsx",
                                                                            "/Users/marleeyork/Documents/project2/data/BADM/AMF_CA-Ca3_BIF_20241204.xlsx"
                                                                            ],
                column='VARIABLE',value='DATAVALUE',measure=['IGBP','CLIMATE_KOEPPEN','MAT','MAP','LOCATION_LAT','LOCATION_LONG','LOCATION_ELEV'],
                file_type='xslx')

all_heatwaves_df = pd.merge(all_heatwaves_df,badm[["Site","MAP","LOCATION_LAT","LOCATION_LONG","LOCATION_ELEV"]],on="Site",how="left")
df = pd.merge(df,badm[["Site","MAT","MAP","CLIMATE_KOEPPEN","LOCATION_LAT","LOCATION_LONG","LOCATION_ELEV"]],on="Site",how="left")

# Importing the dataframe of covariates
# I just loaded this in using the preprocessing_workflow.py

# 1. Average number of heatwave days over time.
# Extract year
df["year"] = df["date"].dt.year

# -----------------------------
# 3. COMPUTE SITE-YEAR METRICS
# -----------------------------
site_year = (
    df.groupby(["Site", "year"])
      .agg(
          n_days_observed=("date", "nunique"),
          n_hw_days=("heatwave_indicator", "sum")  # assumes 1/0
      )
      .reset_index()
)

# Fraction of time in heatwave
site_year["hw_fraction"] = (
    site_year["n_hw_days"] / site_year["n_days_observed"]
)

# -----------------------------
# 4. OPTIONAL: FILTER PARTIAL YEARS
# -----------------------------
# Compute fraction of year observed
site_year["days_in_year"] = np.where(
    site_year["year"].apply(lambda y: pd.Timestamp(year=y, month=12, day=31).is_leap_year),
    366,
    365
)

site_year["year_fraction_observed"] = (
    site_year["n_days_observed"] / site_year["days_in_year"]
)

# Optional threshold (recommended)
min_fraction = 0.25
site_year = site_year.loc[
    site_year["year_fraction_observed"] >= min_fraction
].copy()

# -----------------------------
# 5. SUMMARIZE ACROSS SITES
# -----------------------------
yearly_summary = (
    site_year.groupby("year")
    .agg(
        mean_hw_fraction=("hw_fraction", "mean"),
        sd_hw_fraction=("hw_fraction", "std"),
        n_sites=("Site", "nunique")
    )
    .reset_index()
)

yearly_summary["se"] = (
    yearly_summary["sd_hw_fraction"] / np.sqrt(yearly_summary["n_sites"])
)

# -----------------------------
# 6. PLOT
# -----------------------------
plt.figure(figsize=(9, 5.5))

# Optional: faint site-level lines (very nice visually)
for site, g in site_year.groupby("Site"):
    g = g.sort_values("year")
    plt.plot(
        g["year"],
        g["hw_fraction"],
        color="lightgray",
        alpha=0.3,
        linewidth=1
    )

# Mean line
plt.plot(
    yearly_summary["year"],
    yearly_summary["mean_hw_fraction"],
    color="black",
    linewidth=2.5,
    label="Mean across sites"
)

# SE ribbon
plt.fill_between(
    yearly_summary["year"],
    yearly_summary["mean_hw_fraction"] - yearly_summary["se"],
    yearly_summary["mean_hw_fraction"] + yearly_summary["se"],
    alpha=0.25
)

plt.xlabel("Year")
plt.ylabel("Fraction of days in heatwave")
plt.title("Average Heatwave Frequency Across Sites Over Time")

sns.despine()
plt.tight_layout()
plt.show()


# 2. Heatwave days frequency over time faceted by heatwave type.
# Total observed days per site-year (same denominator for all types)
site_year_total = (
    df.groupby(["Site", "year"])
      .agg(n_days_observed=("date", "nunique"))
      .reset_index()
)

# Heatwave days per type
site_year_type = (
    df[df["heatwave_indicator"] == 1]
    .groupby(["Site", "year", "top_heatwave"])
    .agg(n_hw_days=("heatwave_indicator", "sum"))
    .reset_index()
)

# Merge with total days
site_year_type = site_year_type.merge(
    site_year_total,
    on=["Site", "year"],
    how="left"
)

# Compute fraction
site_year_type["hw_fraction"] = (
    site_year_type["n_hw_days"] / site_year_type["n_days_observed"]
)

# Get all combinations
all_types = df["top_heatwave"].dropna().unique()

full_index = pd.MultiIndex.from_product(
    [
        df["Site"].unique(),
        df["year"].unique(),
        all_types
    ],
    names=["Site", "year", "heatwave_type"]
)

site_year_type = (
    site_year_type.set_index(["Site", "year", "top_heatwave"])
    .reindex(full_index)
    .reset_index()
)

# Merge total days again
site_year_type = site_year_type.merge(
    site_year_total,
    on=["Site", "year"],
    how="left"
)

# Fill missing heatwave counts with 0
site_year_type["n_hw_days"] = site_year_type["n_hw_days"].fillna(0)

# Recompute fraction
site_year_type["hw_fraction"] = (
    site_year_type["n_hw_days"] / site_year_type["n_days_observed_y"]
)

# Filter partial years
site_year_type["days_in_year"] = np.where(
    site_year_type["year"].apply(lambda y: pd.Timestamp(year=y, month=12, day=31).is_leap_year),
    366,
    365
)

site_year_type["year_fraction_observed"] = (
    site_year_type["n_days_observed_y"] / site_year_type["days_in_year"]
)

min_fraction = 0.25
site_year_type = site_year_type.loc[
    site_year_type["year_fraction_observed"] >= min_fraction
].copy()

# Aggregate across sites
yearly_type = (
    site_year_type.groupby(["year", "heatwave_type"])
    .agg(
        mean_hw_fraction=("hw_fraction", "mean"),
        sd_hw_fraction=("hw_fraction", "std"),
        n_sites=("Site", "nunique")
    )
    .reset_index()
)

yearly_type["se"] = (
    yearly_type["sd_hw_fraction"] / np.sqrt(yearly_type["n_sites"])
)

# Faceted plot
g = sns.FacetGrid(
    yearly_type,
    col="heatwave_type",
    col_wrap=3,
    height=3.5,
    sharey=True
)

def plot_with_ci(data, **kwargs):
    plt.plot(
        data["year"],
        data["mean_hw_fraction"],
        color="black",
        linewidth=2
    )
    plt.fill_between(
        data["year"],
        data["mean_hw_fraction"] - data["se"],
        data["mean_hw_fraction"] + data["se"],
        alpha=0.3
    )

g.map_dataframe(plot_with_ci)

g.set_axis_labels("Year", "Fraction of days in heatwave")
g.set_titles("{col_name}",fontsize=30)

for ax in g.axes.flat:
    sns.despine(ax=ax)

plt.tight_layout()
plt.show()

# Figure 3: Seasonal timing of heatwaves
# Get the doy
df["doy"] = df["date"].dt.dayofyear

# Get total number of each day of the year that we have
total_doy = df.groupby("doy").agg(n_days=("date","count")).reset_index()

# Heatwave days per DOY and type
hw_doy_type = df[df.heatwave_indicator==1].groupby(["doy","top_heatwave"]).agg(n_hw_days=("heatwave_indicator","sum")).reset_index()

# Fill in missing DOY-type combos with 0
all_doy = np.arange(1,367)
all_types = df.top_heatwave.dropna().unique()
full_index = pd.MultiIndex.from_product([all_doy,all_types],names=["doy","heatwave_type"])
hw_doy_type = hw_doy_type.set_index(["doy","heatwave_type"]).reindex(full_index).reset_index()
hw_doy_type["n_hw_days"] = hw_doy_type["n_hw_days"].fillna(0)

# Merge with the total number of days and calculate the average
hw_doy_type = pd.merge(hw_doy_type,total_doy,on="doy",how="left")
hw_doy_type["doy_type_avg"] = hw_doy_type["n_hw_days"] / hw_doy_type["n_days"]

# Plotting without a smooth first
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=hw_doy_type,
    x="doy",
    y="doy_type_avg",
    hue="heatwave_type",
    linewidth=2
)

plt.xlabel("Day of Year")
plt.ylabel("Probability of Heatwave Day")
plt.title("Seasonal Timing of Heatwaves by Type")

sns.despine()
plt.tight_layout()
plt.show()

# Lets smooth it now
hw_doy_type = hw_doy_type.sort_values("doy")
hw_doy_type["hw_prob_smooth"] = (hw_doy_type.groupby("heatwave_type")["doy_type_avg"]
                                 .transform(lambda x: x.rolling(window=15,center=True,min_periods=1).mean()))

# Addint an overall average across all heatwave types
# Total heatwave days per DOY (all types combined)
hw_doy_all = (
    df[df["heatwave_indicator"] == 1]
    .groupby("doy")
    .agg(n_hw_days=("heatwave_indicator", "sum"))
    .reset_index()
)

# Merge with total observed days
hw_doy_all = hw_doy_all.merge(total_doy, on="doy", how="left")

# Fill missing DOYs with 0 heatwaves
hw_doy_all["n_hw_days"] = hw_doy_all["n_hw_days"].fillna(0)

# Compute probability
hw_doy_all["hw_probability"] = (
    hw_doy_all["n_hw_days"] / hw_doy_all["n_days"]
)

# Smooth that john
hw_doy_all = hw_doy_all.sort_values("doy")

hw_doy_all["hw_prob_smooth"] = (
    hw_doy_all["hw_probability"]
    .rolling(window=15, center=True, min_periods=1)
    .mean()
)

# Convert DOY to months for visualizing
def doy_to_month(doy):
    return datetime.datetime(2001, 1, 1) + datetime.timedelta(doy - 1)

month_ticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Plotting with the smoothing now
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=hw_doy_type,
    x="doy",
    y="hw_prob_smooth",
    hue="heatwave_type",
    linewidth=2,
    alpha=0.9
)

plt.plot(
    hw_doy_all["doy"],
    hw_doy_all["hw_prob_smooth"],
    color="black",
    linewidth=3.5,
    label="All heatwaves"
)

plt.xlabel("Day of Year")
plt.ylabel("Probability of Heatwave Day")
plt.title("Seasonal Timing of Heatwaves by Type")
plt.xticks(month_ticks, month_labels)

sns.despine()
plt.tight_layout()
plt.show()

# Figure 4: Map of heatwave frequency across sites
# Converting latitude and longitude to float values
all_heatwaves_df["LOCATION_LAT"] = all_heatwaves_df.LOCATION_LAT.astype("float")
all_heatwaves_df["LOCATION_LONG"] = all_heatwaves_df.LOCATION_LONG.astype("float")
df["LOCATION_LAT"] = df.LOCATION_LAT.astype("float")
df["LOCATION_LONG"] = df.LOCATION_LONG.astype("float")

# Summary heatwave frequency by site
site_summary = (
    df.groupby("Site")
      .agg(
          latitude=("LOCATION_LAT", "first"),
          longitude=("LOCATION_LONG", "first"),
          n_days_observed=("date", "nunique"),
          n_hw_days=("heatwave_indicator", "sum")
      )
      .reset_index()
)

site_summary["hw_fraction"] = (
    site_summary["n_hw_days"] / site_summary["n_days_observed"]
)

site_summary["hw_percent"] = site_summary["hw_fraction"] * 100

# Plot
world = gpd.read_file(
    "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
)

north_america = world[world["ADMIN"].isin(["United States of America","Canada"])]

fig, ax = plt.subplots(figsize=(10, 7))

# Plot US background
north_america.plot(ax=ax, color="white", edgecolor="black", linewidth=0.5)

# Plot sites
sc = ax.scatter(
    site_summary["longitude"],
    site_summary["latitude"],
    c=site_summary["hw_percent"],
    s=90,
    cmap="YlOrRd",
    edgecolor="black",
    linewidth=0.5
)

# Colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("Heatwave days (% of observed days)")

# Labels
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Heatwave Frequency Across Sites")

sns.despine()
plt.tight_layout()
plt.show()

# Figure 5: Map of heatwave dominance
# Get counts of each type of heatwave for each site
site_type_counts = (
    df[df["heatwave_indicator"] == 1]
    .groupby(["Site", "top_heatwave"])
    .size()
    .reset_index(name="n_hw_days")
)

# For each site, pick the type with the most heatwave days
dominant_type = (
    site_type_counts.sort_values(["Site", "n_hw_days"], ascending=[True, False])
    .drop_duplicates(subset="Site")
    .rename(columns={"top_heatwave": "dominant_type"})
)

# Add site coordinates
site_coords = (
    df.groupby("Site")
    .agg(
        latitude=("LOCATION_LAT", "first"),
        longitude=("LOCATION_LONG", "first")
    )
    .reset_index()
)

site_dominant = site_coords.merge(dominant_type[["Site", "dominant_type", "n_hw_days"]],
                                  on="Site", how="left")

# Optional: drop sites with no heatwaves
site_dominant = site_dominant.dropna(subset=["dominant_type"]).copy()


# Lets plot it now
type_order = sorted(site_dominant["dominant_type"].dropna().unique())

site_dominant["dominant_type"] = pd.Categorical(
    site_dominant["dominant_type"],
    categories=type_order,
    ordered=True
)


# Build color map
palette = sns.color_palette("tab10", n_colors=len(type_order))
color_dict = dict(zip(type_order, palette))

fig, ax = plt.subplots(figsize=(11, 8))

north_america.plot(
    ax=ax,
    color="whitesmoke",
    edgecolor="black",
    linewidth=0.8
)

# Plot each type separately so legend works cleanly
for hw_type in type_order:
    sub = site_dominant[site_dominant["dominant_type"] == hw_type]
    ax.scatter(
        sub["longitude"],
        sub["latitude"],
        s=110,
        color=color_dict[hw_type],
        edgecolor="black",
        linewidth=0.5,
        label=hw_type,
        zorder=3
    )

ax.set_xlim(-140, -60)
ax.set_ylim(25, 65)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Dominant Heatwave Type Across Sites")

ax.legend(
    title="Dominant heatwave type",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False
)

sns.despine()
plt.tight_layout()
plt.show()

# Barplots of heatwave frequencies by top heatwave type
# Count events by type
event_counts = (
    all_heatwaves_df.groupby("top_heatwave")
      .size()
      .reset_index(name="n_events")
)

# Convert to percent
event_counts["percent"] = (
    event_counts["n_events"] / event_counts["n_events"].sum()
) * 100

# Sort
event_counts = event_counts.sort_values("percent", ascending=False)

# Plot
plt.figure(figsize=(8, 5))

sns.barplot(
    data=event_counts,
    x="top_heatwave",
    y="percent",
    color="steelblue"
)

plt.ylabel("Percent of heatwave events")
plt.xlabel("Heatwave type")
plt.title("Heatwave Event Frequency by Type")

plt.xticks(rotation=45, ha="right")

sns.despine()
plt.tight_layout()
plt.show()


# Count heatwave days by type
day_counts = (
    df[df["heatwave_indicator"] == 1]
    .groupby("top_heatwave")
    .size()
    .reset_index(name="n_days")
)

# Convert to percent
day_counts["percent"] = (
    day_counts["n_days"] / day_counts["n_days"].sum()
) * 100

# Sort same order as events (important for comparison)
day_counts = day_counts.set_index("top_heatwave").loc[
    event_counts["top_heatwave"]
].reset_index()

# Plot
plt.figure(figsize=(8, 5))

sns.barplot(
    data=day_counts,
    x="top_heatwave",
    y="percent",
    color="darkorange"
)

plt.ylabel("Percent of heatwave days")
plt.xlabel("Heatwave type")
plt.title("Heatwave Day Frequency by Type")

plt.xticks(rotation=45, ha="right")

sns.despine()
plt.tight_layout()
plt.show()
