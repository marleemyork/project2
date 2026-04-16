'''
This script includes functions that check the quality of heatwaves based on 
the AmeriFlux QAQC temperature flags.
'''
import pandas as pd
import os
os.chdir(path="/Users/marleeyork/Documents/project2/heatwave_definition")
from define_heatwaves import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def avg_QAQC_check(site_heatwave_dictionary, dates, TA_QAQC, QAQC_threshold,
                   heatwave_threshold):
    '''
    Description
    -----------
    This function identifies heatwaves that are invalid due to having too high
    of a percentage of low quality AmeriFlux data.
    
    Parameters
    ----------
    site_heatwave_dictionary : TYPE
        DESCRIPTION. Dictionary provided by fit_heatwaves with method="EHF" for
        one site. E.g. site_heatwave_dictionary = heatwaves_EHF['US-GLE']
    dates : TYPE
        DESCRIPTION. Dates for AmeriFlux data associated with the following QAQC values.
    TA_QAQC : TYPE
        DESCRIPTION. TA_F_QAQC values associated with the above dates for one given site.
    QAQC_threshold : TYPE
        DESCRIPTION. The bottom TA_F_QAQC threshold that determines a day of inacceptable data.
    heatwave_threshold : TYPE
        DESCRIPTION. The percentage of inacceptable data days a heatwave can have and 
        still be considered a valid heatwave.
    Returns
    -------
    heatwave_qaqc : TYPE
        DESCRIPTION. DataFrame of start_date, end_date, percentage of days in the 
        heatwave that have a QAQC below the necessary threshold (QAQC_percentage), and validity of the
        heatwave based on accepted QAQC_percentage as defined by heatwave threshold (heatwave_invalidity)
    '''
    # If the site didn't have any valid data, then we skip it
    if pd.isna(site_heatwave_dictionary['start_dates']).all():
        return 
    # Otherwise, we check for invalid heatwave flags
    start_dates = site_heatwave_dictionary['start_dates']
    end_dates = site_heatwave_dictionary['end_dates']
    ta_qaqc = pd.DataFrame({'dates':dates,'QAQC':TA_QAQC})
    heatwave_qaqc = pd.DataFrame(columns=['start_date','end_date','QAQC_percentage','heatwave_invalidity'])
    # Loop through each heatwave
    for start, end in zip(start_dates,end_dates):
        # Create a range of dates between the start and end
        date_range = pd.date_range(start=start, end=end)
        # Find the QAQC values in these dates
        heatwave_QAQC_values = ta_qaqc[ta_qaqc['dates'].isin(date_range)]
        # Determine if they are flagged as being below the threshold
        QAQC_flag = []
        for quality in heatwave_QAQC_values.QAQC:
            if (quality < QAQC_threshold):
                QAQC_flag.append(1)
            else:
                QAQC_flag.append(0)
        
        if len(QAQC_flag) == 0:
            continue
        else:
            # Find percentage of flagged days
            QAQC_percentage = sum(QAQC_flag) / len(QAQC_flag)
            invalidity_flag = 0 if (QAQC_percentage < heatwave_threshold) else 1
            # Add start date, end date, percentage of bad data days, and heatwave validity to dataframe
            this_site = pd.DataFrame({'start_date':[start],
                                  'end_date':[end],
                                  'QAQC_percentage':[QAQC_percentage],
                                  'heatwave_invalidity':[invalidity_flag]})
            # Concatenate with QAQC of other heatwaves
            heatwave_qaqc = pd.concat([heatwave_qaqc,this_site])
    return heatwave_qaqc

def minmax_QAQC_check(site_heatwave_dictionary, dates, TA, TA_QAQC, heatwave_threshold,method='max'):
    '''
    Description
    -----------
    This function identifies heatwaves that are invalid due to having too high
    of a percentage of low quality AmeriFlux data. This is for those heatwaves defined by
    hourly data, like the min/max approaches.
    
    I don't think I actually need to use this, but TBD. Could still do an hourly thing
    where if the max/min temperature is bad, it defaults to the next temperature.
    Our min/max temperatures were really well correlated with PRISM data though.
    
    Parameters
    ----------
    site_heatwave_dictionary : DICTIONARY
        DESCRIPTION. Dictionary provided by fit_heatwaves with method="EHF" for
        one site. E.g. site_heatwave_dictionary = heatwaves_EHF['US-GLE']
    dates : datetime vector
        DESCRIPTION. Dates for AmeriFlux data associated with the following QAQC values.
    TA_QAQC : float vector
        DESCRIPTION. TA_F_QAQC values associated with the above dates for one given site.
    QAQC_threshold : float decimal [0,1]
        DESCRIPTION. The bottom TA_F_QAQC threshold that determines a day of inacceptable data.
    heatwave_threshold : float decimal [0,1]
        DESCRIPTION. The percentage of inacceptable data days a heatwave can have and 
        still be considered a valid heatwave.
    method : string ['max' or 'min']
        DESCRIPTION. Specifies whether this is being used on heatwaves defined by
        maximum or minimum daily temperatures
    
    Returns
    -------
    heatwave_qaqc : TYPE
        DESCRIPTION. DataFrame of start_date, end_date, percentage of days in the 
        heatwave that have a QAQC below the necessary threshold (QAQC_percentage), and validity of the
        heatwave based on accepted QAQC_percentage as defined by heatwave threshold (heatwave_invalidity)
    '''
    # If the site doesn't have valid data, then skip it
    if pd.isna(site_heatwave_dictionary['start_dates']).all():
        return 
    
    # Initialize list to hold flags for heatwaves surpassing the valid amount of downscaled data
    flag = []
    # Get find max hourly temperature
    hourly_TA = pd.DataFrame({'dates':dates,'TA':TA,'TA_QAQC':TA_QAQC})
    hourly_TA['dates_dt'] = pd.to_datetime(hourly_TA['dates'].dt.date)
    
    # Loop through each heatwave
    data = site_heatwave_dictionary['summary']
    for i in range(data.shape[0]):
    
        # Get the ith heatwave — this was the major bug
        this_heatwave = data.iloc[i]

        # Build the list of dates in the heatwave
        this_heatwave_dates = pd.date_range(
            this_heatwave.start_dates, 
            this_heatwave.end_dates
            )

        # Extract hourly records in those dates
        this_heatwave_hourly = hourly_TA[
            hourly_TA['dates_dt'].isin(this_heatwave_dates)
            ]
    
        downscaled = []

        for date in this_heatwave_dates:

            # Subset hourly data for the date
            daily_hourly = this_heatwave_hourly[
                this_heatwave_hourly['dates_dt'] == date
                ]

            if len(daily_hourly) == 0:
                # Handle empty day
                downscaled.append(1)   # or whatever logic makes sense
                continue
        
            # Identify the hour with the max/min temperature
            if method == 'max':
                idx = daily_hourly['TA'].idxmax()
            else:
                idx = daily_hourly['TA'].idxmin()
        
            qaqc = daily_hourly.loc[idx, 'TA_QAQC']

            # Mark whether gap filled based on QC coding
            downscaled.append(1 if qaqc == 2 else 0)

        heatwave_percentage = sum(downscaled) / len(downscaled)
        fail = 1 if heatwave_percentage >= heatwave_threshold else 0

        flag.append(fail)

    # Merge this onto the heatwave summary
    data['QAQC_flag'] = flag
    # If it is bad, then flag it as bad
    heatwave_qaqc = data
    
    return heatwave_qaqc



def remove_invalid_heatwaves(heatwaves_dictionary, invalid_heatwaves):
    '''
    Parameters
    ----------
    heatwave_dictionary : Dictionary
        DESCRIPTION. The heatwave dictionary for all site.
    invalid_heatwaves : DataFrame
        DESCRIPTION. DataFrame of invalid heatwaves with columns=['Site','start_dates','end_dates']

    Returns
    -------
    site_heatwave_dictionary : TYPE
        DESCRIPTION. Heatwave dictionary returned with the invalid heatwaves removed.

    '''
    # Remove these invalid heatwaves from the list of heatwaves
    for site in invalid_heatwaves.Site.unique():
        print(f"Cleaning up site {site}...")
        # Isolate invalid heatwaves for that site
        site_invalid = invalid_heatwaves[invalid_heatwaves['Site']==site]
        # Loop through the 
        for i in range(site_invalid.shape[0]):
            invalid_heatwave = site_invalid.iloc[i]
            print(f"Removing invalid heatwave {invalid_heatwave}.")
            # Drop the invalid from the start date
            heatwaves_dictionary[site]['start_dates'] = [
                d for d in heatwaves_dictionary[site]['start_dates']
                if d != invalid_heatwave.start_date
                ]
            # Drop the invalid from the end date
            heatwaves_dictionary[site]['end_dates'] = [
                d for d in heatwaves_dictionary[site]['end_dates']
                if d != invalid_heatwave.end_date
                ]
            # Drop the invalid from the summary
            data = heatwaves_dictionary[site]['summary']
            heatwaves_dictionary[site]['summary'] = data[
                data['start_dates'] != invalid_heatwave.start_date
                ].reset_index(drop=True)
            # Change the indicator values to 0 at these invalid heatwaves
            data = heatwaves_dictionary[site]['indicator']
            mask = (data['date'] >= invalid_heatwave.start_date) & \
                   (data['date'] <= invalid_heatwave.end_date)
            data.loc[mask, 'avg_indicator'] = 0
            heatwaves_dictionary[site]['indicator'] = data
            # Remove these same dates from periods
            data = heatwaves_dictionary[site]['periods']
            mask = (data['date'] >= invalid_heatwave.start_date) & \
                   (data['date'] <= invalid_heatwave.end_date)
            heatwaves_dictionary[site]['periods'] = heatwaves_dictionary[site]['periods'][~mask]
            # Remove the invalid heatwave dates from precip
            data = heatwaves_dictionary[site]['precip']
            mask = (data['start_date'] == invalid_heatwave.start_date) & \
                   (data['end_date'] == invalid_heatwave.end_date)
            heatwaves_dictionary[site]['precip'] = heatwaves_dictionary[site]['precip'][~mask]
            # Remove the invalid heatwave dates from swc
            data = heatwaves_dictionary[site]['swc']
            mask = (data['start_date'] == invalid_heatwave.start_date) & \
                   (data['end_date'] == invalid_heatwave.end_date)
            heatwaves_dictionary[site]['swc'] = heatwaves_dictionary[site]['swc'][~mask]
    
    return heatwaves_dictionary


def plot_flux_QAQC(site, date, NEE, GPP, Reco, NEE_QC, Temp):
    # Build dataframe
    df = pd.DataFrame({
        "site": site,
        "date": pd.to_datetime(date),
        "NEE": NEE,
        "GPP": GPP,
        "Reco": Reco,
        "NEE_QC": NEE_QC,
        "Temp": Temp
    })
    
    # Sort values
    df = df.sort_values(["site", "date"])
    
    # Loop through sites
    for s in df["site"].dropna().unique():
        d = df[df["site"] == s].copy()
        
        fig, ax = plt.subplots(4, 1, figsize=(12, 14))
        fig.suptitle(f"Flux QA/QC: {s}", fontsize=14)
        
        # 1. NEE over time, QC < 0.75 in red
        nee_bad = d["NEE_QC"] < 0.75
        ax[0].scatter(d.loc[~nee_bad, "date"], d.loc[~nee_bad, "NEE"], s=8, label="QC ≥ 0.75", color="grey")
        ax[0].scatter(d.loc[nee_bad, "date"], d.loc[nee_bad, "NEE"], s=8, color="red", label="QC < 0.75")
        ax[0].axhline(0, linestyle="--", linewidth=1)
        ax[0].set_ylabel("NEE")
        ax[0].set_title("NEE over time")
        ax[0].legend()
        
        # 2. GPP over time, GPP < 0 in bright blue
        gpp_neg = d["GPP"] < 0
        ax[1].scatter(d.loc[~gpp_neg, "date"], d.loc[~gpp_neg, "GPP"], s=8, label="GPP ≥ 0", color="grey")
        ax[1].scatter(d.loc[gpp_neg, "date"], d.loc[gpp_neg, "GPP"], s=8, color="deepskyblue", label="GPP < 0")
        ax[1].axhline(0, linestyle="--", linewidth=1)
        ax[1].set_ylabel("GPP")
        ax[1].set_title("GPP over time")
        ax[1].legend()
        
        # 3. Reco over time
        ax[2].scatter(d["date"], d["Reco"], s=8,color="grey")
        ax[2].axhline(0, linestyle="--", linewidth=1)
        ax[2].set_ylabel("Reco")
        ax[2].set_title("Reco over time")
        
        # 4. GPP by temperature, Temp < 0 in bright blue
        temp_freezing = d["Temp"] < 0
        ax[3].scatter(d.loc[~temp_freezing, "Temp"], d.loc[~temp_freezing, "GPP"], s=8, label="Temp ≥ 0°C",color="grey")
        ax[3].scatter(d.loc[temp_freezing, "Temp"], d.loc[temp_freezing, "GPP"], s=8, color="deepskyblue", label="Temp < 0°C")
        ax[3].axvline(0, linestyle="--", linewidth=1)
        ax[3].axhline(0, linestyle="--", linewidth=1)
        ax[3].set_xlabel("Temperature")
        ax[3].set_ylabel("GPP")
        ax[3].set_title("GPP vs Temperature")
        ax[3].legend()
        
        plt.tight_layout()
        plt.show()
        
def plot_flux_QAQC_to_pdf(site, date, NEE, GPP, Reco, NEE_QC, Temp, heatwave,
                          output_pdf="flux_QAQC_all_sites.pdf"):
    """
    Create one multi-page PDF with QA/QC plots for each site.

    Parameters
    ----------
    site : array-like
        Site identifier for each observation.
    date : array-like
        Dates for each observation.
    NEE : array-like
        Net ecosystem exchange values.
    GPP : array-like
        Gross primary productivity values.
    Reco : array-like
        Ecosystem respiration values.
    NEE_QC : array-like
        NEE quality control values.
    Temp : array-like
        Temperature values.
    heatwave : array-like
        Heatwave indicator (1 = heatwave point, otherwise not).
    output_pdf : str
        Output PDF filename.
    """

    df = pd.DataFrame({
        "site": site,
        "date": pd.to_datetime(date),
        "NEE": NEE,
        "GPP": GPP,
        "Reco": Reco,
        "NEE_QC": NEE_QC,
        "Temp": Temp,
        "heatwave": heatwave
    })

    df = df.sort_values(["site", "date"])

    def plot_with_heatwave_outline(ax, x, y, bad_qc_mask, heatwave_mask,
                                   x_label="", y_label="", title="",
                                   add_hline_zero=False, add_vline_zero=False):
        # Base colors: red if QC < 0.75, black otherwise
        colors = ["red" if bad else "lightgrey" for bad in bad_qc_mask]

        # Plot all points
        ax.scatter(x, y, c=colors, s=10, alpha=0.7, linewidths=0)

        # Overlay heatwave points with bold black outline
        hw = heatwave_mask.fillna(False)
        ax.scatter(
            x[hw], y[hw],
            facecolors="none",
            edgecolors="black",
            s=42,
            linewidths=1.4
        )

        if add_hline_zero:
            ax.axhline(0, linestyle="--", linewidth=1, color="gray")
        if add_vline_zero:
            ax.axvline(0, linestyle="--", linewidth=1, color="gray")

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    with PdfPages(output_pdf) as pdf:
        for s in sorted(df["site"].dropna().unique()):
            d = df[df["site"] == s].copy()

            bad_qc = d["NEE_QC"] < 0.75
            heatwave_mask = d["heatwave"] == 1

            fig, ax = plt.subplots(4, 1, figsize=(12, 14))
            fig.suptitle(f"Flux QA/QC: {s}", fontsize=14)

            # 1. NEE over time
            plot_with_heatwave_outline(
                ax=ax[0],
                x=d["date"],
                y=d["NEE"],
                bad_qc_mask=bad_qc,
                heatwave_mask=heatwave_mask,
                y_label="NEE",
                title="NEE over time",
                add_hline_zero=True
            )

            # 2. GPP over time
            plot_with_heatwave_outline(
                ax=ax[1],
                x=d["date"],
                y=d["GPP"],
                bad_qc_mask=bad_qc,
                heatwave_mask=heatwave_mask,
                y_label="GPP",
                title="GPP over time",
                add_hline_zero=True
            )

            # 3. Reco over time
            plot_with_heatwave_outline(
                ax=ax[2],
                x=d["date"],
                y=d["Reco"],
                bad_qc_mask=bad_qc,
                heatwave_mask=heatwave_mask,
                y_label="Reco",
                title="Reco over time",
                add_hline_zero=False
            )

            # 4. GPP vs temperature
            plot_with_heatwave_outline(
                ax=ax[3],
                x=d["Temp"],
                y=d["GPP"],
                bad_qc_mask=bad_qc,
                heatwave_mask=heatwave_mask,
                x_label="Temperature",
                y_label="GPP",
                title="GPP vs Temperature",
                add_hline_zero=False,
                add_vline_zero=True
            )

            # Legend on first panel only
            legend_handles = [
                plt.Line2D([], [], marker='o', linestyle='None',
                           markerfacecolor='black', markeredgecolor='black',
                           markersize=5, label='QC ≥ 0.75'),
                plt.Line2D([], [], marker='o', linestyle='None',
                           markerfacecolor='red', markeredgecolor='red',
                           markersize=5, label='QC < 0.75'),
                plt.Line2D([], [], marker='o', linestyle='None',
                           markerfacecolor='none', markeredgecolor='black',
                           markeredgewidth=1.4, markersize=7, label='Heatwave = 1')
            ]
            ax[0].legend(handles=legend_handles, loc="best")

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved QA/QC plots to {output_pdf}")