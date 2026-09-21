
# Define functions for Gabors and Orientations (GAO) EEG decoding
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from rpy2.robjects import r
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.stats import zscore
from scipy.ndimage import gaussian_filter
import seaborn as sns
from mne.decoding import (
    SlidingEstimator
)
from sklearn.metrics import f1_score, roc_auc_score
from joblib import Parallel, delayed
from statannotations.Annotator import Annotator


bf_package=importr('BayesFactor')
font_sz = 18
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.labelsize'] = font_sz
plt.rcParams['ytick.labelsize'] = font_sz
plt.rcParams['axes.titlesize'] = font_sz
plt.rcParams['legend.fontsize'] = font_sz - 4



def calculate_vtc(sub_run_df_pre, FWHM):
    sub_run_df_whole = sub_run_df_pre.copy()


    task_list = sub_run_df_whole.task
    # print(task_list)
    unique_ordered_list = list(dict.fromkeys(task_list))
    # print(unique_ordered_list)

    return_df = pd.DataFrame()
    for task in unique_ordered_list:
        # print(task)
        sub_run_df = sub_run_df_whole[sub_run_df_whole.task==task]
        # print(sub_run_df['accuracy'])
        sub_run_df.loc[sub_run_df['rt'] == 'nan', "rt"] = None  # set nonresponses to None for interpolation
        sub_run_df.loc[(sub_run_df['accuracy'] != 1), "rt"] = None  # set probe resps to None

        trials_idx_freqcorr  = np.where(~sub_run_df["rt"].isna())[0]
        # print(trials_idx_freqcorr)
        # print(sub_run_df["rt"])
        slope,intercept = np.polyfit(trials_idx_freqcorr,sub_run_df["rt"][~sub_run_df["rt"].isna()],1) #calculate linear trend
        sub_run_df["detrended_RT"] = sub_run_df["rt"]-(intercept+slope*np.arange(len(sub_run_df))) #subtract linear trend
        sub_run_df["median_detrended_RT"] = np.nanmedian(sub_run_df["detrended_RT"])
        sub_run_df["zRT_detrended"] = zscore(sub_run_df["detrended_RT"].values, nan_policy="omit")
        sub_run_df["abs_zRT_detrended"] = sub_run_df["zRT_detrended"].apply(np.abs)
        sub_run_df["abs_zRT_interpolated_detrended"] = sub_run_df["abs_zRT_detrended"].interpolate(method="linear", limit_direction="both")
        sub_run_df['vtc_detrended'] = sub_run_df["abs_zRT_interpolated_detrended"]


        sub_run_df["zRT"] = zscore(sub_run_df["rt"].values, nan_policy="omit")
        sub_run_df["abs_zRT"] = sub_run_df["zRT"].apply(np.abs)
        sub_run_df["abs_zRT_interpolated"] = sub_run_df["abs_zRT"].interpolate(method="linear", limit_direction="both")
        sub_run_df['vtc'] = sub_run_df["abs_zRT_interpolated"]

        # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        # FWHM = 2*sqrt(2*ln(2)) * sigma
        # -> sigma = FWHM / (2*sqrt(2*ln(2)))
        sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
        # TODO - modify filter to adjust weights at edges
        sub_run_df["vtc_smooth"] = gaussian_filter(sub_run_df["abs_zRT_interpolated"].values, sigma=sigma)
        sub_run_df["vtc_smooth_detrended"] = gaussian_filter(sub_run_df["abs_zRT_interpolated_detrended"].values, sigma=sigma)
        sub_run_df['vtc_median'] = np.median(sub_run_df['vtc'])
        sub_run_df['vtc_median_detrended'] = np.median(sub_run_df['vtc_detrended'])
        sub_run_df['vtc_smooth_median_detrended'] = np.median(sub_run_df['vtc_smooth_detrended'])

        # test analysis - calculate quartile
        sub_run_df['vtc_upper_quartile_detrended'] = np.quantile(sub_run_df['vtc_detrended'], 0.75)
        sub_run_df['vtc_lower_quartile_detrended'] = np.quantile(sub_run_df['vtc_detrended'], 0.25)
        sub_run_df['vtc_smooth_upper_quartile_detrended'] = np.quantile(sub_run_df['vtc_smooth_detrended'], 0.75)
        sub_run_df['vtc_smooth_lower_quartile_detrended'] = np.quantile(sub_run_df['vtc_smooth_detrended'], 0.25)

        print(np.nanquantile(sub_run_df['detrended_RT'], 0.75))
        sub_run_df['RT_upper_quartile'] = np.nanquantile(sub_run_df['detrended_RT'], 0.75)
        sub_run_df['RT_lower_quartile'] = np.nanquantile(sub_run_df['detrended_RT'], 0.25)



        return_df = pd.concat([return_df, sub_run_df], axis=0)

        # plt.plot(sub_run_df['vtc_smooth_detrended'])
        # plt.axhline(np.median(sub_run_df['vtc_smooth_detrended']))
        # plt.show()

        # print(np.sum(sub_run_df["vtc_smooth_detrended"]<sub_run_df['vtc_smooth_median_detrended']))
        # print(np.sum(sub_run_df["vtc_smooth_detrended"]>sub_run_df['vtc_smooth_median_detrended']))

    return return_df


def get_image_cat(image):
    return image.split('/')[0]



def calculate_aprime(h,fa):
    '''
    Calculates sensitivity, according to the aprime formula
    '''
    if np.greater(h,fa): a = .5 + (((h-fa) * (1+h-fa)) / (4 * h * (1-fa)))
    elif np.less(h,fa): a = .5 - (((fa-h) * (1+fa-h)) / (4 * fa * (1-h)))
    else: a = .5
    return a


def calculate_CPT_aprime(cpt_df):
    cpt_df['hit'] = np.where(((cpt_df['accuracy']==1) & (cpt_df['correct_response']==1)),1,0)
    cpt_df['miss'] = np.where(((cpt_df['accuracy']==0) & (cpt_df['correct_response']==1)),1,0)
    cpt_df['FA'] = np.where(((cpt_df['accuracy']==0) & (cpt_df['correct_response']==0)),1,0)
    cpt_df['CR'] = np.where(((cpt_df['accuracy']==1) & (cpt_df['correct_response']==0)),1,0)

    h = np.sum(cpt_df['hit'])    / (np.sum(cpt_df['hit']) + np.sum(cpt_df['miss']))
    fa = np.sum(cpt_df['FA'])    / (np.sum(cpt_df['FA']) + np.sum(cpt_df['CR']))

    return calculate_aprime(h,fa), h, fa






def find_old_image(mem_rel_df):
    mem_rel_df['old_image'] = np.where((((mem_rel_df['accuracy_thresh3']==1) & (mem_rel_df['response'].isin(['z','x'])))) | (((mem_rel_df['accuracy_thresh3']==0) & (mem_rel_df['response'].isin(['m','n'])))) , mem_rel_df['image_left'], mem_rel_df['image_right'])
    mem_rel_df['new_image'] = np.where((((mem_rel_df['accuracy_thresh3']==1) & (mem_rel_df['response'].isin(['z','x'])))) | (((mem_rel_df['accuracy_thresh3']==0) & (mem_rel_df['response'].isin(['m','n'])))) , mem_rel_df['image_right'], mem_rel_df['image_left'])
   # mem_rel_df['old_image'] = np.where(((mem_rel_df['accuracy_thresh3']==0) & (mem_rel_df['response'].isin(['z','x']))), mem_rel_df['image_left'], mem_rel_df['image_right'])
    return mem_rel_df
    
def add_new_image(mem_irrel_df):
    # mem_rel_df['old_image'] = np.where((((mem_rel_df['accuracy_thresh3']==1) & (mem_rel_df['response'].isin(['z','x'])))) | (((mem_rel_df['accuracy_thresh3']==0) & (mem_rel_df['response'].isin(['m','n'])))) , mem_rel_df['image_left'], mem_rel_df['image_right'])
    mem_irrel_df['new_image'] = np.where((((mem_irrel_df['accuracy_thresh3']==1) & (mem_irrel_df['response'].isin(['z','x'])))) | (((mem_irrel_df['accuracy_thresh3']==0) & (mem_irrel_df['response'].isin(['m','n'])))) , mem_irrel_df['image_right'], mem_irrel_df['image_left'])
   # mem_rel_df['old_image'] = np.where(((mem_rel_df['accuracy_thresh3']==0) & (mem_rel_df['response'].isin(['z','x']))), mem_rel_df['image_left'], mem_rel_df['image_right'])
    return mem_irrel_df
    
def add_confident_memory(mem_df):
    mem_df['confident_memory'] = np.nan
    mem_df['confident_memory'].iloc[np.where((mem_df['accuracy_thresh3']==1) & (mem_df['response'].isin(['z','m'])))]= 'hit'
    mem_df['confident_memory'].iloc[np.where((mem_df['accuracy_thresh3']==0) & (mem_df['response'].isin(['z','m'])))] = 'FA'
    return mem_df


def calculate_lapses(DF):
    lapse_DF = DF[(DF['accuracy']==0) & (DF['correct_response']==0)]
    nLapses = len(lapse_DF)
    nInfrequent = len(DF[DF['correct_response']==0])
    prop_lapses = nLapses/nInfrequent
    return prop_lapses

def calculate_omissions(DF):
    miss_DF = DF[(DF['accuracy']==0) & (DF['correct_response']==1)]
    nMisses= len(miss_DF)
    nFrequent = len(DF[DF['correct_response']==1])
    prop_misses = nMisses/nFrequent
    return prop_misses



red_color = "#C30010"
blue_color = '#033A9C'
purple_color = '#6E00B3'
lightblue_color = "#676FB5"
lightred_color = "#FF474C"

colors_red = ["black","white", red_color ]
red_cmap = LinearSegmentedColormap.from_list("red_palette", colors_red, N=500)
red_palette = red_cmap(np.linspace(0, 1, 500))


colors_blue = ["black","white",blue_color]
blue_cmap = LinearSegmentedColormap.from_list("blue_palette", colors_blue, N=500)
blue_palette = blue_cmap(np.linspace(0, 1, 500))

colors_purple = ["black","white", purple_color ]
purple_cmap = LinearSegmentedColormap.from_list("purple_palette", colors_purple, N=500)
purple_palette = purple_cmap(np.linspace(0, 1, 500))


colors_lred = ["black","white", lightred_color ]
lightred_cmap = LinearSegmentedColormap.from_list("lightred_palette", colors_lred, N=500)
lightred_palette = lightred_cmap(np.linspace(0, 1, 500))


colors_lblue = ["black","white",lightblue_color]
lightblue_cmap = LinearSegmentedColormap.from_list("lightblue_palette", colors_lblue, N=500)
lightblue_palette = lightblue_cmap(np.linspace(0, 1, 500))







def calculate_BF(DF1, DF2):
    # chance=0.5
    # df = DF1
    # print(df.shape)
    # loop over timepoints, make decoding accuracy into effect size and convert to an r object
    n_timepoints = DF1.shape[1]
    # df_norm = pd.DataFrame(np.empty_like(df))
    # for t in range(n_timepoints):
    #     df_norm[t]=[(i - chance) for i in df[t]]

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data1 = ro.conversion.py2rpy(DF1)
        r_data2 = ro.conversion.py2rpy(DF2)

    bf=[]
    for t in range(n_timepoints):
        try:
            results=bf_package.ttestBF(x=r_data1[t],y=r_data2[t],mu=0,rscale='medium', paired=True)
            # print(r_data[t])
            bf.append(np.asarray(r['as.vector'](results))[0])
        except:
            bf.append(np.nan)
    return bf


# def plot_BF_subplot(times, bf, palette, ax, Naxis, title, topLim, bottomLim, xlabel=True):
#     # fig, ax = plt.subplots(figsize=(10,2))

#     bf_cols = sns.color_palette(palette, 500)
#     exponential_min = bottomLim
#     exponential_max = topLim
#     val_col_map = np.logspace(-(exponential_max-1),(exponential_max-1),num=500)

#     x = times
#     y = bf
#     markerline, stemlines, baseline = ax[Naxis].stem(x, y,bottom=1,linefmt='k', markerfmt=None, basefmt=None)

#     markerline.set_markerfacecolor('w')
#     markerline.set_markeredgecolor('w')
#     baseline.set_color('k')
#     stemlines.set_linewidth(0.5)

#     cols_idx = [np.argmin(np.abs(val_col_map-i)) for i in y]  
#     [ax[Naxis].plot(x[i],y[i],color=bf_cols[cols_idx[i]],marker='.',markersize=10,lw=0,markeredgecolor=None) for i in range(len(cols_idx))]
#     ax[Naxis].set_yscale('log')
#     ax[Naxis].set_ylim([10**-exponential_min,10**exponential_max])
#     ax[Naxis].set_yticks([10**-exponential_min,10**exponential_max])
#     ax[Naxis].axvline(0.0, color="k", linestyle="-")
#     # ax.axhline(1, color="k", linestyle="-")

#     ax[Naxis].spines['right'].set_visible(False)
#     ax[Naxis].spines['top'].set_visible(False)

#     if xlabel:
#         ax[Naxis].set_xlabel('time (s)',fontsize=font_sz)
#         ax[Naxis].get_xaxis().get_major_formatter().labelOnlyBase = True
#     else:
#         ax[Naxis].set_xticks([])

#     ax[Naxis].set_ylabel('BF (log)',fontsize=font_sz)
#     custom_lines = [Line2D([0], [0], color='red', lw=0)] # make an empty line to plot in legend
#     ax[Naxis].legend(custom_lines, {title}, frameon=False, loc='upper right', bbox_to_anchor=(1.0, 1.3))
#     # ax.legend()

#     plt.tight_layout()
#     # plt.savefig(figname)
#     # plt.show(), 


def plot_BF_subplot(times, bf, palette, ax, title, topLim, bottomLim, xlabel=True):
    # fig, ax = plt.subplots(figsize=(10,2))

    bf_cols = sns.color_palette(palette, 500)
    exponential_min = bottomLim
    exponential_max = topLim
    val_col_map = np.logspace(-(exponential_max),(exponential_max),num=500)

    x = times
    y = bf
    markerline, stemlines, baseline = ax.stem(x, y,bottom=1,linefmt='k', markerfmt=None, basefmt=None)

    markerline.set_markerfacecolor('w')
    markerline.set_markeredgecolor('w')
    baseline.set_color('k')
    stemlines.set_linewidth(0.5)

    cols_idx = [np.argmin(np.abs(val_col_map-i)) for i in y]  
    [ax.plot(x[i],y[i],color=bf_cols[cols_idx[i]],marker='.',markersize=10,lw=0,markeredgecolor=None) for i in range(len(cols_idx))]
    ax.set_yscale('log')
    ax.set_ylim([10**-exponential_min,10**exponential_max])
    ax.set_yticks([10**-exponential_min,10**exponential_max])
    ax.axvline(0.0, color="k", linestyle="-")
    # ax.axhline(1, color="k", linestyle="-")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if xlabel:
        ax.set_xlabel('time (s)',fontsize=font_sz)
        ax.get_xaxis().get_major_formatter().labelOnlyBase = True
    else:
        ax.set_xticks([])

    ax.set_ylabel('BF (log)',fontsize=font_sz)
    custom_lines = [Line2D([0], [0], color='red', lw=0)] # make an empty line to plot in legend
    ax.legend(custom_lines, {title}, frameon=False, loc='upper right', bbox_to_anchor=(1.0, 1.4))
    # ax.legend()

    plt.tight_layout()
    # plt.savefig(figname)
    # plt.show(), 




def calculate_COCE_zone(sub_run_df_pre, FWHM):
    sub_run_df_whole = sub_run_df_pre.copy()


    task_list = sub_run_df_whole.task
    # print(task_list)
    unique_ordered_list = list(dict.fromkeys(task_list))
    # print(unique_ordered_list)

    return_df = pd.DataFrame()
    for task in unique_ordered_list:
        # print(task)
        sub_run_df = sub_run_df_whole[sub_run_df_whole.task==task]
        # print(sub_run_df['accuracy'])

        lapse_inds = np.where((sub_run_df['correct_response']==0) & (sub_run_df['accuracy']==0))[0]
        # print(lapse_inds)
        pre_lapse_inds = np.array([*lapse_inds - 1, *lapse_inds - 2, *lapse_inds - 3])
        # print(pre_lapse_inds)

        CO_inds = np.where((sub_run_df['correct_response']==0) & (sub_run_df['accuracy']==1))[0]
        pre_CO_inds = np.array([*CO_inds - 1, *CO_inds - 2, *CO_inds - 3])

        # In case there were two infrequent images within a few trials of one another, remove any duplicates
        unique_lapse = list(set(pre_lapse_inds) - set(pre_CO_inds))
        unique_CO = list(set(pre_CO_inds) - set(pre_lapse_inds))

        # print(pre_lapse_inds)
        # print(unique_lapse)

        sub_run_df['COCE_zone'] = np.nan
        sub_run_df['COCE_zone'].iloc[unique_lapse] = 0
        sub_run_df['COCE_zone'].iloc[unique_CO] = 1
        # print()
        # for ind,row in sub_run_df:
        #     row['COCE_zone']
        # print(sub_run_df['COCE_zone'])
        sub_run_df.loc[sub_run_df['rt'] == 'nan', "rt"] = None  # set nonresponses to None for interpolation
        sub_run_df.loc[(sub_run_df['accuracy'] != 1), "rt"] = None  # set probe resps to None

        trials_idx_freqcorr  = np.where(~sub_run_df["rt"].isna())[0]
        # print(trials_idx_freqcorr)
        # print(sub_run_df["rt"])
        slope,intercept = np.polyfit(trials_idx_freqcorr,sub_run_df["rt"][~sub_run_df["rt"].isna()],1) #calculate linear trend
        sub_run_df["detrended_RT"] = sub_run_df["rt"]-(intercept+slope*np.arange(len(sub_run_df))) #subtract linear trend
        sub_run_df["median_detrended_RT"] = np.nanmedian(sub_run_df["detrended_RT"])
        sub_run_df["zRT_detrended"] = zscore(sub_run_df["detrended_RT"].values, nan_policy="omit")
        sub_run_df["abs_zRT_detrended"] = sub_run_df["zRT_detrended"].apply(np.abs)
        sub_run_df["abs_zRT_interpolated_detrended"] = sub_run_df["abs_zRT_detrended"].interpolate(method="linear", limit_direction="both")
        sub_run_df['vtc_detrended'] = sub_run_df["abs_zRT_interpolated_detrended"]


        sub_run_df["zRT"] = zscore(sub_run_df["rt"].values, nan_policy="omit")
        sub_run_df["abs_zRT"] = sub_run_df["zRT"].apply(np.abs)
        sub_run_df["abs_zRT_interpolated"] = sub_run_df["abs_zRT"].interpolate(method="linear", limit_direction="both")
        sub_run_df['vtc'] = sub_run_df["abs_zRT_interpolated"]

        # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        # FWHM = 2*sqrt(2*ln(2)) * sigma
        # -> sigma = FWHM / (2*sqrt(2*ln(2)))
        sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
        # TODO - modify filter to adjust weights at edges
        sub_run_df["vtc_smooth"] = gaussian_filter(sub_run_df["abs_zRT_interpolated"].values, sigma=sigma)
        sub_run_df["vtc_smooth_detrended"] = gaussian_filter(sub_run_df["abs_zRT_interpolated_detrended"].values, sigma=sigma)
        sub_run_df['vtc_median'] = np.median(sub_run_df['vtc'])
        sub_run_df['vtc_median_detrended'] = np.median(sub_run_df['vtc_detrended'])
        sub_run_df['vtc_smooth_median_detrended'] = np.median(sub_run_df['vtc_smooth_detrended'])

        # test analysis - calculate quartile
        sub_run_df['vtc_upper_quartile_detrended'] = np.quantile(sub_run_df['vtc_detrended'], 0.75)
        sub_run_df['vtc_lower_quartile_detrended'] = np.quantile(sub_run_df['vtc_detrended'], 0.25)
        sub_run_df['vtc_smooth_upper_quartile_detrended'] = np.quantile(sub_run_df['vtc_smooth_detrended'], 0.75)
        sub_run_df['vtc_smooth_lower_quartile_detrended'] = np.quantile(sub_run_df['vtc_smooth_detrended'], 0.25)

        print(np.nanquantile(sub_run_df['detrended_RT'], 0.75))
        sub_run_df['RT_upper_quartile'] = np.nanquantile(sub_run_df['detrended_RT'], 0.75)
        sub_run_df['RT_lower_quartile'] = np.nanquantile(sub_run_df['detrended_RT'], 0.25)

        sub_run_df['vtc_zone'] = np.where(sub_run_df['vtc_detrended']<sub_run_df['vtc_median_detrended'],1,0)

        

        return_df = pd.concat([return_df, sub_run_df], axis=0)

        # plt.plot(sub_run_df['vtc_smooth_detrended'])

    return return_df


import pingouin as pg



def make_timecourse_plot(plot_dict):


    DF = pd.DataFrame({plot_dict['bar1_label']: plot_dict['bar1'], plot_dict['bar2_label']: plot_dict['bar2']})
    DF_long = pd.melt(DF)
    print(pg.ttest(plot_dict['bar1'], plot_dict['bar2'], paired=True))
    print(plot_dict['bar1_label'])
    print(np.mean(plot_dict['bar1']))
    print(np.std(plot_dict['bar1']))
    print(plot_dict['bar2_label'])
    print(np.mean(plot_dict['bar2']))
    print(np.std(plot_dict['bar2']))

    fig = plt.figure(figsize=(13, 4))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[3, 1],
        height_ratios=[3, 1],
        wspace=0.1,
        hspace=0.2
    )

    # Left column
    ax_dec = fig.add_subplot(gs[0, 0])
    ax_bf = fig.add_subplot(gs[1, 0])

    # Right side of decoding plot
    ax_in_irrel = fig.add_subplot(gs[0, 1], sharey=ax_dec)
    # ax_out_irrel = fig.add_subplot(gs[0, 2], sharey=ax_dec)


    # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), gridspec_kw={'wspace': 0.4, 'hspace': 0.4, 'height_ratios': [3, 1]})
    ax_dec.plot(plot_dict['x_axis'], plot_dict['tc1'], plot_dict['tc1_color'], linestyle='-',label=plot_dict['tc1_label'])
    ax_dec.fill_between(plot_dict['x_axis'], plot_dict['tc1'] - plot_dict['tc1_SE'], plot_dict['tc1'] + plot_dict['tc1_SE'], color=plot_dict['tc1_color'], alpha=0.3)
    ax_dec.plot(plot_dict['x_axis'], plot_dict['tc2'], plot_dict['tc2_color'], linestyle=plot_dict['tc2_linestyle'],label=plot_dict['tc2_label'])
    ax_dec.fill_between(plot_dict['x_axis'], plot_dict['tc2'] - plot_dict['tc2_SE'], plot_dict['tc2'] + plot_dict['tc2_SE'], color=plot_dict['tc2_color'], alpha=0.3)
    ax_dec.set_ylim(0.45, 0.7)
    ax_dec.set_yticks([0.5, 0.6, 0.7])


    ax_dec.spines['right'].set_visible(False)
    ax_dec.spines['top'].set_visible(False)
    ax_dec.axhline(0.5, color="k", linestyle="--")
    ax_dec.set_ylabel("Area under\nthe curve", fontsize=font_sz)  # Area Under the Curve
    ax_dec.legend(frameon=False, framealpha=0)
    ax_dec.axvline(0.0, color="k", linestyle="-")
    # ax_dec.set_title("Irrelevant image category decoding", fontsize=font_sz)
    ax_dec.set_xticks([])

    pairs = [(plot_dict['bar1_label'], plot_dict['bar2_label'])]
    bf10 = float(pg.ttest(plot_dict['bar1'], plot_dict['bar2'], paired=True)['BF10'].values[0])
    print(bf10)
    if bf10 > 10000 or bf10 < 0.001:
        formatted_bf = r"$\mathrm{BF}_{10}$" + f" = {bf10:.2e}"
    else:
        formatted_bf = r"$\mathrm{BF}_{10}$" + f" = {bf10:.2f}"

    custom_annotations = [formatted_bf]
    print(custom_annotations)

    annotator = Annotator(ax_in_irrel, pairs, data=DF_long, x="variable", y="value")
    annotator.configure(loc='outside', fontsize=font_sz)
    annotator.set_custom_annotations(custom_annotations)
    annotator.annotate()

    sns.boxplot(x="variable", y="value", data=DF_long, hue="variable", palette = [plot_dict['tc1_color'], plot_dict['tc2_color']], linewidth=2, fliersize=0,  ax=ax_in_irrel)
    sns.stripplot(x="variable", y="value", data=DF_long, color="black", alpha=0.5, ax=ax_in_irrel)
    # ax_in_irrel.scatter(0,np.mean(in_irrel_mean), color='white')
    ax_in_irrel.axhline(0.5, color="k", linestyle="--")
    ax_in_irrel.set_xlabel("")
    ax_in_irrel.set_ylabel("")
    sns.despine(right=True)

    # ax_out_irrel.tick_params(axis='y', left=False, labelleft=False)

    plot_BF_subplot(plot_dict['x_axis'], plot_dict['BF'], plot_dict['BF_palette'], ax=ax_bf, title=plot_dict['BF_title'], topLim=plot_dict['BF_top'], bottomLim=2, xlabel=True)
    # plt.show()
    plt.savefig(plot_dict['save_name'], bbox_inches='tight')
    plt.show()








def make_Gabor_timecourse_plot(plot_dict):


    DF = pd.DataFrame({plot_dict['bar1_label']: plot_dict['bar1'], plot_dict['bar2_label']: plot_dict['bar2']})
    DF_long = pd.melt(DF)
    print(pg.ttest(plot_dict['bar1'], plot_dict['bar2'], paired=True))
    print(plot_dict['bar1_label'])
    print(np.mean(plot_dict['bar1']))
    print(np.std(plot_dict['bar1']))
    print(plot_dict['bar2_label'])
    print(np.mean(plot_dict['bar2']))
    print(np.std(plot_dict['bar2']))

    fig = plt.figure(figsize=(13, 4))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[3, 1],
        height_ratios=[3, 1],
        wspace=0.1,
        hspace=0.2
    )

    # Left column
    ax_dec = fig.add_subplot(gs[0, 0])
    ax_bf = fig.add_subplot(gs[1, 0])

    # Right side of decoding plot
    ax_in_irrel = fig.add_subplot(gs[0, 1], sharey=ax_dec)
    # ax_out_irrel = fig.add_subplot(gs[0, 2], sharey=ax_dec)


    # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), gridspec_kw={'wspace': 0.4, 'hspace': 0.4, 'height_ratios': [3, 1]})
    ax_dec.plot(plot_dict['x_axis'], plot_dict['tc1'], plot_dict['tc1_color'], linestyle='-',label=plot_dict['tc1_label'])
    ax_dec.fill_between(plot_dict['x_axis'], plot_dict['tc1'] - plot_dict['tc1_SE'], plot_dict['tc1'] + plot_dict['tc1_SE'], color=plot_dict['tc1_color'], alpha=0.3)
    ax_dec.plot(plot_dict['x_axis'], plot_dict['tc2'], plot_dict['tc2_color'], linestyle=plot_dict['tc2_linestyle'],label=plot_dict['tc2_label'])
    ax_dec.fill_between(plot_dict['x_axis'], plot_dict['tc2'] - plot_dict['tc2_SE'], plot_dict['tc2'] + plot_dict['tc2_SE'], color=plot_dict['tc2_color'], alpha=0.3)
    ax_dec.set_ylim(0.4, 0.7)
    ax_dec.set_yticks([0.5, 0.6, 0.7])


    ax_dec.spines['right'].set_visible(False)
    ax_dec.spines['top'].set_visible(False)
    ax_dec.axhline(0.5, color="k", linestyle="--")
    ax_dec.set_ylabel("Area under\nthe curve", fontsize=font_sz)  # Area Under the Curve
    ax_dec.legend(frameon=False, framealpha=0)
    ax_dec.axvline(0.0, color="k", linestyle="-")
    # ax_dec.set_title("Irrelevant image category decoding", fontsize=font_sz)
    ax_dec.set_xticks([])

    pairs = [(plot_dict['bar1_label'], plot_dict['bar2_label'])]
    bf10 = float(pg.ttest(plot_dict['bar1'], plot_dict['bar2'], paired=True)['BF10'].values[0])
    print(bf10)
    if bf10 > 10000 or bf10 < 0.001:
        formatted_bf = r"$\mathrm{BF}_{10}$" + f" = {bf10:.2e}"
    else:
        formatted_bf = r"$\mathrm{BF}_{10}$" + f" = {bf10:.2f}"

    custom_annotations = [formatted_bf]
    print(custom_annotations)

    annotator = Annotator(ax_in_irrel, pairs, data=DF_long, x="variable", y="value")
    annotator.configure(loc='outside', fontsize=font_sz)
    annotator.set_custom_annotations(custom_annotations)
    annotator.annotate()

    sns.boxplot(x="variable", y="value", data=DF_long, hue="variable", palette = [plot_dict['tc1_color'], plot_dict['tc2_color']], linewidth=2, fliersize=0,  ax=ax_in_irrel)
    sns.stripplot(x="variable", y="value", data=DF_long, color="black", alpha=0.5, ax=ax_in_irrel)
    # ax_in_irrel.scatter(0,np.mean(in_irrel_mean), color='white')
    ax_in_irrel.axhline(0.5, color="k", linestyle="--")
    ax_in_irrel.set_xlabel("")
    ax_in_irrel.set_ylabel("")
    sns.despine(right=True)

    # ax_out_irrel.tick_params(axis='y', left=False, labelleft=False)

    plot_BF_subplot(plot_dict['x_axis'], plot_dict['BF'], plot_dict['BF_palette'], ax=ax_bf, title=plot_dict['BF_title'], topLim=plot_dict['BF_top'], bottomLim=2, xlabel=True)
    # plt.show()
    plt.savefig(plot_dict['save_name'], bbox_inches='tight')
    plt.show()








def make_EyeCorr_timecourse_plot(plot_dict):


    DF = pd.DataFrame({plot_dict['bar1_label']: plot_dict['bar1'], plot_dict['bar2_label']: plot_dict['bar2']})
    DF_long = pd.melt(DF)
    print(pg.ttest(plot_dict['bar1'], plot_dict['bar2'], paired=True))
    print(plot_dict['bar1_label'])
    print(np.mean(plot_dict['bar1']))
    print(np.std(plot_dict['bar1']))
    print(plot_dict['bar2_label'])
    print(np.mean(plot_dict['bar2']))
    print(np.std(plot_dict['bar2']))

    fig = plt.figure(figsize=(13, 4))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[3, 1],
        height_ratios=[3, 1],
        wspace=0.2,
        hspace=0.3
    )

    # Left column
    ax_dec = fig.add_subplot(gs[0, 0])
    ax_bf = fig.add_subplot(gs[1, 0])

    # Right side of decoding plot
    ax_in_irrel = fig.add_subplot(gs[0, 1], sharey=ax_dec)
    # ax_out_irrel = fig.add_subplot(gs[0, 2], sharey=ax_dec)


    # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), gridspec_kw={'wspace': 0.4, 'hspace': 0.4, 'height_ratios': [3, 1]})
    ax_dec.plot(plot_dict['x_axis'], plot_dict['tc1'], plot_dict['tc1_color'], linestyle='-',label=plot_dict['tc1_label'])
    ax_dec.fill_between(plot_dict['x_axis'], plot_dict['tc1'] - plot_dict['tc1_SE'], plot_dict['tc1'] + plot_dict['tc1_SE'], color=plot_dict['tc1_color'], alpha=0.3)
    ax_dec.plot(plot_dict['x_axis'], plot_dict['tc2'], plot_dict['tc2_color'], linestyle=plot_dict['tc2_linestyle'],label=plot_dict['tc2_label'])
    ax_dec.fill_between(plot_dict['x_axis'], plot_dict['tc2'] - plot_dict['tc2_SE'], plot_dict['tc2'] + plot_dict['tc2_SE'], color=plot_dict['tc2_color'], alpha=0.3)
    ax_dec.set_ylim(-0.1, 0.3)
    ax_dec.set_yticks([-0.1, 0, 0.1, 0.2])


    ax_dec.spines['right'].set_visible(False)
    ax_dec.spines['top'].set_visible(False)
    ax_dec.axhline(0, color="k", linestyle="--")
    ax_dec.set_ylabel("Correlation (r)\nEEG, eye predictions", fontsize=font_sz)  # Area Under the Curve
    ax_dec.legend(frameon=False, framealpha=0)
    ax_dec.axvline(0.0, color="k", linestyle="-")
    # ax_dec.set_title("Irrelevant image category decoding", fontsize=font_sz)
    ax_dec.set_xticks([])

    pairs = [(plot_dict['bar1_label'], plot_dict['bar2_label'])]
    bf10 = float(pg.ttest(plot_dict['bar1'], plot_dict['bar2'], paired=True)['BF10'].values[0])
    print(bf10)
    if bf10 > 10000 or bf10 < 0.001:
        formatted_bf = r"$\mathrm{BF}_{10}$" + f" = {bf10:.2e}"
    else:
        formatted_bf = r"$\mathrm{BF}_{10}$" + f" = {bf10:.2f}"

    custom_annotations = [formatted_bf]
    print(custom_annotations)

    annotator = Annotator(ax_in_irrel, pairs, data=DF_long, x="variable", y="value")
    annotator.configure(loc='outside', fontsize=font_sz)
    annotator.set_custom_annotations(custom_annotations)
    annotator.annotate()

    sns.boxplot(x="variable", y="value", data=DF_long, hue="variable", palette = [plot_dict['tc1_color'], plot_dict['tc2_color']], linewidth=2, fliersize=0,  ax=ax_in_irrel)
    sns.stripplot(x="variable", y="value", data=DF_long, color="black", alpha=0.5, ax=ax_in_irrel)
    # ax_in_irrel.scatter(0,np.mean(in_irrel_mean), color='white')
    ax_in_irrel.axhline(0, color="k", linestyle="--")
    ax_in_irrel.set_xlabel("")
    ax_in_irrel.set_ylabel("")
    sns.despine(right=True)

    # ax_out_irrel.tick_params(axis='y', left=False, labelleft=False)

    plot_BF_subplot(plot_dict['x_axis'], plot_dict['BF'], plot_dict['BF_palette'], ax=ax_bf, title=plot_dict['BF_title'], topLim=plot_dict['BF_top'], bottomLim=2, xlabel=True)
    # plt.show()
    plt.savefig(plot_dict['save_name'], bbox_inches='tight')
    plt.show()


