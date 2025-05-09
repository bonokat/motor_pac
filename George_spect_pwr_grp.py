# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:11:53 2024

@author: a1147969
"""

import os
import numpy as np
import mne
from mne.stats import permutation_cluster_1samp_test as perm_test
import pandas as pd
import pickle
import itertools
import scipy
from functools import partial

import matplotlib.pyplot as plt


from specparam import SpectralGroupModel
from specparam.bands import Bands
from specparam.analysis import get_band_peak_group
from specparam.plts.spectra import plot_spectra

from params import ID, cond, time, mne_datIn, srate, grp_dat_path, mPath
from plot import spect_plot, spect_check

zero_policy = 'exc'

os.chdir(f'{grp_dat_path}')

# load pos file (needed for adjacency matrix and topoplots)
pos = pickle.load(open(f'{mPath}/working/pos.pkl', 'rb'))

#check if stats outputs already exist, load if so
if os.path.isfile(f'{grp_dat_path}/spect_stats_out.pkl') and os.path.isfile(f'{grp_dat_path}/spect_stats_dat.pkl'):
    stats_dat = pickle.load(open(f'{grp_dat_path}/spect_stats_dat.pkl', 'rb'))
    stats_out = pickle.load(open(f'{grp_dat_path}/spect_stats_out.pkl', 'rb'))

#otherwise format data and run stats
else:    
    # load individual df's
    grp_spect = pickle.load(open('grp_pwr_spectd.pkl', 'rb'))
    
    # group into np array, removing labels, address 0's (from poor fits/no peak) based on policy
    ind_dat = [grp_spect[i].iloc[:,:62].to_numpy(dtype=float) for i in range(len(grp_spect))]
    ind_dat = np.stack(ind_dat, axis=-1)
    
    if zero_policy == 'inc':
        avg_dat = np.mean(ind_dat, axis=2)
    elif zero_policy == 'exc':
        ind_dat[ind_dat == 0] = np.nan
        avg_dat = np.nanmean(ind_dat, axis=2)
        
    avg_df = grp_spect[0].copy(deep=True)
    
    avg_df.iloc[:,:62] = avg_dat[:]
    
    # extract data sets for stats (subj x elec, per time/band/condition), after checking/replacing missing elecs
    # participants with too many missing are excluded
    stats_dat = {}
    
    #find indices of PRE values
    pre_val_ind = [i for i,v in enumerate(avg_df['time'] == 'PRE') if v]
    
    #find subj to reject, based on baseline missing data (>40% missing)
    sub_rej = {}
    for x in pre_val_ind:
        tmp_dat = ind_dat[x,:,:].T    
        rej = []
        
        for sub in range(len(tmp_dat)):
            
            n_nan = np.count_nonzero(np.isnan(tmp_dat[sub,:]))
            
            if x in pre_val_ind and 1-(n_nan/62) < 0.6:
                rej.append(sub)
        
        sub_rej[f'{avg_df.iloc[x, 62]}_{avg_df.iloc[x, 64]}'] = rej
        
    stats_dat['subRej'] = sub_rej
    
    #deal with missing data
    for x in range(ind_dat.shape[0]):
        
        tmp_dat = ind_dat[x,:,:].T
        del_ind = []
        
        for sub in range(len(tmp_dat)):
            
            n_nan = np.count_nonzero(np.isnan(tmp_dat[sub,:]))        
    
            #replace nans with avg for cond/time point
            if n_nan > 0: 
                nan_loc = np.where(np.isnan(tmp_dat[sub,:]))
                
                tmp_dat[sub, nan_loc] = avg_dat[x, nan_loc]
                
        tmp_dat = np.delete(tmp_dat, sub_rej[f'{avg_df.iloc[x, 62]}_{avg_df.iloc[x, 64]}'], axis=0)
        
        cond_n = avg_df.iloc[x, 62] + '_' + avg_df.iloc[x, 63] + '_' + avg_df.iloc[x, 64]
        
        stats_dat[cond_n] = tmp_dat
        
    
    #run cluster-tests    
    adj_matrix = mne.channels.find_ch_adjacency(pos, 'eeg')[0]
    
    #for stats output
    stats_out = {}
    
    for cnd in cond:
        for bnd in ('alpha', 'l-beta', 'u-beta'):
            
            # skip data if less than 6 participants
            if len(stats_dat[f'{bnd}_PRE_{cnd}']) < 6 or len(stats_dat[f'{bnd}_POS_{cnd}']) < 6:
                continue
            # otherwise run cluster stats
            else:        
                #get input data
                x1 = stats_dat[f'{bnd}_PRE_{cnd}']
                x2 = stats_dat[f'{bnd}_POS_{cnd}']
                
                X = x1 - x2
                
                #set threshold
                pval = 0.005  # arbitrary
                df = len(x1) - 1  # degrees of freedom for the test
                thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution
                
                #for non-paired
                # pval = 0.025  # arbitrary
                # dfn = len(time) - 1  # degrees of freedom numerator (conditions - 1)
                # dfd = len(x1) - len(time)  # degrees of freedom denominator (observations - conditions)
                # thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution
                
                #run test
                t, cl, p, h0 = perm_test(X, #[x1, x2] for non-paired
                                         threshold = thresh,
                                         adjacency = adj_matrix,                                     
                                         n_permutations = 10000,
                                         out_type = 'mask',
                                         verbose=True)
                
                stats_out[f'{bnd}_{cnd}'] = dict(tval = t,
                                                 clusters = cl,
                                                 pval = p,
                                                 maxstat = h0)
    
    if ~os.path.isfile(f'{grp_dat_path}/stats_out.pkl') or ~os.path.isfile(f'{grp_dat_path}/stats_dat.pkl'):
        #save stats
        pickle.dump(stats_out, open(f'{grp_dat_path}/spect_stats_out.pkl', 'wb'))
        pickle.dump(stats_dat, open(f'{grp_dat_path}/spect_stats_dat.pkl', 'wb'))

###---- plots ----###
#Define bands for topoplots
bands = Bands({'alpha': [7, 14],
               'l-beta': [15, 20],
               'u-beta': [21, 30]}) 

#get data in right format for plot function (df with cond, band, time labels)
temp = []
for bnd_id, (bnd, _ ) in enumerate(bands):
    for tm in time:
        for cnd in cond:
            temp.append(np.hstack((np.mean(stats_dat[f'{bnd}_{tm}_{cnd}'], axis=0), bnd, tm, cnd)))

labels = pos.ch_names[:]
[labels.append(x) for x in ('band', 'time', 'cond')]
plt_df = pd.DataFrame(temp, columns=labels)

#setup figure
fig = plt.figure(figsize=(8, 5), dpi=600)
fig.set_layout_engine('constrained')
subfigs = fig.subfigures(1,2, width_ratios=[0.05, 0.95])

#get masks in list
mask = {}
for cnd in cond:
    for bnd in ('alpha', 'l-beta', 'u-beta'):
        
        if stats_out[f'{bnd}_{cnd}']['clusters']:
            sig_clust = []
            
            for n, clust_p in enumerate(stats_out[f'{bnd}_{cnd}']['pval']):
                if clust_p < 0.01:
                    sig_clust.append(stats_out[f'{bnd}_{cnd}']['clusters'][n])
                    
            if sig_clust:
                sig_clust = np.array(sig_clust)
                sig_clust = sig_clust.max(axis=0)
        
                mask[f'{bnd}_{cnd}'] = sig_clust

#plot topographies
avg_plot = spect_plot(bands, plt_df, pos, fig=subfigs[1], mask=mask)

ax = subfigs[0].subplots()
ax.set_axis_off()

# for lb in list(zip(('Beta', 'Alpha'), (0.13, 0.53))):
#     for y in (0.2, 0.6):
#         ax.annotate(lb[0], (0.5, lb[1]), rotation='vertical', fontsize=20)
        
fig.savefig(f'{mPath}/figures/grp_spect_dat.jpg', dpi=600)




