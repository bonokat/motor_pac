import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from utils import check_paths
from mne.channels.layout import find_layout
import scipy.stats
from mne.stats import permutation_cluster_1samp_test, spatio_temporal_cluster_1samp_test, permutation_cluster_test


eeg_data_dir = 'D:\\BonoKat\\research project\\# study 1\\eeg_data\\set'
group = 'Y'
subs = os.listdir(os.path.join(eeg_data_dir, group))

task = '_MAIN' # ['_BL', '_MAIN']
task_stage = '_plan' # '_plan' or '_go'
block_name = '_baseline' # ['_baseline', '_adaptation']

group_save_path = os.path.join(eeg_data_dir, f'{group} group')
pac_stats_save_path = os.path.join(group_save_path, 'pac_stats')
check_paths(pac_stats_save_path)




pac_list = []
for sub_name in subs: # [subs[0]]
    sub_dir = os.path.join(eeg_data_dir, group, sub_name)
    pac_dir = os.path.join(sub_dir, 'pac_results')

    if sub_name == subs[0]: # read one epochs file to extract info
        # Load EEG data
        epochs_path = os.path.join(eeg_data_dir, group, sub_name, 'preproc', 'analysis') 
        epochs = mne.read_epochs(os.path.join(epochs_path, f"{sub_name}{task}_epochs{task_stage}{block_name}-epo.fif"), preload=True)
        eeg_channel_names = epochs.copy().pick("eeg").ch_names
        epochs.pick(eeg_channel_names)
        # info = epochs.info
        # epochs.pick(choi)

    # Load PAC data
    pac = np.load(os.path.join(pac_dir, f"pac_mi_TOPO_{sub_name[-5:]}{task}{task_stage}{block_name}.npy"))
    pac_t = np.transpose(pac, (1, 0, 2))
    pac_list.append(pac_t)
    
# Stack them along a new first axis (subject axis)
pac_all = np.stack(pac_list, axis=0)
print(pac_all.shape) # (24, 30, 20, 20) subs x electrodes x ph_freqs x amp_freqs


# find_ch_adjacency first attempts to find an existing "neighbor"
# (adjacency) file for given sensor layout.
# If such a file doesn't exist, an adjacency matrix is computed on the fly,
# using Delaunay triangulations.
sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(epochs.info, "eeg")
# print(sensor_adjacency)
# print(ch_names)

adjacency = mne.stats.combine_adjacency(
    sensor_adjacency, pac_all.shape[2], pac_all.shape[3]
)


# We want a two-tailed test
tail = 1

# In this example, we wish to set the threshold for including data bins in
# the cluster forming process to the t-value corresponding to p=0.05 for the
# given data.
#
# Because we conduct a one-tailed test, we  DON'T divide the p-value by 2
# As the degrees of freedom, we specify the number of observations
# (here: participants) minus 1.
# Finally, we subtract 0.05 from 1, to get the critical t-value
# on the right tail (this is needed for MNE-Python internals)
degrees_of_freedom = pac_all.shape[0] - 1
t_thresh = scipy.stats.t.ppf(1 - 0.01, df=degrees_of_freedom)

# Set the number of permutations to run.
n_permutations = 1000

# Run the analysis
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
    pac_all,
    n_permutations=n_permutations,
    threshold=t_thresh,
    tail=tail,
    adjacency=adjacency,
    out_type="mask",
    max_step=1,
    verbose=True,
)



alpha = 0.05  # significance threshold
significant_clusters = [i for i, p in enumerate(cluster_p_values) if p < alpha]
print(f"Found {len(significant_clusters)} significant clusters")