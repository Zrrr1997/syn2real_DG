import numpy as np
import os
from matplotlib import pyplot as plt

datasets = ['Sims', 'ETRI', 'Toyota']
modalities = ['heatmaps', 'limbs', 'optical_flow', 'rgb']
mod_aliases = {'heatmaps' : 'H', 'limbs' : 'L', 'optical_flow': 'OF', 'rgb': 'RGB'}
mod_ind = {'heatmaps' : 0, 'limbs' : 1, 'optical_flow': 2, 'rgb': 3}


mmd_dirs = [os.path.join('MMD_tables', el) for el in os.listdir('MMD_tables') if '.npy' in el]
mmd_tables = [('_'.join(el.split('/')[-1][:-4].split('_')[1:3]), np.load(el)) for el in mmd_dirs] # ([Sims, Toyota], dist-matrix]
mmd_tables_dict = {key: value for (key, value) in mmd_tables}


dist_dirs = [os.path.join('distance_distributions', 'npy', el) for el in os.listdir(os.path.join('distance_distributions', 'npy')) if '.npy' in el]

energy_dirs = [os.path.join('distance_distributions', 'energy_dists', el) for el in os.listdir(os.path.join('distance_distributions', 'energy_dists')) if '.txt' in el]




intra_dataset_mmd = []
intra_dataset_dist = []
intra_dataset_energy = []
intra_labels = []
inter_dataset_mmd = []
inter_dataset_dist = []
inter_dataset_energy = []
inter_labels = []

for j, d in enumerate(datasets):
	for d_2 in datasets[j:]:
		for i, m in enumerate(modalities):
			for m_2 in modalities[i:]:
				mmd_val = mmd_tables_dict['_'.join([d, d_2])][mod_ind[m]][mod_ind[m_2]]
				dist_path = [el for el in dist_dirs if '_'.join([d, d_2, m, m_2]) in el][0]
				energy_path = [el for el in energy_dirs if '_'.join([d, d_2, m, m_2]) in el][0]

				f = open(energy_path, 'r')
				energy_val = float(f.readlines()[0])
				f.close()
				dist_val = np.mean(np.load(dist_path))

				if d == d_2: # intra-dataset
					if m == m_2: # MMD for same modality is 0
						continue
					intra_dataset_mmd.append(mmd_val)
					intra_dataset_dist.append(dist_val)
					intra_dataset_energy.append(energy_val)
					intra_labels.append(''.join([mod_aliases[m], mod_aliases[m_2]]))

				else:
					inter_dataset_mmd.append(mmd_val)
					inter_dataset_dist.append(dist_val)
					inter_dataset_energy.append(energy_val)
					inter_labels.append(''.join([mod_aliases[m], mod_aliases[m_2]]))

					
plt.figure(figsize=(20,5))
plt.plot(intra_dataset_mmd + inter_dataset_mmd, label='Maximum Mean Discrepancy (MMD)', linestyle='--')
plt.plot(intra_dataset_dist+ inter_dataset_dist, label='Mean Pairwise Distance (MPD)', linestyle='--')
plt.plot(intra_dataset_energy+ inter_dataset_energy, label='Energy Distance', linestyle='--')
plt.axvline(x=6)
plt.axvline(x=12)
plt.axvline(x=18)
plt.axvline(x=28)
plt.axvline(x=38)

plt.xticks(np.arange(len(intra_labels + inter_labels)) * 1, intra_labels + inter_labels, rotation=45, fontsize=6)
plt.legend()

plt.savefig('mmd_dist_metric_comparison.svg')
					
					
					
				
			



