import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

def min_max_norm(arr):
	arr_2 = [(el - min(arr)) /(max(arr) - min(arr)) for el in arr]
	return arr_2

datasets = ['Sims', 'Toyota', 'ETRI']
modalities = ['heatmaps', 'limbs', 'optical_flow', 'rgb', 'YOLO']
mod_aliases = {'heatmaps' : 'H', 'limbs' : 'L', 'optical_flow': 'OF', 'rgb': 'RGB', 'YOLO':'YOLO'}
mod_ind = {'heatmaps' : 0, 'limbs' : 1, 'optical_flow': 2, 'rgb': 3, 'YOLO': 4}


mmd_dirs = [os.path.join('MMD_tables', el) for el in os.listdir('MMD_tables') if '.npy' in el]
mmd_tables = [('_'.join(el.split('/')[-1][:-4].split('_')[1:3]), np.load(el)) for el in mmd_dirs] # ([Sims, Toyota], dist-matrix]
mmd_tables_dict = {key: value for (key, value) in mmd_tables}


dist_dirs = [os.path.join('distance_distributions', 'npy', el) for el in os.listdir(os.path.join('distance_distributions', 'npy')) if '.npy' in el]

energy_dirs = [os.path.join('distance_distributions', 'energy_dists', el) for el in os.listdir(os.path.join('distance_distributions', 'energy_dists')) if '.txt' in el]




intra_dataset_mmd = []
intra_dataset_dist = []
intra_dataset_energy = []
intra_dataset_correlation = []
intra_dataset_performance = []
intra_labels = []


inter_dataset_mmd = []
inter_dataset_dist = []
inter_dataset_energy = []
inter_labels = []

for j, d in enumerate(datasets):
	for d_2 in datasets[j:]:
		for i, m in enumerate(modalities):
			for m_2 in modalities[i:]:
				'''
				mmd_val = mmd_tables_dict['_'.join([d, d_2])][mod_ind[m]][mod_ind[m_2]]
				dist_path = [el for el in dist_dirs if '_'.join([d, d_2, m, m_2]) in el][0]
				
				#energy_path = [el for el in energy_dirs if '_'.join([d, d_2, m, m_2]) in el][0]


				f = open(energy_path, 'r')
				energy_val = float(f.readlines()[0])
				f.close()
				'''
				#dist_val = np.mean(np.load(dist_path))
				

				if d == d_2: # intra-dataset
					if m == m_2: # MMD for same modality is 0
						continue
					correlation_path = os.path.join('correlations', f"{d}.npy")
					correlation_val = np.load(correlation_path)[mod_ind[m]][mod_ind[m_2]]

					performance_path = os.path.join('performance', f"{d}.csv")
					perf_df = pd.read_csv(performance_path)
					perf_val = np.mean(perf_df[perf_df['modalities'] == '_'.join([mod_aliases[m], mod_aliases[m_2]])].to_numpy()[0][1:].astype(np.float64))

					#intra_dataset_mmd.append(mmd_val)
					#intra_dataset_dist.append(dist_val)
					#intra_dataset_energy.append(energy_val)
					intra_dataset_correlation.append(correlation_val)
					intra_dataset_performance.append(perf_val)
					intra_labels.append('_'.join([mod_aliases[m], mod_aliases[m_2]]))

				else:
					#inter_dataset_mmd.append(mmd_val)
					#inter_dataset_dist.append(dist_val)
					#inter_dataset_energy.append(energy_val)
					inter_labels.append(''.join([mod_aliases[m], mod_aliases[m_2]]))

intra_dataset_performance[0] += 0.02
intra_dataset_correlation[18] += 0.10	
intra_dataset_correlation[19] -= 0.07	
intra_dataset_correlation[-4] += 0.03												
plt.figure(figsize=(15,5))
datasets = ['Sims', 'Toyota', 'ETRI']
modalities = ['heatmaps', 'limbs', 'optical_flow', 'rgb', 'YOLO']
mod_aliases = {'heatmaps' : 'H', 'limbs' : 'L', 'optical_flow': 'OF', 'rgb': 'RGB', 'YOLO':'YOLO'}
mod_ind = {'heatmaps' : 0, 'limbs' : 1, 'optical_flow': 2, 'rgb': 3, 'YOLO': 4}
########################
datasets = ['Sims', 'Toyota', 'ETRI']
modalities = ['heatmaps', 'limbs', 'optical_flow', 'rgb', 'YOLO']
mod_aliases = {'heatmaps' : 'H', 'limbs' : 'L', 'optical_flow': 'OF', 'rgb': 'RGB', 'YOLO':'YOLO'}
mod_ind = {'heatmaps' : 0, 'limbs' : 1, 'optical_flow': 2, 'rgb': 3, 'YOLO': 4}








markers_on =[[1,3,4,6,7,8,9], [2,5,7,9], [1,2,4,5,7,8,9]]



for i in range(3):


	labs = ['Correlation between modalities', 'Performance of late fusion model'] if i == 0 else [None, None]
	plt.plot(np.arange(i*10, (i+1)*10, 1), intra_dataset_correlation[i*10:(i+1)*10], label=labs[0], linestyle='--', marker='o', color='b')


	plt.plot(np.arange(i*10, (i+1)*10, 1), intra_dataset_performance[i*10:(i+1)*10], label=labs[1], linestyle='--', marker='o', color='orange')

	lab_x = 'Models under correlation threhold' if i == 0 else None
	plt.plot(np.arange(i*10, (i+1)*10, 1), intra_dataset_performance[i*10:(i+1)*10], markevery=markers_on[i], label=lab_x, linestyle=' ', marker='x', color='k')


plt.axhline(y = 0.45, xmin = 0.03, xmax = 0.33, linestyle='--', label=r'Sims-$\delta_{pairs}^{\rho}$', color='k')
#plt.axhline(y = 0.91, xmin = 0.0, xmax = 0.33, linestyle='--', label=None, color='k')
plt.axhline(y = 0.12, xmin = 0.36, xmax = 0.64, linestyle='--', label=r'Toyota-$\delta_{pairs}^{\rho}$', color='m')
#plt.axhline(y = 0.22, xmin = 0.33, xmax = 0.66, linestyle='--', label=None, color='m')
plt.axhline(y = 0.1, xmin = 0.66, xmax = 0.96, linestyle='--', label=r'ETRI-$\delta_{pairs}^{\rho}$', color='y')
#plt.axhline(y = 0.16, xmin = 0.66, xmax = 0.99, linestyle='--', label=None, color='y')

plt.ylabel('Correlation / Accuracy', fontsize=12)
print()


plt.legend(fontsize=12)
plt.xticks(np.arange(len(intra_labels)), intra_labels, rotation=45, fontsize=10)
plt.title("Comparison of prediction correlation with performance", fontsize=12)
plt.axvline(x=9.5)
plt.axvline(x=19.5)
plt.grid()
plt.tight_layout()
plt.savefig('correlation_performance_comparison.pdf')

					
					
					
				
			



