import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
import argparse
from scipy.spatial import distance_matrix


parser = argparse.ArgumentParser(description='Maximum Mean Discrepancy (MMD) Matrix.')
parser.add_argument('--datasets', type=str, nargs='+', choices=['Sims', 'ETRI', 'Toyota'],
                    help='Datasets to compute the MMD matrix.')
args = parser.parse_args()

datasets = args.datasets
modalities = ['heatmaps', 'limbs', 'optical_flow', 'rgb'] # YOLO has a different dimensionality
mods = ['H', 'L', 'OF', 'RGB']
aliases = {'heatmaps': 'H', 'limbs': 'L', 'optical_flow': 'OF', 'rgb': 'RGB'} # YOLO has a different dimensionality
mean_vecs_d1 = []
mean_vecs_d2 = []
mean_vecs = [mean_vecs_d1, mean_vecs_d2]
for i, d in enumerate(datasets):
	for m in modalities:
		mean_vecs[i].append(np.load(f"{d}/{m}/mean_vec.npy"))
mean_vecs = [np.array(el) for el in mean_vecs]

mmd = distance_matrix(mean_vecs[0], mean_vecs[1])
data = {}
for i, m in enumerate(mods):
	data[m] = mmd[i]

mmd_matrix = pd.DataFrame(data = data, index = mods, columns = mods)

sn.heatmap(mmd_matrix, annot=True, annot_kws={"size": 26}, cbar=False,square=True)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
d = 'Sims'
if datasets[0] == 'Toyota':
	d = 'Toyota Smarthome'
elif datasets[0] == 'ETRI':
	d = 'ETRI'
plt.title(f"{d}", fontsize=26)

#plt.ylabel(args.datasets[0])
#plt.xlabel(args.datasets[1])
plt.tight_layout()
np.save(f"MMD_tables/mmd_{args.datasets[0]}_{args.datasets[1]}.npy", mmd)
plt.savefig(f'MMD_tables/mmd_{args.datasets[0]}_{args.datasets[1]}.svg')
plt.savefig(f'MMD_tables/mmd_{args.datasets[0]}_{args.datasets[1]}.pdf')
plt.savefig(f'MMD_tables/pngs/mmd_{args.datasets[0]}_{args.datasets[1]}.png')


