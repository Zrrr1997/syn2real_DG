import numpy as np
import argparse
import torch
from matplotlib import pyplot as plt
import seaborn as sns

import dcor



def energy_dist(datasets=None, modalities=None):
	embs = []
	for (d, m) in zip(datasets, modalities):
		emb_alias = 'results_test_YOLO_detections_only_frac_YOLO_mlp_32f_embeddings.npy' if m == 'YOLO' else 'results_test_sims_video_multimodal_frac_s3d_32f_embeddings.npy'
		emb = np.load(f"{d}/{m}/{emb_alias}")
		emb = emb.reshape((-1, emb.shape[2]))
		embs.append(emb)
	print(embs[0].shape, embs[1].shape)
	min_shape = min(embs[0].shape[0], embs[1].shape[0])
	print(min_shape)

	
	X = torch.from_numpy(embs[0][:1000])
	Y = torch.from_numpy(embs[1][:1000])



	#energy_dist = Loss(X, Y).item()
	energy_dist = dcor.energy_distance(X, Y) 
	#print(energy_dist)


	#dist_matrix = distance_matrix(embs[0][:1000], embs[1][:1000])
	return energy_dist

parser = argparse.ArgumentParser(description='Distance distribution computation')
parser.add_argument('--datasets', type=str, nargs='+', choices=['Sims', 'ETRI', 'Toyota'],
                    help='Datasets to compute the distance distribution.')
parser.add_argument('--modalities', type=str, nargs='+', choices=['heatmaps', 'limbs', 'optical_flow', 'rgb', 'YOLO'],
                    help='Datasets to compute the distance distribution.')
args = parser.parse_args()

datasets = args.datasets
modalities = args.modalities

print(datasets, modalities, '\n')
energy_distance = energy_dist(datasets=datasets, modalities=modalities)
print(energy_distance)
exit()
with open(f"distance_distributions/energy_dists/{args.datasets[0]}_{args.datasets[1]}_{args.modalities[0]}_{args.modalities[1]}.txt", 'w') as f:
	f.write(f"{energy_distance}")


#plt.savefig(f"distance_distributions/{args.datasets[0]}_{args.datasets[1]}_{args.modalities[0]}_{args.modalities[1]}.svg")
#np.save(f"distance_distributions/npy/{args.datasets[0]}_{args.datasets[1]}_{args.modalities[0]}_{args.modalities[1]}.npy", dist_matrix.flatten())



