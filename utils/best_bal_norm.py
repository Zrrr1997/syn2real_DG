import pandas as pd
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process accuracies of models.')
parser.add_argument('--root_dir', type=str, default = '',
                    help='Root of the modality models with all the checkpoints.')
args = parser.parse_args()

dirs = os.listdir(args.root_dir)
dirs = [el for el in dirs if 'best' not in el]
dirs = [os.path.join(args.root_dir, el, 'logs', 'results_test_sims_video_multimodal_frac_s3d_32f._balanced_accuracies.csv') for el in dirs]
dirs = [el for el in dirs if 'best' not in el]

best_bals = np.zeros(3)
best_normals = np.zeros(3)
for d in dirs:
	df = pd.read_csv(d)
	bals = np.array([df['Mean_Balanced_Accuracy Top1'].values[0], df['Mean_Balanced_Accuracy Top3'].values[1], df['Mean_Balanced_Accuracy Top5'].values[2]])
	normals = np.array([df['Normal_Accuracy Top1'].values[0], df['Normal_Accuracy Top3'].values[1], df['Normal_Accuracy Top5'].values[2]])
	best_bals = np.maximum(best_bals, bals)
	best_normals = np.maximum(best_normals, normals)

print('Balanced:', best_bals)
print('Normal:', best_normals)
