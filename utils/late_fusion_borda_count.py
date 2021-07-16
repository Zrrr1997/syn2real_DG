import pandas as pd
import numpy as np
import argparse
from matplotlib import pyplot as plt

''' Borda count voting for the classification
	- args:
		all_preds: 
				shape: (n_modalities, 5)
				All the prediction IDs of the modalities (top5)
	- returns:
		The ID of the "best" action class according to the borda count.
'''
def borda_count(all_preds):
	points = np.zeros(10)
	for preds in all_preds:
		for i, candidate in enumerate(preds):
			points[candidate] += 5 - i
	return np.argmax(points)

def extract_cols(row, modalities):
	cols = []
	for modality in modalities:
		cols.append([row['pred1_' + modality], row['pred2_' + modality], row['pred3_' + modality], row['pred4_' + modality], row['pred5_' + modality]])
	return cols

# Mean per-class accuracy ~ Balanced accuracy
def balanced_accuracy(labels, predictions):
	class_accs = np.ones(10)
	assert len(labels) == len(predictions)

	for i in range(10):
		correct = 0
		n_samples = sum(labels == i)
		for (label, pred) in zip(labels, predictions):
			if label == pred and label == i:
				correct += 1
		class_accs[i] = correct / n_samples
	return np.mean(class_accs) 
		
parser = argparse.ArgumentParser()
parser.add_argument('--csv_roots', default=['./heatmaps.csv'], type=str, nargs='+', help="The set of csv results files for the late fusion via post-processing.")
parser.add_argument('--modalities', default=['heatmaps'], type=str, nargs='+', help="The set of modalities.")
args = parser.parse_args()
csv_roots = args.csv_roots
modalities = args.modalities

dfs = [pd.read_csv(csv_root) for csv_root in csv_roots]
ids = [df['vid_id'].values for df in dfs]
for i, df_id in enumerate(ids[:-1]):
	assert np.all(df_id == ids[i+1]) 


for (modality, df) in zip(modalities, dfs):
	for col_name in ['pred1', 'pred2', 'pred3', 'pred4', 'pred5']:
		df.rename(columns={col_name:col_name + "_" + modality}, inplace=True)

	if modality != modalities[0]:

		df.drop(['label'], axis=1, inplace=True) # Leave label column only for first modality

final_df = pd.concat(dfs, axis=1)

final_df['final_prediction'] = final_df.apply(lambda row : borda_count(
extract_cols(row, modalities)),
axis=1)

all_preds = [final_df['pred1_' + modality].values for modality in modalities]
pred_fused = final_df['final_prediction'].values
labels = final_df['label'].values
print('Estimation count:', len(pred_fused))
for i, modality in enumerate(modalities):
	print(modality + " accuracy:", balanced_accuracy(labels, all_preds[i]))
print('Late fusion accuracy', balanced_accuracy(labels, pred_fused), 'for', modalities)

