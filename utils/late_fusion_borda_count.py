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
def borda_count(all_preds, top=1):
	points = np.zeros(10)
	for preds in all_preds:
		for i, candidate in enumerate(preds):
			points[candidate] += 5 - i
	ranking = (-points).argsort()[:5]
	assert ranking[0] == np.argmax(points)
	return ranking[top - 1]

def extract_cols(row, modalities):
	cols = []
	for modality in modalities:
		cols.append([row['pred1_' + modality], row['pred2_' + modality], row['pred3_' + modality], row['pred4_' + modality], row['pred5_' + modality]])
	return cols

# Mean per-class (top-1) accuracy ~ Balanced accuracy
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
extract_cols(row, modalities), top=1),
axis=1)

final_df['final_prediction_2'] = final_df.apply(lambda row : borda_count(extract_cols(row, modalities), top=2),axis=1)
final_df['final_prediction_3'] = final_df.apply(lambda row : borda_count(extract_cols(row, modalities), top=3),axis=1)
final_df['final_prediction_4'] = final_df.apply(lambda row : borda_count(extract_cols(row, modalities), top=4),axis=1)
final_df['final_prediction_5'] = final_df.apply(lambda row : borda_count(extract_cols(row, modalities), top=5),axis=1)

all_preds = [final_df['pred1_' + modality].values for modality in modalities]
all_preds_2 = [final_df['pred2_' + modality].values for modality in modalities]
all_preds_3 = [final_df['pred3_' + modality].values for modality in modalities]
all_preds_4 = [final_df['pred4_' + modality].values for modality in modalities]
all_preds_5 = [final_df['pred5_' + modality].values for modality in modalities]

pred_fused = final_df['final_prediction'].values
pred_fused_2 = final_df['final_prediction_2'].values
pred_fused_3 = final_df['final_prediction_3'].values
pred_fused_4 = final_df['final_prediction_4'].values
pred_fused_5 = final_df['final_prediction_5'].values


labels = final_df['label'].values
print('Estimation count:', len(pred_fused))
for i, modality in enumerate(modalities):
	top1_acc = balanced_accuracy(labels, all_preds[i])
	top3_acc = balanced_accuracy(labels, all_preds_2[i]) + balanced_accuracy(labels, all_preds_3[i]) + top1_acc
	top5_acc = balanced_accuracy(labels, all_preds_4[i]) + balanced_accuracy(labels, all_preds_5[i]) + top3_acc
	print(modality + " accuracy: (top1, top3, top5)", top1_acc, top3_acc, top5_acc)
top1_fused = balanced_accuracy(labels, pred_fused)
top3_fused = balanced_accuracy(labels, pred_fused_2) + balanced_accuracy(labels, pred_fused_3) + top1_fused
top5_fused = balanced_accuracy(labels, pred_fused_4) + balanced_accuracy(labels, pred_fused_5) + top3_fused
print('Late fusion accuracy (top1, top3, top5)', top1_fused, top3_fused, top5_fused, 'for', modalities)

