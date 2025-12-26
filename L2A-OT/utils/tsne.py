import numpy as np
import argparse



#n_iterations_tqdm = 36
#embeddings = embeddings.reshape((n_iterations_tqdm * 28), 3072)
#labels = labels.reshape((n_iterations_tqdm * 28))


import torch

import pickle
import numpy as np
import sys
import logging
import argparse
import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime

import time
import matplotlib.patheffects as PathEffects


import seaborn as sns
from sklearn.manifold import TSNE


def fashion_scatter(x, colors, path):
    label_mapping =     {0: "Heatmaps",
    1 : "Limbs",
    2 : "Optical Flow",
    3 : "RGB",
    4 : "Heatmaps Gen.",
    5 : "Limbs Gen.",
    6 : "Optical Flow Gen.",
    7 : "RGB Gen."
    }
    num_classes = len(np.unique(colors))
    sort_labels = np.sort(np.unique(colors))

    x_sep = [[] for i in range(num_classes)]
    for el, label in zip(x, colors):
        ind = list(sort_labels).index(label)
        x_sep[ind].append(el)

    print("Number of identities is: " + str(num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    for num, ids in enumerate(x_sep):
        xs = []
        ys = []
        for el in ids:
            xs.append(el[0])
            ys.append(el[1])
        ax.scatter(xs, ys, lw=0, s=20, alpha = 0.8, label = str(label_mapping[sort_labels[num]]))
        #plt.text(np.mean(np.array(xs)), np.mean(np.array(ys)), label_mapping[sort_labels[num]], fontsize=16)


    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    ax.legend(fontsize = 15, markerscale=3, bbox_to_anchor=(1.05, 1.0), loc='upper left')


    f.tight_layout()
    f.savefig(path)

    return f, ax

def tSNE(path, embeddings, labels):
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    RS = 123

    today = str(datetime.now()) 
    print("Embedding images...")
    emb_gallery = []
    lbs = []


    
    X_train = embeddings
    y_train = labels
    time_start = time.time()

    print("Computing tSNE...")
    fashion_tsne = TSNE(random_state=RS).fit_transform(X_train)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    print(fashion_tsne.shape)
    print(len(y_train))
    f, ax = fashion_scatter(fashion_tsne, y_train, path)
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="parser")



    arg_parser.add_argument("--embeddings_path", type=str, default=None,
                                  help="Path to embeddings to visualize")
    arg_parser.add_argument("--labels_path", type=str, default=None,
                                  help="Path to labels of the embeddings")
    arg_parser.add_argument("--save_path", type=str, default=None,
                                  help="Path to save the visualization.")
    args = arg_parser.parse_args()

    embeddings = np.load(args.embeddings_path)
    labels = np.load(args.labels_path)
    assert embeddings.shape[0] == labels.shape[0] # equal number of embeddings and labels
    tSNE(args.save_path)



