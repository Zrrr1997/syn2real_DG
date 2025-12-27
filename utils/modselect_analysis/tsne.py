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

def export_legend(legend, filename="legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def fashion_scatter(x, colors, path):
    label_mapping =     {0: "Cook",
    1 : "Drink",
    2 : "Eat",
    3 : "Getup",
    4 : "Readbook",
    5 : "Usecomputer",
    6 : "Usephone",
    7 : "Usetablet",
    8 : "Walk",
    9 : "WatchTV"
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
    title = 'Heatmaps'
    if 'limbs' in path:
        title = 'Limbs'
    if 'optical_flow' in path:
        title = 'Optical Flow'
    if 'rgb' in path:
        title = 'RGB'
    if 'YOLO' in path:
        title = 'YOLO'
    plt.title(title, fontsize=45)

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    #if 'heatmaps' in path:
    #    legend = ax.legend(fontsize = 15, markerscale=3, bbox_to_anchor=(0.95, 1.0), loc='upper left',ncol=10)
    #    export_legend(legend, 'legend.pdf')


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

    embeddings = embeddings.reshape((-1, embeddings.shape[2]))
    labels = labels.flatten()
    new_e = []
    new_l = []
    if 'optical_flow' in args.embeddings_path:
        for i, e in enumerate(embeddings):
            if np.linalg.norm(e) > 12:
                new_e.append(e)
                new_l.append(labels[i])

        embeddings = np.array(new_e)
        labels = np.array(new_l)
    
    print(embeddings.shape, labels.shape)
    assert embeddings.shape[0] == labels.shape[0] # equal number of embeddings and labels

    tSNE(args.save_path, embeddings, labels)



