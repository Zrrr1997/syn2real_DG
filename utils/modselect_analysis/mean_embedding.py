import numpy as np
import argparse


arg_parser = argparse.ArgumentParser(description="Compute mean embedding and write it to file.")
arg_parser.add_argument("--embeddings_path", type=str, default=None,
                                  help="Path to embeddings.")
arg_parser.add_argument("--save_path", type=str, default=None,
                                  help="Path to save mean embedding vector.")

args = arg_parser.parse_args()

vecs = np.load(args.embeddings_path)
vecs = vecs.reshape((-1, vecs.shape[2])) # (num_vecs, emb_size)
mean_vec = np.mean(vecs, axis=0)
np.save(args.save_path, mean_vec)
