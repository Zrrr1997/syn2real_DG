import json
import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm
from shutil import copyfile

from multiprocessing import Process

''' Copies the needed ADL rgb videos in the consistent format ''' 
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default=None, type=str,
                    help='Root directory with the videos.')
parser.add_argument('--result_dir', default=None, type=str,
                    help='Directory to store the copied videos.')
parser.add_argument('--n_workers', default=3, type=int,
                    help='Number of worker processes..')

''' Only copy needed classes '''
def custom_filter(path):
	return path.startswith(('Cook', 'Drink', 'Eat', 'Getup', 'Readbook', 'Sitdown', 'Uselaptop', 'Usetablet', 'Usetelephone', 'Walk', 'WatchTV'))

def copy_ADL(paths, result_dir):
	for path in tqdm(paths):
		copyfile(path, os.path.join(result_dir, os.path.basename(path)))
		
		
if __name__ == '__main__':
	args = parser.parse_args()
	paths = [os.path.join(args.root_dir, el) for el in sorted(os.listdir(args.root_dir))] # All video paths
	paths = [path for path in paths if custom_filter(os.path.basename(path))]

	processes = []
	size = len(paths) // args.n_workers
	for i in range(args.n_workers):
		start_ind = i * size
		end_ind = (i + 1) * size
		if i == args.n_workers - 1:
			end_ind = len(paths)
		print(start_ind, end_ind)
		p = Process(target=copy_ADL, args=(paths[start_ind: end_ind], args.result_dir,))
		processes.append(p)
	for pr in processes:
		pr.start()
	for pr in processes:
		pr.join()
	print("All workers are done!")




