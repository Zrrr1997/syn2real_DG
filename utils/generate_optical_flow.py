import json
import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm

from multiprocessing import Process

''' Generates the optical flow of the video sequences ''' 
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default=None, type=str,
                    help='Root directory with the videos.')
parser.add_argument('--height', default=368, type=int,
                    help='Image height.')
parser.add_argument('--width', default=640, type=int,
                    help='Image width.')
parser.add_argument('--result_dir', default=None, type=str,
                    help='Directory to store the calculated optical flow.')
parser.add_argument('--n_workers', default=3, type=int,
                    help='Number of worker processes..')


''' Return an indexing with a certain length '''
def integer_to_filename(num, length = 5):
    base = '0' * length
    return (base + str(num))[-length:]

''' 
	The method expects the root_dir to contain directories, e.g. Cook/Co_S1K1_fC6, Drink/Dr_S1K1_fC8 etc. IMPORTANT: Last edit includes the classnames in the root_dir!
    
    	It generates images or a video of the optical flow and stores them in result_dir with the same directory names as the root_dir 
'''

def generate_heatmaps_and_limbs(H=640, W=368, paths=None, root_dir=None, result_dir=None):
	assert root_dir is not None
	assert result_dir is not None
	assert paths is not None

	#paths = [os.path.join(root_dir, el) for el in sorted(os.listdir(root_dir))]

	print('Calculating optical flow...')
	for path in tqdm(paths): 
		#video = cv2.VideoCapture(os.path.join(root_dir, os.path.basename(path), 'AlphaPose_' + os.path.basename(path) + '.avi'))
		video = cv2.VideoCapture(path)
		n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

		if not os.path.exists(os.path.join(result_dir, 'optical_flow', os.path.basename(path.split('.')[0]))):
			os.mkdir(os.path.join(result_dir, 'optical_flow', os.path.basename(path.split('.')[0])))

		writer_optical_flow = cv2.VideoWriter(os.path.join(result_dir, 'optical_flow', os.path.basename(path.split('.')[0]), 'optical_flow.avi'), cv2.VideoWriter_fourcc(*"MJPG"), 30,(W, H))

		ret, frame1 = video.read()
		prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
		hsv = np.zeros_like(frame1)
		hsv[...,1] = 255

		init_shape = frame1.shape
		img_counter = 0
		while img_counter < n_frames:
			ret, frame2 = video.read()
			if frame2 is None:
				frame2 = np.zeros(init_shape, dtype=np.uint8)
			next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
			flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
			hsv[...,0] = ang * 180 / np.pi / 2
			hsv[...,2] = cv2.normalize(mag, None, 0, 255 , cv2.NORM_MINMAX)
			bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

			writer_optical_flow.write(bgr)
			#cv2.imwrite(os.path.join(result_dir, 'optical_flow', os.path.basename(path), 'image_' + integer_to_filename(img_counter + 1, length=5) +'.jpg'), bgr)
			prvs = next
			img_counter += 1
		writer_optical_flow.release()
		assert(n_frames == img_counter)
	print('Done!')

if __name__ == '__main__':
	args = parser.parse_args()
	class_paths = [os.path.join(args.root_dir, el) for el in sorted(os.listdir(args.root_dir))] # or only paths... (see last edit note)
	paths = []
	for c_p in class_paths:
		basepaths = sorted(os.listdir(c_p))
		paths += [os.path.join(c_p, b_p) for b_p in basepaths]


	processes = []
	size = len(paths) // args.n_workers
	for i in range(args.n_workers):
		start_ind = i * size
		end_ind = (i + 1) * size
		if i == args.n_workers - 1:
			end_ind = len(paths)
		print(start_ind, end_ind)
		p = Process(target=generate_heatmaps_and_limbs, args=(args.height, args.width, paths[start_ind: end_ind], args.root_dir, args.result_dir,))
		processes.append(p)
	for pr in processes:
		pr.start()
	for pr in processes:
		pr.join()
	print("All workers are done!")




