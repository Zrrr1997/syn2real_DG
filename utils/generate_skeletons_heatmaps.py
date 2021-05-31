import json
import argparse
import os

import numpy as np
import cv2

from tqdm import tqdm

''' Generates the skeleton heatmaps from the Alphapose estimations ''' 
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default=None, type=str,
                    help='Root directory with the already calculated AlphaPose joints.')
parser.add_argument('--height', default=368, type=int,
                    help='Image height.')
parser.add_argument('--width', default=640, type=int,
                    help='Image width.')
parser.add_argument('--scale', default=6, type=int,
                    help='Gaussian standard deviation for joint heatmaps. The higher the value, the larger the heatmaps.')
parser.add_argument('--result_dir', default=None, type=str,
                    help='Directory to store the calculated joint heatmaps and limbs.')
parser.add_argument('--min_frac_detections', default=0.5, type=float,
                    help='Minimum fraction of detections/frames per video needed.')

''' Draw a line from pt1 to pt2 with the given thickness '''
def draw_line(img, pt1, pt2, thickness=3):
	c = min(pt1[2], pt2[2]) # Take minimum confidence of both joints
	cv2.line(img, [int(p) for p in pt1[0:2]], [int(p) for p in pt2[0:2]], 255 * c, thickness)  

''' Read sequence three elements at a time'''
def trios(seq):
    result = []
    for e in seq:
        result.append(e)
        if len(result) == 3:
            yield result
            result = []

''' Draw Gaussian kernels of all joints (pts) onto the image (img) '''
def drawGaussian(img, pts, sigma):
    """Draw 2d gaussian on input image.
    Parameters
    ----------
    img: np.array
        A np.array with shape: `(3, H, W)` ---> The image.
    pt: a list of lists or tuples
        A list of points: [(x, y), (x, y)...] ---> The joint positions on the image plane
    sigma: int
        Sigma of gaussian distribution.
    """
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    for pt in pts:
        ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
        br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

        if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
            continue
        # Generate gaussian
        size = 2 * tmpSize + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) * (pt[2] / 6.0) # ---> We divide the uncertainty of the joint pt[2] by 6.0 because we use the coco-version of AlphaPose!

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], img.shape[1])
        img_y = max(0, ul[1]), min(br[1], img.shape[0])

        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(g[g_y[0]:g_y[1], g_x[0]:g_x[1]],img[img_y[0]:img_y[1], img_x[0]:img_x[1]])
    return img

''' Map gaussians onto empty image '''
def points_to_gaussian_heatmap(centers, height, width, scale):
    img = np.zeros((height, width))
    centers = np.array(centers)
    img = drawGaussian(img, centers, scale)
    return img	

''' Return an indexing with a certain length '''
def integer_to_filename(num, length = 5):
    base = '0' * length
    return (base + str(num))[-length:]

''' Remove duplicate detections from the joints (important for the WatchTV class) '''
def filter_duplicates(joints, img_ids):
	d = {}
	for i in img_ids: d[i] = i in d
	uniques = [k for k in d if not d[k]] # img_ids which only occur exactly once -> no double detections

	new_joints = []
	new_img_ids = []
	for j, i in zip(joints, img_ids):
		if i in uniques: # only single detection
			new_joints.append(j)
			new_img_ids.append(i)

		elif i not in new_img_ids: # this id has not been added yet
			joints_with_this_img_id = [joint for (joint, img) in zip(joints, img_ids) if img == i]
			joint_centers_with_this_img_id = np.array([np.array([np.array(el) for el in trios(joint)]) for joint in joints_with_this_img_id])
			areas = [(np.max(joint_center[:, 0]) - np.min(joint_center[:, 0])) * (np.max(joint_center[:, 1]) - np.min(joint_center[:, 1])) for joint_center in joint_centers_with_this_img_id]
			index = np.argmax(np.array(areas))
			new_joints.append(joints_with_this_img_id[index])
			new_img_ids.append(i)


		else: # this multiple detection is already taken care of
			continue
	return new_joints, new_img_ids

''' 
	The method expects the root_dir to contain directories, e.g. Co_S1K1_fC6, which contain alphapose-results.json each 
    
    	It generates images of the joint heatmaps and limbs and stores them in result_dir in the same directory names as the root_dir 
'''
def generate_heatmaps_and_limbs(H=640, W=368, scale=6, root_dir=None, result_dir=None, min_frac_detections=0.5):
	assert root_dir is not None
	assert result_dir is not None

	paths = [os.path.join(root_dir, el) for el in os.listdir(root_dir)]

	# Joint pairs to form lines
	l_pairs = [
		        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
		        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
		        (5.5, 11), (5.5, 12),  # Body
		        (11, 13), (12, 14), (13, 15), (14, 16)
	]                

	if not os.path.exists(os.path.join(result_dir, 'heatmaps')):
		os.mkdir(os.path.join(result_dir, 'heatmaps'))
	if not os.path.exists(os.path.join(result_dir, 'limbs')):
		os.mkdir(os.path.join(result_dir, 'limbs'))
	dropped_videos = 0
	print('Calculating all joint heatmaps and limbs...')
	for path in tqdm(paths): 



		f = open(os.path.join(path, 'alphapose-results.json'))
		data = json.load(f)
		joints = [el['keypoints'] for el in data] 
		img_ids = [el['image_id'] for el in data]
		img_ids = [int(el.split('.')[0]) for el in img_ids] 
		joints, img_ids = filter_duplicates(joints, img_ids) # Remove all duplicates -> Leave only detection with largest area


		''' Check if we have enough detections to even consider the video in our dataset '''
		video = cv2.VideoCapture(os.path.join(root_dir, os.path.basename(path), 'AlphaPose_' + os.path.basename(path) + '.avi'))
		n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
		if len(img_ids) / n_frames < min_frac_detections:
			dropped_videos += 1
			print("Skipping video", os.path.basename(path), "because it has too few detections:", round(100 * len(img_ids) / n_frames, 1), '% < ', str(100 * min_frac_detections), '%')
			continue

		if not os.path.exists(os.path.join(result_dir, 'heatmaps', os.path.basename(path))):
			os.mkdir(os.path.join(result_dir, 'heatmaps', os.path.basename(path)))
		if not os.path.exists(os.path.join(result_dir, 'limbs', os.path.basename(path))):
			os.mkdir(os.path.join(result_dir, 'limbs', os.path.basename(path)))

		writer_heatmaps = cv2.VideoWriter(os.path.join(result_dir, 'heatmaps', os.path.basename(path), 'heatmaps.avi'), cv2.VideoWriter_fourcc(*"MJPG"), 30,(W, H))
		writer_limbs = cv2.VideoWriter(os.path.join(result_dir, 'limbs', os.path.basename(path), 'limbs.avi'), cv2.VideoWriter_fourcc(*"MJPG"), 30,(W, H))
	

		img_counter = 0
		for i in range(n_frames):	
			''' Check if there is a detection for this frame '''
			heatmap, limbs = None, None
			if i not in img_ids:
				heatmap = np.zeros((H, W), dtype=np.uint8)
				limbs = np.zeros((H, W), dtype=np.uint8)
			else:
				''' Generate Heatmaps '''
				joint = joints[img_counter]
				joint_centers = np.array([np.array(el) for el in trios(joint)]) # [x, y, c] for each joint
				heatmap = points_to_gaussian_heatmap(joint_centers, H, W, scale)
				heatmap = ((heatmap - np.min(heatmap)) / (max(np.max(heatmap) - np.min(heatmap), 1e-4))) * 255 # Normalize to [0, 255], avoid division by 0
				heatmap = np.array(heatmap, dtype=np.uint8)
				
				''' Generate Limbs '''
				limbs = np.zeros((H, W), dtype=np.uint8)
				for l_pair in l_pairs:
					if l_pair[0] == 5.5: # Neck joint is not included into the original list for some reason -> Take mean of both shoulders
						draw_line(limbs, (joint_centers[5] + joint_centers[6]) / 2, joint_centers[l_pair[1]])
					else:
						draw_line(limbs, joint_centers[l_pair[0]], joint_centers[l_pair[1]])
				img_counter += 1

			''' Save images in result_dir '''
			writer_heatmaps.write(cv2.merge([heatmap, heatmap, heatmap]))
			writer_limbs.write(cv2.merge([limbs, limbs, limbs]))
			#cv2.imwrite(os.path.join(result_dir, 'heatmaps', os.path.basename(path), 'image_' + integer_to_filename(i + 1, length=5) +'.jpg'), heatmap)
			#cv2.imwrite(os.path.join(result_dir, 'limbs', os.path.basename(path), 'image_' + integer_to_filename(i + 1, length=5) +'.jpg'), limbs)
		writer_heatmaps.release()
		writer_limbs.release()
		assert(img_counter==len(img_ids))
	print('Done with', dropped_videos, 'dropped videos.')

if __name__ == '__main__':
	args = parser.parse_args()
	generate_heatmaps_and_limbs(H=args.height, 
				W=args.width, 
				scale=args.scale, 
				root_dir=args.root_dir,
				result_dir=args.result_dir,
				min_frac_detections=args.min_frac_detections)


