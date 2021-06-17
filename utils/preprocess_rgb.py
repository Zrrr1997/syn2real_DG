import os, sys
import shutil
from tqdm import tqdm


''' Root dir contains directories with actions classes:
	Cook Drink Eat ... etc. -> Each of which contains ONLY videos
		- Co_S1D1_fC2.avi, Co_S2D1_fC2.avi ... etc.
'''
if len(sys.argv) != 2:
	print('Usage: python3 preprocess_rgb.py [root_dir]')
	exit()
root_dir = sys.argv[1]
action_dirs = sorted(os.listdir(root_dir))
print("Found actions:", action_dirs)

for a in tqdm(action_dirs):
	video_dirs = sorted(os.listdir(os.path.join(root_dir, a)))
	video_dirs = [el for el in video_dirs if el[-3:] == 'avi']
	if len(video_dirs) == 0:
		print("Action", a, "already processed...")
		continue
	video_dirs_full_path = [os.path.join(root_dir, a, el) for el in video_dirs]
	print('Video_dirs:', video_dirs_full_path)
	for vid_dir in video_dirs_full_path:
		if not os.path.exists(vid_dir[:-4]):
			os.mkdir(vid_dir[:-4])
			print('Directory created', vid_dir[:-4])
		shutil.move(vid_dir, os.path.join(vid_dir[:-4], 'rgb.avi'))

		
