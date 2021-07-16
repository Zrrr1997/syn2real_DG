import os
import shutil
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default=None, type=str,
                    help='Root directory with directories for each class Cook, Drink etc. and only videos inside e.g. Cook/Co_W2_fC2.avi. Output file system will contain directories with each video and each video will be renamed as rgb.avi')
parser.add_argument('--extension', default='.avi', type=str,
                    help='Video format. Example: .avi')
args = parser.parse_args()
root_dir = args.root_dir
extension = args.extension
if root_dir is None:
	print("No root directory given!")
	exit()
actions = sorted(os.listdir(root_dir))
print('Actions:', actions)
action_dirs = [os.path.join(root_dir, action) for action in actions]
print('Action directories:', action_dirs)
for action_dir in action_dirs:
	video_files = sorted(os.listdir(action_dir))
	video_dirs = [video_file[:-4] for video_file in video_files] # remove extension
	for video_dir in tqdm(video_dirs):
		if not os.path.exists(os.path.join(action_dir, video_dir)):
			os.mkdir(os.path.join(action_dir, video_dir))
		shutil.move(os.path.join(action_dir, video_dir + extension), os.path.join(action_dir, video_dir, 'rgb' + extension))
	
		
	
	
