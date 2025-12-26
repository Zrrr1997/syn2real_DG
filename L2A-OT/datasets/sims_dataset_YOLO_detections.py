import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import cv2
from torchvision import transforms

import utils.augmentation
from datasets.sims_dataset_video_with_YOLO_detections import SimsDataset_Video_with_YOLO_detections
from utils.action_encodings import sims_simple_dataset_encoding, sims_simple_dataset_decoding


class SimsDataset_YOLO_detections(SimsDataset_Video_with_YOLO_detections):
    def __init__(self,
                 dataset_root=None,
                 split_mode='train',
                 vid_transform=None,
                 seq_len=30,
                 seq_shifts=None,
                 downsample_vid=1,
                 max_samples=None,
                 split_policy="frac",
                 split_val_frac=0.1,
                 split_test_frac=0.1,
                 split_train_file=None,
                 split_val_file=None,
                 split_test_file=None,
                 sample_limit=None,
                 return_data=("label", "detection"),
                 use_cache=True,
                 cache_folder="cache",
                 random_state=42,
                 per_class_samples=None,
                 dataset_name="Sims Dataset only for YOLO detections",
                 modality='rgb',
                 n_channels=3,
                 dataset_root_second_modality=None,
                 n_channels_first_modality=1,
                 color_jitter=False,
                 color_jitter_trans=None,
                 detection_path=None,
                 test_on_sims=False) -> None:

        self.detection_path = detection_path
        self.test_on_sims = test_on_sims
        self.action_dict_encode = sims_simple_dataset_encoding

        super().__init__(dataset_root=dataset_root,
                         split_mode=split_mode,
                         vid_transform=vid_transform,
                         seq_len=seq_len,
                         seq_shifts=seq_shifts,
                         downsample_vid=downsample_vid,
                         max_samples=max_samples,
                         split_policy=split_policy,
                         split_val_frac=split_val_frac,
                         split_test_frac=split_test_frac,
                         split_train_file=split_train_file,
                         split_val_file=split_val_file,
                         split_test_file=split_test_file,
                         sample_limit=sample_limit,
                         return_data=return_data,
                         use_cache=use_cache,
                         cache_folder=cache_folder,
                         random_state=random_state,
                         per_class_samples=per_class_samples,
                         dataset_name=dataset_name,
                         modality=modality,
                         n_channels=n_channels,
                         detection_path=detection_path,
                         test_on_sims=test_on_sims)



if __name__ == "__main__":
    import os
    import utils.augmentation as uaug
    from torchvision import transforms

    color_jitter_trans = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)

    trans = transforms.Compose([uaug.RandomSizedCrop(size=128, crop_area=(0.5, 0.5), consistent=True, asnumpy=True), uaug.ToTensor()])
    genad = SimsDataset_Video_with_YOLO_detections(dataset_root="/cvhci/temp/zmarinov/rgb", use_cache=False, vid_transform=trans, modality="rgb", n_channels=3, n_channels_first_modality=3, color_jitter=False, color_jitter_trans=color_jitter_trans, detection_path="/home/zmarinov/repos/yolov3/runs/Sims_YOLO/")
    print('Length of the dataset',len(genad))

    print('Detection bitvector size test', genad[0]['detection'].shape)
    print(genad[0]['detection'][0])
    print(genad[0]['label'])
