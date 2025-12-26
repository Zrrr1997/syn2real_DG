import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import cv2
from torchvision import transforms

import utils.augmentation
from datasets.sims_dataset_video import SimsDataset_Video
from utils.action_encodings import sims_simple_dataset_encoding, sims_simple_dataset_decoding


class SimsDataset_Video_with_YOLO_detections(SimsDataset_Video):
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
                 split_test_frac=1.0,
                 split_train_file=None,
                 split_val_file=None,
                 split_test_file=None,
                 sample_limit=None,
                 return_data=("vclip", "label", "detection"),
                 use_cache=True,
                 cache_folder="cache",
                 random_state=42,
                 per_class_samples=None,
                 dataset_name="Sims Dataset with YOLO detections",
                 modality="heatmaps",
                 n_channels=3,
                 dataset_root_second_modality=None,
                 n_channels_first_modality=1,
                 color_jitter=False,
                 color_jitter_trans=None,
                 detection_path=None,
                 test_on_sims=False) -> None:
        self.n_channels_first_modality = n_channels_first_modality
        self.detection_path = detection_path

        if split_mode == 'test':
           color_jitter = False
        self.color_jitter=color_jitter
        self.color_jitter_trans=color_jitter_trans
        self.test_on_sims=test_on_sims



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
                         n_channels=n_channels)
        if self.color_jitter and (self.modality=='rgb'):
           print("Using color jitter on the RGB modality!")


    def __getitem__(self, index): # -> T_co:
        ret_dict = {}
        gad_v = SimsDataset_Video
        if self.chunk_length is None:
            # operating on videos
            sample = self.video_info.iloc[index]
            start_frame = 0
            end_frame = sample["frame_count"] - 1
        else:
            vid_idx = self.chunk_to_vid_idxs[index]
            sample = self.video_info.loc[vid_idx]
            chunk = index - sample["chunk_start_idx"]
            start_frame = self.chunk_length * chunk
            end_frame = self.chunk_length * (chunk + 1)
            ret_dict["chunk"] = chunk

        ret_dict["vid_id"] = sample["vid_id"]

        if "label" in self.return_data:
            ret_dict["label"] = torch.tensor(self.action_dict_encode[sample["action"]])

        v_len = sample["frame_count"]
        random_first = np.random.randint(low=start_frame, high=end_frame - self.seq_len - 1)
        frame_indices_vid = [np.arange(random_first, random_first + self.seq_len,  1)]
        if "vclip" in self.return_data:

            # Read only [start_frame - end_frame] from video with cv2.VideoCapture
            seq_vid_first_modality = gad_v.frame_loader(frame_indices_vid[0],  os.path.join(self.dataset_root, sample["vid_path_first_modality"], self.modality + '.avi'), self.n_channels_first_modality, self.modality, self.color_jitter, self.color_jitter_trans)


            ''' Add another channel dimension for grayscale images to be able to concatenate with other inputs
                
                Transform PIL images to numpy arrays because PIL does not support volumetric images, i.e. concatenation of modalities.
            '''

            if self.n_channels_first_modality == 1:
                seq_vid_first_modality = np.array([np.expand_dims(np.array(el), axis=2) for el in seq_vid_first_modality])
            elif not self.color_jitter:
                seq_vid_first_modality = np.array([np.array(el) for el in seq_vid_first_modality])


            seq_vid = seq_vid_first_modality

            t_seq = self.vid_transform(seq_vid)

            del seq_vid
            # (self.seq_len, C, H, W) -> (C, self.seq_len, H, W)
            ret_dict["vclip"] = torch.stack(t_seq, 0).transpose(0, 1)

        if "detection" in self.return_data:
            ret_dict["detection"] = self.get_detections(frame_indices_vid[0], os.path.join(self.detection_path, sample["vid_path_first_modality"], 'detections.csv'))

        if "skmotion" in self.return_data:
            raise NotImplementedError()

        assert len(ret_dict.keys()) > 0

        return ret_dict

    def class_count(self):
        return len(self.action_dict_encode)

    ''' Get the detection bitvectors for each frame '''
    def get_detections(self, frame_indices, csv_path):
        df = pd.read_csv(csv_path)
        #print(df)

        # TODO - use the args.parameters to adjust these
        use_confidence = False
        use_one_hot = False
        use_inverse_distances = True

        all_dets = []
        for fr_id in frame_indices:
            det_one_hot = np.zeros(80)
            df_curr = df[df['frame_id'] == float(fr_id)]

            class_ids = df_curr['class_id'].values
            confidences = df_curr['confidence'].values
            xs = df_curr['x'].values
            ys = df_curr['y'].values
            bbox_centers = np.c_[xs, ys]        
            

            if use_confidence:
                for i, c_id in enumerate(class_ids):
                    det_one_hot[int(c_id)] = max(confidences[i], det_one_hot[int(c_id)])
            elif use_inverse_distances and 0 in class_ids:
                person_index = np.where(class_ids == 0)[0][0]
                person_bbox_center = bbox_centers[person_index]
                for i, c_id in enumerate(class_ids):
                    if c_id == 0:
                        continue
                    distance = np.linalg.norm(person_bbox_center - bbox_centers[i])
                    det_one_hot[int(c_id)] = max(1.0 / max(np.linalg.norm(person_bbox_center - bbox_centers[i]), 1e-4), det_one_hot[int(c_id)])
                summed_contribution = np.sum(det_one_hot)
                if summed_contribution != 0:
                    det_one_hot = det_one_hot / np.sum(det_one_hot) # Normalize to sum up to 1 when there are detections available
            elif use_one_hot: 
                for i, c_id in enumerate(class_ids):
                    det_one_hot[int(c_id)] = 1
            all_dets.append(det_one_hot)
        return torch.from_numpy(np.array(all_dets))
        

    def detect_videos(self, dataset_root, cache_folder, use_cache, dataset_name):
        """
        In its generic form, this method expects a file system structure like "dataset_root/<action_names>/<video_ids>".
        It returns a dataframe which extracts this information and also contains the number of frames per video.
        :param dataset_name:
        :param dataset_root: The root folder of the dataset.
        :param cache_folder: If the cache is used, the dataframe is read from a file in this folder, instead.
        :param use_cache: If False, the dataframe is not read from file, but still written to a file.
        :return: A dataframe with columns ["video_id", "action", "frame_count", "vid_path_first_modality", "vid_path_second_modality"]
        """

        vid_cache_name = f"video_info_cache_{dataset_name.replace(' ', '-')}.csv"
        if use_cache and os.path.exists(os.path.join(cache_folder, vid_cache_name)):
            print("Loading cache...", os.path.join(cache_folder, vid_cache_name))
            video_info = pd.read_csv(os.path.join(cache_folder, vid_cache_name), index_col=False)
            print("Loaded video info from cache.")

        else:

            print("Searching for video folders on the file system for first modality...", end="")
            video_paths = self.index_video_paths(dataset_root)
            print("\b\b\b finished.")

            video_ids = [os.path.split(p)[1] for p in video_paths]

            actions = [self.extract_info_from_path(p) for p in video_paths]
            actions = [a[0] if a[1] is None else ".".join(a) for a in actions]

            print("Determining frame count per video...")
            vinfo = {"vid_id":      video_ids,
                     "vid_path_first_modality":    [os.path.relpath(p, dataset_root) for p in video_paths],
                     "action":      actions,
                     "frame_count": list(tqdm(map(lambda p: SimsDataset_Video.count_frames_in_video(p, self.modality), video_paths)))}



            video_info = pd.DataFrame(vinfo)


            if cache_folder is not None:
                if not os.path.exists(cache_folder):
                    os.makedirs(cache_folder)
                video_info.to_csv(os.path.join(cache_folder, vid_cache_name))
                print("WROTE TO CACHE", os.path.join(cache_folder, vid_cache_name), len(video_info))
        if self.test_on_sims:
            # filter out training subjects from Toyota ADL

            drop_idx = []
            train_ids = ['p03', 'p04', 'p06', 'p07', 'p09', 'p12', 'p13', 'p15', 'p17', 'p19', 'p25']
            for idx, row in video_info.iterrows():
                vid_id = row.vid_id
                for train_id in train_ids:
                    if train_id in vid_id:
                        drop_idx.append(idx)


            print(f"Dropped {len(drop_idx)} samples for testing \n" f"Remaining dataset size: {len(video_info) - len(drop_idx)}")

            video_info = video_info.drop(drop_idx, axis=0)
            if cache_folder is not None:
                if not os.path.exists(cache_folder):
                    os.makedirs(cache_folder)
                video_info.to_csv(os.path.join(cache_folder, vid_cache_name))
        return video_info

    def get_split_from_file(self, video_info: pd.DataFrame):
        split_df = pd.read_csv(self.split_train_file)

        split_df["vid_id"] = split_df["VideoName"].apply(lambda vn: os.path.splitext(vn)[0])

        video_info = video_info.merge(split_df[["vid_id", "Split"]], on="vid_id", validate="one_to_one")

        video_info = video_info[video_info["Split"] == self.split_mode]

        return video_info


if __name__ == "__main__":
    import os
    import utils.augmentation as uaug
    from torchvision import transforms

    color_jitter_trans = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)

    trans = transforms.Compose([uaug.RandomSizedCrop(size=128, crop_area=(0.5, 0.5), consistent=True, asnumpy=True), uaug.ToTensor()])
    genad = SimsDataset_Video_with_YOLO_detections(dataset_root="/path/to/rgb", use_cache=False, vid_transform=trans, modality="rgb", n_channels=3, n_channels_first_modality=3, color_jitter=False, color_jitter_trans=color_jitter_trans, detection_path="/path/to/yolo_detections")
    print('Length of the dataset',len(genad))

    print('Image size test', genad[0]['vclip'].shape)
    print('Detection bitvector size test', genad[0]['detection'].shape)
    print(torch.max(genad[5]['detection'], dim=0)[0])

    print(genad[0]['label'])
