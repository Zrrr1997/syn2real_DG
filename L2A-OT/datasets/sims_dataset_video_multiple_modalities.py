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


class SimsDataset_Video_Multiple_Modalities(SimsDataset_Video):
    def __init__(self,
                 dataset_root=None,
                 split_mode='train',
                 vid_transform=None,
                 seq_len=30,
                 seq_shifts=None,
                 downsample_vid=1,
                 max_samples=None,
                 split_policy="frac",
                 split_val_frac=0.0,
                 split_test_frac=1.0,
                 split_train_file=None,
                 split_val_file=None,
                 split_test_file=None,
                 sample_limit=None,
                 return_data=("vclip", "label"),
                 use_cache=True,
                 cache_folder="cache",
                 random_state=42,
                 per_class_samples=None,
                 dataset_name="Sims Dataset Multiple Modalities",
                 modalities=None,
                 n_channels=3,
                 dataset_roots=None,
                 n_channels_each_modality=None,
                 color_jitter=False,
                 color_jitter_trans=None,
                 test_on_sims=False,
                 fine_tune_late_fusion=False) -> None:
        self.test_on_sims = test_on_sims
        self.fine_tune_late_fusion=fine_tune_late_fusion
        self.n_channels_each_modality = n_channels_each_modality
        self.color_jitter=color_jitter
        self.color_jitter_trans=color_jitter_trans
        if split_mode == 'test':
           color_jitter = False

        assert not (self.color_jitter_trans is None and color_jitter)
        assert sum(self.n_channels_each_modality) == n_channels

        self.modalities = modalities
        self.dataset_roots = dataset_roots
        self.action_dict_encode = sims_simple_dataset_encoding
        self.action_dict_decode = sims_simple_dataset_decoding
        print("Using", self.n_channels_each_modality, "channels for each modality:", self.modalities)
        print(split_test_frac, split_mode)
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
                         dataset_name=dataset_name
                         )
        if self.color_jitter and ('rgb' in self.modalities):
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

        if "vclip" in self.return_data:
            v_len = sample["frame_count"]
            random_first = np.random.randint(low=start_frame, high=end_frame - self.seq_len - 1)
            frame_indices_vid = [np.arange(random_first, random_first + self.seq_len,  1)]

            # Read only [start_frame - end_frame] from video with cv2.VideoCapture
            seq_vid_all_modalities = []
            for i in range(len(self.modalities)):
                seq_vid_all_modalities.append(gad_v.frame_loader(frame_indices_vid[0],  os.path.join(self.dataset_roots[i], sample["vid_path_modality_"+str(i+1)], self.modalities[i] + '.avi'), self.n_channels_each_modality[i], self.modalities[i], self.color_jitter, self.color_jitter_trans))
            #seq_vid_second_modality = gad_v.frame_loader(frame_indices_vid[0],  os.path.join(self.dataset_root_second_modality, sample["vid_path_second_modality"], self.second_modality + '.avi'), self.n_channels_second_modality, self.second_modality, self.color_jitter, self.color_jitter_trans)


            ''' Add another channel dimension for grayscale images to be able to concatenate with other inputs
                
                Transform PIL images to numpy arrays because PIL does not support multi-channel images, i.e. concatenation of modalities.
            
                Make sure all modalities are of the same resolution, otherwise concatenation is impossible 
            '''
            different_shapes = False
            for i in range(len(self.modalities) - 1):
                if np.array(seq_vid_all_modalities[i][0]).shape[0:2] != np.array(seq_vid_all_modalities[i+1][0]).shape[0:2]:
                    different_shapes = True
            if len(self.modalities) > 1:
                second_shape_temp = np.array(seq_vid_all_modalities[1][0]).shape[0:2]
                second_shape = (second_shape_temp[1], second_shape_temp[0])

            for i in range(len(self.modalities)):
                if self.n_channels_each_modality[i] == 1:
                    if different_shapes:
                        seq_vid_all_modalities[i] = np.array([np.expand_dims(cv2.resize(np.array(el), second_shape, interpolation=cv2.INTER_AREA), axis=2) for el in seq_vid_all_modalities[i]])
                    else:
                        seq_vid_all_modalities[i] = np.array([np.expand_dims(np.array(el), axis=2) for el in seq_vid_all_modalities[i]])
                else:
                    if different_shapes:
                        seq_vid_all_modalities[i] = np.array([cv2.resize(np.array(el), second_shape, interpolation=cv2.INTER_AREA) for el in seq_vid_all_modalities[i]])
                    else:
                        seq_vid_all_modalities[i] = np.array([np.array(el) for el in seq_vid_all_modalities[i]])

            if len(self.modalities) > 1:
                seq_vid = np.concatenate((seq_vid_all_modalities[0], seq_vid_all_modalities[1]), axis=3)
            else:
                seq_vid = seq_vid_all_modalities[0]

            for i in np.arange(2, len(self.modalities)):
                seq_vid = np.concatenate((seq_vid, seq_vid_all_modalities[i]), axis=3)

            t_seq = self.vid_transform(seq_vid)

            del seq_vid
            # (self.seq_len, C, H, W) -> (C, self.seq_len, H, W)
            ret_dict["vclip"] = torch.stack(t_seq, 0).transpose(0, 1)
            if self.fine_tune_late_fusion:
                  assert len(self.n_channels_each_modality) == 2 # Only implemented for two modalities for now
                  #torch.stack([torch.stack(t_seq, 0).transpose(0, 1)[:self.n_channels_each_modality[0]], torch.stack(t_seq, 0).transpose(0, 1)[self.n_channels_each_modality[0]:]], 0, out=ret_dict["vclip"])
                  ret_dict["vclip"] = torch.cat([torch.stack(t_seq, 0).transpose(0, 1)[:self.n_channels_each_modality[0]], torch.stack(t_seq, 0).transpose(0, 1)[self.n_channels_each_modality[0]:]], dim=0)
        if "skmotion" in self.return_data:
            raise NotImplementedError()

        assert len(ret_dict.keys()) > 0

        return ret_dict

    def class_count(self):
        return len(self.action_dict_encode)

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
            video_info = pd.read_csv(os.path.join(cache_folder, vid_cache_name), index_col=False)
            #for i in range(len(self.modalities)):
            #      video_info['vid_path_modality_' + str(i+1)] = video_info['vid_path']
            print("Loaded video info from cache.")
            return video_info
        else:


            print("Searching for video folders for modality", self.modalities[0], "...")
            video_paths = self.index_video_paths(self.dataset_roots[0])
            print("\b\b\b finished.")

            video_ids = [os.path.split(p)[1] for p in video_paths]

            actions = [self.extract_info_from_path(p) for p in video_paths]
            actions = [a[0] if a[1] is None else ".".join(a) for a in actions]

            print("Determining frame count per video...")
            vinfo = {"vid_id":      video_ids,
                     "vid_path_modality_1":    [os.path.relpath(p, self.dataset_roots[0]) for p in video_paths],
                     "action":      actions,
                     "frame_count": list(tqdm(map(lambda p: SimsDataset_Video.count_frames_in_video(p, self.modalities[0]), video_paths)))}

            video_info_first_modality = pd.DataFrame(vinfo)
            if not self.test_on_sims and len(self.modalities) == 1:
                return video_info_first_modality
            for i in range(len(self.modalities)):
                if i == 0:
                   continue


                print("Searching for video folders for modality", self.modalities[i], "...")
                video_paths = self.index_video_paths(self.dataset_roots[i])
                print("\b\b\b finished.")

                video_ids = [os.path.split(p)[1] for p in video_paths]

                actions = [self.extract_info_from_path(p) for p in video_paths]
                actions = [a[0] if a[1] is None else ".".join(a) for a in actions]

                print("Determining frame count per video...")
                vinfo = {"vid_id":      video_ids,
                         "vid_path_modality_" + str(i + 1):    [os.path.relpath(p, self.dataset_roots[i]) for p in video_paths],
                         "action":      actions,
                         "frame_count_second_modality": list(tqdm(map(lambda p: SimsDataset_Video.count_frames_in_video(p, self.modalities[i]), video_paths)))}

                video_info_second_modality = pd.DataFrame(vinfo)


                video_info = pd.merge(video_info_first_modality, video_info_second_modality, how='inner', on=['vid_id', 'action'])

                ''' Make sure both modalities are aligned, i.e. have the same number of frames per video ---> Tolerated misalignments are ~6 frames, which is equivalent to 0.003s for 60s video'''
                diff = video_info[['frame_count']].values - video_info[['frame_count_second_modality']].values
                #print('Maximum frame misalignment offset:', np.max(np.abs(diff)), 'Number of misalignments:', np.sum(diff != 0))
                smaller_frame_count = video_info[['frame_count', 'frame_count_second_modality']].min(axis=1).values
                video_info['frame_count'] = smaller_frame_count
                assert np.all(video_info[['frame_count']].values <= video_info[['frame_count_second_modality']].values)

                columns = video_info.columns
                new_columns = [el if el != 'frame_count' else 'frame_count' for el in columns]
                video_info.columns = new_columns
                del video_info['frame_count_second_modality']
                video_info_first_modality = video_info # Update video_info
        if self.test_on_sims:
            # filter out training subjects from Toyota ADL
            if len(self.modalities) == 1:
                video_info = video_info_first_modality
            drop_idx = []
            train_ids = ['p03', 'p04', 'p06', 'p07', 'p09', 'p12', 'p13', 'p15', 'p17', 'p19', 'p25']
            for idx, row in video_info.iterrows():
                vid_id = row.vid_id
                for train_id in train_ids:
                    if train_id in vid_id:
                        drop_idx.append(idx)
            '''
            for idx, row in video_info.iterrows():
                vid_id = row.vid_id
                if not os.path.exists(os.path.join('/path/to/yolo_detections', row.vid_path_modality_1, 'detections.csv')):
                        drop_idx.append(idx)
            '''

            print(f"Dropped {len(drop_idx)} samples for testing \n" f"Remaining dataset size: {len(video_info) - len(drop_idx)}")

            video_info = video_info.drop(drop_idx, axis=0)
            if cache_folder is not None:
                if not os.path.exists(cache_folder):
                    os.makedirs(cache_folder)
                #video_info.to_csv(os.path.join(cache_folder, vid_cache_name))

        
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
    genad = SimsDataset_Video_Multiple_Modalities(dataset_roots=["/path/to/heatmaps", "/path/to/optical_flow"], use_cache=False, vid_transform=trans, modalities=["heatmaps", "optical_flow"], n_channels=4, n_channels_each_modality=[1,3], color_jitter=True, color_jitter_trans=color_jitter_trans, test_on_sims=True, fine_tune_late_fusion=True)
    print('Length of the dataset',len(genad))
    for i in range(10):
        print('Image size test', genad[i]['vclip'].shape)
