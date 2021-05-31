import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms

import utils.augmentation
from datasets.sims_dataset_video import SimsDataset_Video
from utils.action_encodings import sims_simple_dataset_encoding, sims_simple_dataset_decoding


class SimsDataset_Video_Two_Modalities(SimsDataset_Video):
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
                 return_data=("vclip", "label"),
                 use_cache=True,
                 cache_folder="cache",
                 random_state=42,
                 per_class_samples=None,
                 dataset_name="Sims Dataset Two Modalities",
                 modality="heatmaps",
                 n_channels=3,
                 second_modality="limbs",
                 dataset_root_second_modality=None,
                 n_channels_first_modality=1,
                 n_channels_second_modality=1) -> None:
        self.n_channels_first_modality = n_channels_first_modality
        self.n_channels_second_modality = n_channels_second_modality
        print("Using", self.n_channels_first_modality, "channels for first modality and", self.n_channels_second_modality, "channels for second modality!")
        assert self.n_channels_first_modality + self.n_channels_second_modality == n_channels
        self.dataset_root_second_modality = dataset_root_second_modality
        self.second_modality = second_modality
        self.action_dict_encode = sims_simple_dataset_encoding
        self.action_dict_decode = sims_simple_dataset_decoding
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
            seq_vid_first_modality = gad_v.frame_loader(frame_indices_vid[0],  os.path.join(self.dataset_root, sample["vid_path_first_modality"], self.modality + '.avi'), self.n_channels_first_modality)
            seq_vid_second_modality = gad_v.frame_loader(frame_indices_vid[0],  os.path.join(self.dataset_root_second_modality, sample["vid_path_second_modality"], self.second_modality + '.avi'), self.n_channels_second_modality)


            ''' Add another channel dimension for grayscale images to be able to concatenate with other inputs
                
                Transform PIL images to numpy arrays because PIL does not support multi-channel images, i.e. concatenation of modalities.
            '''
            if self.n_channels_first_modality == 1:
                seq_vid_first_modality = np.array([np.expand_dims(np.array(el), axis=2) for el in seq_vid_first_modality])
            else:
                seq_vid_first_modality = np.array([np.array(el) for el in seq_vid_first_modality])

            if self.n_channels_second_modality == 1:
                seq_vid_second_modality = np.array([np.expand_dims(np.array(el), axis=2) for el in seq_vid_second_modality])
            else:
                seq_vid_second_modality = np.array([np.array(el) for el in seq_vid_second_modality])
            seq_vid = np.concatenate((seq_vid_first_modality, seq_vid_second_modality), axis=3)


            t_seq = self.vid_transform(seq_vid)

            del seq_vid
            # (self.seq_len, C, H, W) -> (C, self.seq_len, H, W)
            ret_dict["vclip"] = torch.stack(t_seq, 0).transpose(0, 1)

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
            print("Loaded video info from cache.")
        else:
            print("Searching for video folders on the file system...(for first modality)", end="")
            video_paths = self.index_video_paths(dataset_root)
            print("\b\b\b finished.")

            video_ids = [os.path.split(p)[1] for p in video_paths]

            actions = [self.extract_info_from_path(p) for p in video_paths]
            actions = [a[0] if a[1] is None else ".".join(a) for a in actions]

            print("Determining frame count per video...")
            vinfo = {"vid_id":      video_ids,
                     "vid_path_first_modality":    [os.path.relpath(p, dataset_root) for p in video_paths],
                     "action":      actions,
                     "frame_count_first_modality": list(tqdm(map(lambda p: SimsDataset_Video.count_frames_in_video(p, self.modality), video_paths)))}

            video_info_first_modality = pd.DataFrame(vinfo)


            print("Searching for video folders on the file system...(for second modality)", end="")
            video_paths = self.index_video_paths(self.dataset_root_second_modality)
            print("\b\b\b finished.")

            video_ids = [os.path.split(p)[1] for p in video_paths]

            actions = [self.extract_info_from_path(p) for p in video_paths]
            actions = [a[0] if a[1] is None else ".".join(a) for a in actions]

            print("Determining frame count per video...")
            vinfo = {"vid_id":      video_ids,
                     "vid_path_second_modality":    [os.path.relpath(p, self.dataset_root_second_modality) for p in video_paths],
                     "action":      actions,
                     "frame_count_second_modality": list(tqdm(map(lambda p: SimsDataset_Video.count_frames_in_video(p, self.second_modality), video_paths)))}

            video_info_second_modality = pd.DataFrame(vinfo)

            video_info = pd.merge(video_info_first_modality, video_info_second_modality, how='inner', on=['vid_id', 'action'])

            ''' Make sure both modalities are aligned, i.e. have the same number of frames per video '''
            assert np.all(video_info[['frame_count_first_modality']].values == video_info[['frame_count_second_modality']].values)
            columns = video_info.columns
            new_columns = [el if el != 'frame_count_first_modality' else 'frame_count' for el in columns]
            video_info.columns = new_columns
            del video_info['frame_count_second_modality']

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
    augmentation_settings = {
        "rot_range":      (-abs(min([20.])), abs(max([20.]))),
        "hue_range":      (0. - abs(min([0.5])), 0. + abs(max([0.5]))),
        "sat_range":      (1. - abs(min([1])), 1. + abs(max([1]))),
        "val_range":      (1. - abs(min([0.8])), 1. + abs(max([0.8]))),
        "hue_prob":       1.,
        "cont_range":     (1. - abs(min([0.0])), 1. + abs(max([0.0]))),
        "crop_arr_range": (1., 1.)
        }
    color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)

    normalization = utils.augmentation.Normalize((0.5,), (0.5,))
    transform_train = transforms.Compose([
        utils.augmentation.RandomRotation(degree=augmentation_settings["rot_range"], asnumpy=True),
        utils.augmentation.RandomSizedCrop(size=128, crop_area=augmentation_settings["crop_arr_range"],
                                           consistent=True, force_inside=True, asnumpy=True),
        utils.augmentation.ToTensor(),
        normalization
        ])
    trans = transforms.Compose([uaug.RandomSizedCrop(size=128, crop_area=(0.5, 0.5), consistent=True, asnumpy=True), uaug.ToTensor()])
    genad = SimsDataset_Video_Two_Modalities(dataset_root="/cvhci/temp/zmarinov/joints_and_limbs/heatmaps", use_cache=False , vid_transform=transform_train, dataset_root_second_modality="/cvhci/temp/zmarinov/joints_and_limbs/limbs", modality="heatmaps", second_modality="limbs", n_channels=2, n_channels_first_modality=1, n_channels_second_modality=1)
    print(len(genad))
    print('Image size test', genad[0]['vclip'].shape)
