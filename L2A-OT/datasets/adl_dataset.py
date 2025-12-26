import glob
import os
import re

import numpy as np
import pandas as pd
#from torch.utils.data.dataset import T_co
from tqdm import tqdm

from datasets.generic_action_dataset import GenericActionDataset
from utils.action_encodings import adl_dataset_encoding, adl_dataset_decoding, sims_simple_dataset_encoding, sims_simple_dataset_decoding
from utils.utils import toyota_to_sims_df


class ADLDataset(GenericActionDataset):
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
                 test_on_sims=False,
                 dataset_name="ADL Dataset") -> None:
        # TODO: I think its bad style to place this before super, but currently it has to be here.
        self.action_pat_complex = re.compile(r"(.+)\.(.+)_p(\d\d)_r(\d\d)_v(\d\d)_c(\d\d)")
        self.action_pat_simple = re.compile(r"([^._]+)_p(\d\d)_r(\d\d)_v(\d\d)_c(\d\d)")
        self.test_on_sims = test_on_sims

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
                         dataset_name=dataset_name)

        self.action_dict_encode = adl_dataset_encoding
        self.action_dict_decode = adl_dataset_decoding
        if self.test_on_sims:
            self.action_dict_encode = sims_simple_dataset_encoding
            self.action_dict_decode = sims_simple_dataset_decoding

    def __getitem__(self, index):# -> T_co:
        return super().__getitem__(index)

    def index_video_paths(self, dataset_root):
        """
        This is different from super class, since the adl dataset does not make use of a separate folder
        per action class.
        :param dataset_root:
        :return:
        """
        return glob.glob(os.path.join(dataset_root, "*"))

    def extract_info_from_path(self, path):
        """
        This is also different due to the action not being a separate folder.
        :param path:
        :return:
        """
        file = os.path.split(path)[1]
        match = self.action_pat_simple.match(file)
        match_complex = self.action_pat_complex.match(file)
        if match:
            action = match.group(1)
            sub_action = None
            person = match.group(2)
            # r = match.group(3) -> ?
            # v = match.group(4) -> ?
            camera = match.group(5)

        elif match_complex:
            action = match_complex.group(1)
            sub_action = match_complex.group(2)
            person = match_complex.group(3)
            # r = match.group(4) -> ?
            # v = match.group(5) -> ?
            camera = match_complex.group(6)
        else:
            raise ValueError

        return action, sub_action, person, camera

    def detect_videos(self, dataset_root, cache_folder, use_cache, dataset_name):
        """
        In its generic form, this method expects a file system structure like "dataset_root/<action_names>/<video_ids>".
        It returns a dataframe which extracts this information and also contains the number of frames per video.
        :param dataset_name:
        :param dataset_root: The root folder of the dataset.
        :param cache_folder: If the cache is used, the dataframe is read from a file in this folder, instead.
        :param use_cache: If False, the dataframe is not read from file, but still written to a file.
        :return: A dataframe with columns ["video_id", "action", "frame_count", "base_path"]
        """

        vid_cache_name = f"video_info_cache_{dataset_name.replace(' ', '-')}.csv"
        if use_cache and os.path.exists(os.path.join(cache_folder, vid_cache_name)):
            video_info = pd.read_csv(os.path.join(cache_folder, vid_cache_name), index_col=False)
            print("Loaded video info from cache.")
        else:
            print("Searching for video folders on the file system...", end="")
            video_paths = self.index_video_paths(dataset_root)
            print("\b\b\b finished.")

            video_ids = [os.path.split(p)[1] for p in video_paths]
            info_tuples = [self.extract_info_from_path(p) for p in video_paths]

            print("Determining frame count per video...")
            vinfo = {"vid_id":      video_ids,
                     "vid_path":    [os.path.relpath(p, dataset_root) for p in video_paths],
                     "action":      [a[0] if a[1] is None else ".".join(a) for a in [itu[:2] for itu in info_tuples]],
                     "main_action": [itu[0] for itu in info_tuples],
                     "sub_action":  [itu[1] for itu in info_tuples],
                     "person":      [itu[2] for itu in info_tuples],
                     "camera":      [itu[3] for itu in info_tuples],
                     "frame_count": list(tqdm(map(lambda p: GenericActionDataset.count_frames(p), video_paths)))}

            video_info = pd.DataFrame(vinfo)
            if self.split_mode == 'test' and self.test_on_sims:
                video_info = toyota_to_sims_df(video_info)
                print("Filtered Toyota->Sims Actions", video_info['action'].unique())

            if cache_folder is not None and not self.test_on_sims:
                if not os.path.exists(cache_folder):
                    os.makedirs(cache_folder)
                video_info.to_csv(os.path.join(cache_folder, vid_cache_name))

        return video_info

    def prepare_split(self, video_info: pd.DataFrame, random_state=42):
        rng = np.random.default_rng(seed=random_state)

        vid_ids = np.array(sorted(list(set(video_info["vid_id"]))))

        if self.split_policy == "frac":
            train_msk = rng.random(size=len(vid_ids)) < self.split_train_frac
            not_train_msk = [not b for b in train_msk]

            rest_sample_count = sum(not_train_msk)

            val_msk = rng.random(size=rest_sample_count) < self.split_val_frac / (1 - self.split_train_frac)
            test_msk = [not b for b in val_msk]

            if self.split_mode == "train":
                final_id_set = set(vid_ids[train_msk])
            elif self.split_mode == "val":
                final_id_set = vid_ids[not_train_msk]
                final_id_set = set(final_id_set[val_msk])
            elif self.split_mode == "test":
                final_id_set = vid_ids[not_train_msk]
                final_id_set = set(final_id_set[test_msk])
            else:
                raise ValueError()

            fin_mask = video_info["vid_id"].apply(lambda vi: vi in final_id_set)
            return video_info[fin_mask]
        elif self.split_policy == "file":
            return self.get_split_from_file(video_info)
        elif self.split_policy == "cross-subject":
            train_subjects = [3, 4, 6, 7, 9, 12, 13, 15, 17, 19, 25]

            if self.split_mode == "train" or self.split_mode == "val":
                video_info = video_info[video_info["person"].isin(train_subjects)]

                train_msk = rng.random(size=len(video_info)) < 0.8  # TODO: Currently train:val hardcoded as 80:20

                if self.split_mode == "train":
                    return video_info[train_msk]
                else:
                    not_train_msk = [not b for b in train_msk]
                    return video_info[not_train_msk]
            elif self.split_mode == "test":
                video_info = video_info[~video_info["person"].isin(train_subjects)]
                return video_info
        elif self.split_policy == "cross-view-1" or self.split_policy == "cross-view-2":
            train_views = [1] if self.split_policy == "cross-view-1" else [1, 3, 4, 6, 7]
            val_views = [5]
            test_views = [2]

            if self.split_policy == "cross-view-2":
                # For cv-2, actions are also limited to the 19 cv1 actions.
                cv1_actions = {'Pour.Frombottle', 'Drink.Fromcan', 'Getup', 'Usetelephone', 'Drink.Frombottle', 'Leave',
                               'Readbook', 'Cutbread', 'Drink.Fromglass', 'Enter', 'Eat.Snack', 'Walk', 'Drink.Fromcup',
                               'Usetablet', 'Sitdown', 'Eat.Attable', 'Uselaptop', 'Pour.Fromcan', 'Takepills'}
                video_info = video_info[video_info["action"].isin(cv1_actions)]

            if self.split_mode == "train":
                return video_info[video_info["camera"].isin(train_views)]
            if self.split_mode == "val":
                return video_info[video_info["camera"].isin(val_views)]
            if self.split_mode == "test":
                return video_info[video_info["camera"].isin(test_views)]
        else:
            raise ValueError()
