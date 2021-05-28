import glob
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils import data
#from torch.utils.data.dataset import T_co
from tqdm import tqdm


class GenericActionDataset(data.Dataset):

    def __init__(self,
                 dataset_root=None,
                 split_mode='train',
                 vid_transform=None,
                 seq_len=32,
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
                 chunk_length=90,
                 return_data=("vclip", "label"),
                 use_cache=True,
                 cache_folder="cache",
                 frame_name_template="image_{:05}.jpg",
                 random_state=42,
                 per_class_samples=None,
                 dataset_name="Generic Action Dataset"
                 ) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.split_mode = split_mode
        self.vid_transform = vid_transform
        self.seq_len = seq_len
        self.min_length = self.seq_len
        self.seq_shifts = seq_shifts
        self.downsample_vid = downsample_vid
        self.max_samples = max_samples
        self.chunk_length = chunk_length

        self.split_policy = split_policy

        if split_policy == "frac":
            self.split_val_frac = split_val_frac if split_val_frac is not None else 0.
            self.split_test_frac = split_test_frac if split_test_frac is not None else 0.
            self.split_train_frac = 1.0 - self.split_val_frac - self.split_test_frac
        elif split_policy == "file":
            self.split_train_file = split_train_file
            self.split_val_file = split_val_file
            self.split_test_file = split_test_file

        self.sample_limit = sample_limit
        self.return_data = return_data

        self.use_cache = use_cache
        self.cache_folder = cache_folder

        self.frame_name_template = frame_name_template

        self.random_state = random_state

        self.dataset_name = dataset_name

        print("=================================")
        print(f'{self.dataset_name} ({self.split_mode} set). Dataset root: \n{self.dataset_root}')
        print(f'Split policy: Split {self.split_policy}')

        self.video_info = self.detect_videos(self.dataset_root, self.cache_folder, self.use_cache,
                                             self.dataset_name)
        print(f'Total number of video samples in this dataset: {len(self.video_info)}')

        print(f'Frames per sequence: {self.seq_len}\n'
              f'Downsampling on video frames: {self.downsample_vid}')

        # get action list
        actions = sorted(list(set(self.video_info["action"])))
        self.action_dict_encode = {a: i for i, a in enumerate(actions)}
        self.action_dict_decode = {i: a for a, i in self.action_dict_encode.items()}

        # filter out too short videos:
        drop_idx = []
        for idx, row in self.video_info.iterrows():
            vlen = row.frame_count
            if vlen < self.min_length:
                drop_idx.append(idx)

        print(f"Dropped {len(drop_idx)} samples due to insufficient length (less than {self.seq_len} frames).\n"
              f"Remaining dataset size: {len(self.video_info) - len(drop_idx)}")

        self.video_info = self.video_info.drop(drop_idx, axis=0)

        self.video_info = self.prepare_split(self.video_info, random_state=random_state)

        print(f"Number of samples in split {self.split_mode}: {len(self.video_info)}")

        if "skmotion" in self.return_data:
            raise NotImplementedError()

        if per_class_samples is not None:
            class_sample_list = []
            for action in actions:
                action_samples = self.video_info[self.video_info["action"] == action]
                if len(action_samples) > 0:
                    if len(action_samples) < per_class_samples:
                        print(f"Sampling class {action} multiple times because it does not contain "
                              f"enough samples ({len(action_samples)}/{per_class_samples}))")
                        class_sample_list.append(action_samples.sample(per_class_samples, replace=True))
                    else:
                        class_sample_list.append(action_samples.sample(per_class_samples, replace=False))

            self.video_info = pd.concat(class_sample_list, ignore_index=True)

        if self.max_samples is not None:
            self.video_info = self.video_info.sample(max_samples, random_state=random_state)

        if self.chunk_length:
            print(f"Interpreting videos as collections of chunk samples of length {self.chunk_length}.")
            self.video_info, self.chunk_to_vid_idxs = self.prepare_chunks(self.video_info, self.chunk_length)

            print(f"This increased the number of samples to {len(self.chunk_to_vid_idxs)}.")

        print("=================================")

    def __getitem__(self, index): # -> T_co:
        ret_dict = {}

        gad = GenericActionDataset
        if self.chunk_length is None:
            # operating on videos
            sample = self.video_info.iloc[index]
            start_frame = 0
            end_frame = sample["frame_count"]
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
            frame_indices = gad.idx_sampler(v_len, self.seq_len, sample["vid_path"], self.seq_shifts,
                                            frame_range=(start_frame, end_frame))

            frame_indices_vid = [idxs[::self.downsample_vid] for idxs in frame_indices]

            # If sequences are overlapping, we save IO time by only loading images once. Often, IO is the bottleneck.
            file_paths = [
                [os.path.join(self.dataset_root, sample["vid_path"], self.frame_name_template.format(i + 1)) for i in
                 idxs] for idxs in frame_indices_vid]

            file_path_set = set([fp for subl in file_paths for fp in subl])
            img_dict = {fp: gad.pil_loader(fp) for fp in file_path_set}

            seq_vids = [[img_dict[fp].copy() for fp in subl] for subl in file_paths]

            del img_dict

            # At this point we only make use of a single clip per sample. Can be changed in the future.
            seq_vid = seq_vids[0]

            t_seq = self.vid_transform(seq_vid)

            del seq_vid, seq_vids

            # (self.seq_len, C, H, W) -> (C, self.seq_len, H, W)
            t_seq = torch.stack(t_seq, 0).transpose(0, 1)

            ret_dict["vclip"] = t_seq

        if "skmotion" in self.return_data:
            raise NotImplementedError()

        assert len(ret_dict.keys()) > 0

        return ret_dict

    def __len__(self):
        return len(self.video_info) if self.chunk_length is None else len(self.chunk_to_vid_idxs)

    def prepare_split(self, video_info: pd.DataFrame, random_state=42):
        rng = np.random.default_rng(seed=random_state)

        if self.split_policy == "frac":
            train_msk = rng.random(size=len(video_info)) < self.split_train_frac
            not_train_msk = [not b for b in train_msk]

            rest_sample_count = sum(not_train_msk)

            val_msk = rng.random(size=rest_sample_count) < self.split_val_frac / (1 - self.split_train_frac)
            test_msk = [not b for b in val_msk]

            if self.split_mode == "train":
                return video_info[train_msk]
            elif self.split_mode == "val":
                not_train_samples = video_info[not_train_msk]
                return not_train_samples[val_msk]
            elif self.split_mode == "test":
                not_train_samples = video_info[not_train_msk]
                return not_train_samples[test_msk]
            else:
                raise ValueError()
        elif self.split_policy == "file":
            return self.get_split_from_file(video_info)
        else:
            raise ValueError()

    def class_count(self):
        len(self.action_dict_encode)

    def get_split_from_file(self, video_info):
        raise NotImplementedError

    def index_video_paths(self, dataset_root):
        return glob.glob(os.path.join(dataset_root, "*/*"))

    def extract_info_from_path(self, path):
        return os.path.split(os.path.split(path)[0])[1], None  # Second would be subaction if applicable

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
            actions = [self.extract_info_from_path(p) for p in video_paths]
            actions = [a[0] if a[1] is None else ".".join(a) for a in actions]

            print("Determining frame count per video...")
            vinfo = {"vid_id":      video_ids,
                     "vid_path":    [os.path.relpath(p, dataset_root) for p in video_paths],
                     "action":      actions,
                     "frame_count": list(tqdm(map(lambda p: GenericActionDataset.count_frames(p), video_paths)))}

            video_info = pd.DataFrame(vinfo)

            if cache_folder is not None:
                if not os.path.exists(cache_folder):
                    os.makedirs(cache_folder)
                video_info.to_csv(os.path.join(cache_folder, vid_cache_name))

        return video_info

    @staticmethod
    def count_frames(path, ending=".jpg"):
        return len(glob.glob(os.path.join(path, "*" + ending)))

    @staticmethod
    def idx_sampler(vlen, seq_len, vpath, sample_discretization=None, start_frame=None, frame_range=None,
                    multi_time_shifts=None):
        # cases:
        # - sampling with start frame
        # - sampling randomly along discretely chosen blocks
        # - sampling with random start

        # Special case handling: if multi time shifts is not None, but one of its entries, it means that the entry
        # should be chosen randomly within a possible range.

        # Copy is necessary, otherwise random value is replaced for None only at the first time and then propagated.
        shift_span = 0

        if multi_time_shifts is not None:
            multi_time_shifts = multi_time_shifts.copy()

            for idx, shift in enumerate(multi_time_shifts):
                if shift is not None:
                    continue
                else:  # Replace with random.
                    min_shift = min(shift for shift in multi_time_shifts if shift is not None)
                    max_shift = max(shift for shift in multi_time_shifts if shift is not None)

                    span = max_shift - min_shift

                    remainer = vlen - (span + seq_len)

                    possible_shifts = list(range(min_shift - remainer + 1, max_shift + remainer - 1))
                    if len(possible_shifts) > 4 * seq_len:
                        for i in range(-seq_len, seq_len):
                            possible_shifts.remove(i)

                    possible_shifts = possible_shifts if len(possible_shifts) > 0 else [
                        0]  # First and last are the same.

                    multi_time_shifts[idx] = np.random.choice(possible_shifts)

            # - Either a single sequence or multiple shifted sequences.

            shift_span = int(0 if multi_time_shifts is None else max(multi_time_shifts) - min(multi_time_shifts))

            # Make sure video filtering worked correctly.
            if vlen - (seq_len + shift_span) < 0:
                print(f"Tried to sample a video which is too short. \nVideo path: {vpath}")
                return [None]

        # Find the boundaries which are not out of range.
        time_shifts_positive = multi_time_shifts is None or min(multi_time_shifts) > 0

        first_possible_start = int(0 if time_shifts_positive else 0 - min(multi_time_shifts))

        last_possible_start = int(vlen - (seq_len + shift_span))

        assert first_possible_start <= last_possible_start

        if frame_range is not None:
            first_possible_start = frame_range[0]
            last_possible_start = min(frame_range[1] - seq_len, last_possible_start)

        # Sampling with a pre-chosen start-frame:
        if start_frame is not None:
            if first_possible_start <= start_frame <= last_possible_start:
                start_idx = start_frame
            else:
                print(f"Not all frames were available at position {start_frame}, for limited vlen {vlen} of {vpath}."
                      f" Sampling in the middle.")
                start_idx = first_possible_start + (last_possible_start - first_possible_start) // 2

        # Sampling discrete blocks.
        elif sample_discretization is not None:
            starts = range(first_possible_start, last_possible_start, sample_discretization)
            starts = starts if len(starts) > 0 else [first_possible_start]  # First and last are the same.

            start_idx = np.random.choice(starts)

        # Base case: sample a random start.
        else:
            starts = list(range(first_possible_start, last_possible_start))
            starts = starts if len(starts) > 0 else [first_possible_start]  # First and last are the same.

            start_idx = np.random.choice(starts)

        # Here we have one start index and we know that it is possible to sample all provided time shifts (if any)
        if multi_time_shifts is not None:
            seq_idxs = [start_idx + np.arange(seq_len) + time_shift for time_shift in multi_time_shifts]
        else:
            seq_idxs = [start_idx + np.arange(seq_len)]

        return seq_idxs

    @staticmethod
    def pil_loader(path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img.load()
                return img.convert('RGB')
    
    
    def prepare_chunks(self, video_info: pd.DataFrame, chunk_length):
        video_info["chunk_count"] = (video_info["frame_count"] // chunk_length).apply(lambda v: max(v, 1))
        video_info["chunk_start_idx"] = 0
        video_info["chunk_start_idx"] = [0] + list(video_info["chunk_count"].cumsum().iloc[:-1])

        chunk_to_vid_idx = []

        for idx, row in video_info.iterrows():
            chunk_to_vid_idx.extend([idx] * row.chunk_count)

        return video_info, chunk_to_vid_idx
    




if __name__ == "__main__":
    import utils.augmentation as uaug
    from torchvision import transforms

    trans = transforms.Compose([uaug.RandomSizedCrop(size=128, crop_area=(0.5, 0.5), consistent=True), uaug.ToTensor()])
    genad = GenericActionDataset(dataset_root=os.path.expanduser("~/datasets/sims_dataset/frames"),
                                 dataset_name="Sims Dataset",
                                 vid_transform=trans)
    print(len(genad))
    print(genad[0])
