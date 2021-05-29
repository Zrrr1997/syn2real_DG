import glob
import os
import cv2

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from datasets.generic_action_dataset import GenericActionDataset


class GenericActionDataset_Video(GenericActionDataset):
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
                 use_cache=False,
                 cache_folder="cache",
                 random_state=42,
                 per_class_samples=None,
                 test_on_sims=False,
                 dataset_name="Generic Acton Dataset for Videos",
                 modality="heatmaps",
                 n_channels=3) -> None:

        self.modality = modality
        self.n_channels=n_channels
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

    def __getitem__(self, index): # -> T_co:
        ret_dict = {}
        gad_v = GenericActionDataset_Video
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
            frame_indices = gad_v.idx_sampler(v_len, self.seq_len, sample["vid_path"], self.seq_shifts,
                                            frame_range=(start_frame, end_frame))
            frame_indices_vid = [idxs[::self.downsample_vid] for idxs in frame_indices]

            # Read only [start_frame - end_frame] from video with cv2.VideoCapture
            seq_vid = gad_v.frame_loader(frame_indices_vid[0],  os.path.join(self.dataset_root, sample["vid_path"], self.modality + '.avi'), self.n_channels)

            t_seq = self.vid_transform(seq_vid)

            del seq_vid
            # (self.seq_len, C, H, W) -> (C, self.seq_len, H, W)
            ret_dict["vclip"] = torch.stack(t_seq, 0).transpose(0, 1)

        if "skmotion" in self.return_data:
            raise NotImplementedError()

        assert len(ret_dict.keys()) > 0

        return ret_dict


    def get_split_from_file(self, video_info):
        raise NotImplementedError

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
                     "frame_count": list(tqdm(map(lambda p: GenericActionDataset_Video.count_frames_in_video(p, self.modality), video_paths)))}

            video_info = pd.DataFrame(vinfo)

            if cache_folder is not None:
                if not os.path.exists(cache_folder):
                    os.makedirs(cache_folder)
                video_info.to_csv(os.path.join(cache_folder, vid_cache_name))

        return video_info

    @staticmethod
    def count_frames_in_video(path, modality):
        video = cv2.VideoCapture(os.path.join(path, modality + '.avi'))
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return n_frames


    @staticmethod
    def frame_loader(frame_indices, path, n_channels):
        assert len(frame_indices) != 0
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
        seq = []
        for i in range(len(frame_indices)):
            if n_channels == 3:
               seq.append(Image.fromarray(cap.read()[1]))
            elif n_channels == 1:
               img = cap.read()[1]
               seq.append(Image.fromarray(img[:,:,0].reshape((img.shape[0], img.shape[1])), 'L'))
            else:
               raise NotImplementedError()

        cap.release()
        return seq



if __name__ == "__main__":
    import utils.augmentation as uaug
    from torchvision import transforms

    trans = transforms.Compose([uaug.RandomSizedCrop(size=128, crop_area=(0.5, 0.5), consistent=True), uaug.ToTensor()])
    genad = GenericActionDataset(dataset_root=os.path.expanduser("~/datasets/sims_dataset/frames"),
                                 dataset_name="Sims Dataset",
                                 vid_transform=trans)
    print(len(genad))
    #print(genad[0])
