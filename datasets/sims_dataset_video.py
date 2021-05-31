import os
import pandas as pd
#from torch.utils.data.dataset import T_co

from datasets.generic_action_dataset_video import GenericActionDataset_Video
from utils.action_encodings import sims_simple_dataset_encoding, sims_simple_dataset_decoding


class SimsDataset_Video(GenericActionDataset_Video):
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
                 dataset_name="Sims Dataset",
                 modality="heatmaps",
                 n_channels=3) -> None:
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

        self.action_dict_encode = sims_simple_dataset_encoding
        self.action_dict_decode = sims_simple_dataset_decoding

    def __getitem__(self, index): # -> T_co:
        return super().__getitem__(index)

    def class_count(self):
        return len(self.action_dict_encode)

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

    trans = transforms.Compose([uaug.RandomSizedCrop(size=128, crop_area=(0.5, 0.5), consistent=True), uaug.ToTensor()])
    genad = SimsDataset_Video(dataset_root=os.path.expanduser("/cvhci/temp/zmarinov/joints_and_limbs/heatmaps"), vid_transform=trans,
                        split_mode="train", use_cache=False, n_channels=1)
    print(len(genad))
    print(genad[0]['vclip'].shape)
