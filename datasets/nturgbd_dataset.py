#from torch.utils.data.dataset import T_co

from datasets.generic_action_dataset import GenericActionDataset


class NTURGBDDataset(GenericActionDataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index): # -> T_co:
        return super().__getitem__(index)
