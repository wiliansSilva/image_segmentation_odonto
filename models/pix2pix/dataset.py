import numpy as np
import os

from torch.utils.data import Dataset

class BitewingDataset(Dataset):
    def __init__(self, ds_path: str="dataset/"):
        self.ds_path = ds_path

    def __len__(self):
        return 0

    def __getitem__(self, index):
        img_path = os.path.join(self.ds_path, )
