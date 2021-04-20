import numpy as np
import os

from torch.utils.data import Dataset
import torch

from PIL import Image


class MaskDataset(Dataset):
    def __init__(self, root, train, transform=None,
                 target_transform=None, modify="not_modify"):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.modify = modify
        self.classes = ['good', 'none', 'bad']

        if self._check_exists():
            self.data, self.targets = self.load_processed_data()
        else:
            "Preprocessed data not found. Please preprocess the data first."

    def load_processed_data(self):
        train_str = "train" if self.train else "test"
        modify_str = {"not_modify": "", "pad": "_pad", "crop": "_crop"}
        data = np.load(f"{self.root}/pre_processed/{train_str}_data{modify_str[self.modify]}.npy")
        targets = np.load(f"{self.root}/pre_processed/{train_str}_targets{modify_str[self.modify]}.npy") 

        return torch.tensor(data), torch.tensor(targets)

    def _check_exists(self) -> bool:
        train_str = "train" if self.train else "test"
        modify_str = {"not_modify": "", "pad": "_pad", "crop": "_crop"}
        return (os.path.exists(f"{self.root}/pre_processed/{train_str}_data{modify_str[self.modify]}.npy") and
                os.path.exists(f"{self.root}/pre_processed/{train_str}_targets{modify_str[self.modify]}.npy"))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = np.transpose(img.numpy(), (1, 2, 0))
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
