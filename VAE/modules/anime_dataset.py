import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AnimeDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.max_num = 21551
        self.data = self.load_img()

    def load_img(self):
        root, max_num = self.root, self.max_num
        data = []
        for i in range(1, max_num+1):
            img = Image.open(f"{root}/anime-images/{i}.png").convert("RGB")
            img = np.array(img)
            img = np.transpose(img, (2, 0, 1))
            data.append(img)

        return torch.tensor(np.array(data)) / 255

    def __getitem__(self, index):
        img = self.data[index]
        img = transforms.ToPILImage()(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)
