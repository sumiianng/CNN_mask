import numpy as np
import pandas as pd
import os

import torch
import torchvision.transforms as transforms

from PIL import Image


class Preprocess:
    def __init__(self, root, train, download=True,
                 over_sampling=False, modify="not_modify"):
        self.root = root
        self.train = train
        self.modify = modify

        if (download) or (not self._check_exists()):
            self.df = self.load_df()
            self.data, self.targets = self._data_process()

            if over_sampling:
                self.data, self.targets = self.oversampling_and_shuffle()
            self.save_processed_data()

    def _data_process(self):
        print("Process data.....")
        df = self.df
        # data
        data = []
        done_file = {}
        for image_name, xmin, ymin, xmax, ymax in zip(
                df['filename'],
                df['xmin'], df['ymin'],
                df['xmax'], df['ymax']):

            # open image
            if image_name not in done_file:
                done_file[image_name] = 1
                image_path = f"{self.root}/raw_data/images/{image_name}"
                img = Image.open(image_path)
                img = img.convert("RGB")

            img_partial = img.crop((xmin, ymin, xmax, ymax))

            if self.modify == "pad":
                x = xmax-xmin
                y = ymax-ymin
                img_partial = self.pad_to_square(img_partial, x, y)
            elif self.modify == "crop":
                size = min(xmax-xmin, ymax-ymin)
                crop = transforms.CenterCrop(size)
                img_partial = crop(img_partial)

            img_partial = img_partial.resize((48, 48))

            data.append(np.array(img_partial))

        data = np.array(data)
        data = torch.tensor(np.transpose(data, (0, 3, 1, 2)))

        # target
        targets = np.array(df['label'])
        targets[targets == "good"] = 0
        targets[targets == "none"] = 1
        targets[targets == "bad"] = 2
        targets = torch.tensor(targets.astype(np.int64), dtype=torch.long)

        return data, targets

    def oversampling_and_shuffle(self, ratio=(1, 0.3, 0.7)):
        train_data = self.data
        train_target = self.targets
        d0, d1, d2, d3 = train_data.numpy().shape

        each_class_num = train_target.bincount()
        catogoreis_num = len(each_class_num)
        max_num = each_class_num.max()
        class_sample_num = np.array(max_num*np.array(ratio)-each_class_num, dtype=np.int)

        for class_ in range(catogoreis_num):
            class_data = train_data[train_target == class_].clone()
            class_sample = np.random.randint(len(class_data),
                                             size=class_sample_num[class_])

            new_class_data = class_data[class_sample] + (torch.rand(len(class_sample), d1, d2, d3)-0.5)/1000000
            new_class_target = torch.ones(class_sample_num[class_],
                                          dtype=torch.long)*class_
            train_data = torch.cat((train_data,
                                    new_class_data), dim=0)
            train_target = torch.cat((train_target, new_class_target))

        index = np.arange(train_data.shape[0])
        np.random.shuffle(index)
        train_data = train_data[index]
        train_target = train_target[index]
        print(train_target.bincount())

        return train_data, train_target

    @staticmethod
    def pad_to_square(img, x, y):
        y_pad = x - min(x, y)
        x_pad = y - min(x, y)
        pad = transforms.Pad((x_pad, y_pad))
        return pad(img)

    def save_processed_data(self):
        if not os.path.isdir("./pre_processed"):
            os.mkdir("./pre_processed")

        train_str = "train" if self.train else "test"
        modify_str = {"not_modify": "", "pad": "_pad", "crop": "_crop"}

        np.save(f"{self.root}/pre_processed/{train_str}_data{modify_str[self.modify]}.npy", self.data)
        np.save(f"{self.root}/pre_processed/{train_str}_targets{modify_str[self.modify]}.npy", self.targets)

    def load_df(self):
        if self.train:
            return pd.read_csv(f'{self.root}/raw_data/train.csv')
        self.over_sampling = False
        return pd.read_csv(f'{self.root}/raw_data/test.csv')

    def _check_exists(self) -> bool:
        train_str = "train" if self.train else "test"
        modify_str = {"not_modify": "", "pad": "_pad", "crop": "_crop"}
        return (os.path.exists(f"{self.root}/pre_processed/{train_str}_data{modify_str[self.modify]}.npy") and
                os.path.exists(f"{self.root}/pre_processed/{train_str}_targets{modify_str[self.modify]}.npy"))
