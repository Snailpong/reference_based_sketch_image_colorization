from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import random
import torch
from PIL import Image
import numpy as np

from utils import appearnace_transformation, spatial_transformation


class Dataset(DataLoader):
    def __init__(self, dataset_dir, dirs):
        self.dataset_dir = dataset_dir
        self.dirs = dirs
        self.train_lists_a = os.listdir(f'{dataset_dir}/{dirs[0]}')
        self.train_lists_b = os.listdir(f'{dataset_dir}/{dirs[1]}')
        self.resize = transforms.Resize((256,256))
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        image_i = Image.open(f'{self.dataset_dir}/{self.dirs[0]}/{self.train_lists_a[index]}')
        image_s = Image.open(f'{self.dataset_dir}/{self.dirs[1]}/{self.train_lists_b[index]}')

        image_i = np.array(self.resize(image_i), dtype=np.float32)
        image_gt = appearnace_transformation(image_i)
        image_r = spatial_transformation(image_gt)

        image_s = self.to_tensor(self.resize(image_s)) * 2. - 1.
        image_r = torch.from_numpy(image_r).permute(2, 0, 1) / 127.5 - 1.
        image_gt = torch.from_numpy(image_gt).permute(2, 0, 1) / 127.5 - 1.

        return [image_s, image_r], image_gt

    def __len__(self):
        return len(self.train_lists_a)