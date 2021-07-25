from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import random
import torch
from PIL import Image
import numpy as np


class Dataset(DataLoader):
    def __init__(self, dataset_dir, dirs):
        self.dataset_dir = dataset_dir
        self.dirs = dirs
        self.train_lists_a = os.listdir(f'{dataset_dir}/{dirs[0]}')
        self.train_lists_b = os.listdir(f'{dataset_dir}/{dirs[1]}')

    def __getitem__(self, index):
        image_i = Image.open(f'{self.dataset_dir}/{self.dirs[0]}/{self.train_lists_a[index]}')
        image_s = Image.open(f'{self.dataset_dir}/{self.dirs[1]}/{self.train_lists_b[index]}')

        image_i = np.array(image_i)
        image_gt = appearnace_transformation(image_i)
        image_r = spatial_transformation(image_gt)

        image_s = torch.unsqueeze(torch.from_numpy(image_s), dim=0)
        image_r = torch.from_numpy(image_r).permute(2, 0, 1)
        image_gt = torch.from_numpy(image_gt).permute(2, 0, 1)

        return [image_s, image_r], image_gt

    def __len__(self):
        return len(self.train_lists_a)