import os
import torch
import random
from datetime import datetime
from PIL import Image

from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import transforms

import numpy as np

from utils import init_device_seed, load_args_test
from model import Generator


def test():
    args = load_args_test()
    device = init_device_seed(1234, args.cuda_visible)
    output_dir = './result/' + datetime.now().strftime('%Y-%m-%d %H_%M_%S')
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load('./model/model_dict', map_location=device)
    generator = Generator(is_train=False).to(device)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    to_tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    to_pil = transforms.Compose([
        transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
        transforms.ToPILImage()
    ])

    file_names_list = os.listdir(f'{args.image_path}/color')

    for idx, file_name in enumerate(file_names_list):
        print('\r{}/{} {}'.format(idx, len(file_names_list), file_name), end=' ')
    
        image_r = Image.open(f'{args.image_path}/color/{file_name}')
        image_s = Image.open(f'{args.image_path}/sketch/{file_name}')
        image_r = torch.unsqueeze(to_tensor(image_r) * 2 - 1, 0).to(device)
        image_s = torch.unsqueeze(to_tensor(image_s) * 2 - 1, 0).to(device)

        output = generator(image_r, image_s)[0].detach().cpu()[0]
        output = to_pil(output)

        output.save(f'{output_dir}/{file_name}')

if __name__ == '__main__':
    test()