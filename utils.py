import torch
import numpy as np
import random
import os
import argparse


def init_device_seed(seed, cuda_visible):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    return device

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_visible', default='0', help='set CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    return args