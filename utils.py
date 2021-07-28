import torch
import numpy as np
import random
import os
import argparse
import cv2
import thinplate as tps


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
    parser.add_argument('--load_model', type=bool, default=False, help='loading pretrained model')
    parser.add_argument('--cuda_visible', default='0', help='set CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    return args

def load_args_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_visible', default='0', help='set CUDA_VISIBLE_DEVICES')
    parser.add_argument('--image_path', default='./data/val', help='set validation path')
    args = parser.parse_args()
    return args

def appearnace_transformation(image):
    image[..., 0] += random.randint(-50, 50)
    image[..., 1] += random.randint(-50, 50)
    image[..., 2] += random.randint(-50, 50)

    return np.clip(image, 0, 255)

def spatial_transformation(image):
    c_src = np.array([[0,0],[0,0.5],[0,1],[0.5,0],[0.5,0.5],[0.5,1],[1,0],[1,0.5],[1,1]])
    c_dst = np.clip(c_src + np.random.rand(9, 2) * 0.5 - 0.25, 0, 1)
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, (256, 256))
    mapx, mapy = tps.tps_grid_to_remap(grid, image.shape)
    return np.clip(cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC), 0, 255)