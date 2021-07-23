import torch

from model import *
from utils import init_device_seed, load_args

def train():
    args = load_args()
    device = init_device_seed(1234, args.cuda_visible)

    array = torch.randn(4, 3, 256, 256)
    arrayb = torch.randn(4, 1, 256, 256)
    V, feature_maps = Encoder(3)(array)

    array0 = torch.randn(8, 992, 16, 16)
    array1 = torch.randn(8, 992, 16, 16)
    # print(SCFT_Module()(array, array1).shape)
    # print(Decoder()(array0, feature_maps + [array0]).shape)
    # print(Generator()(array, arrayb).shape)
    print(Discriminator()(array).shape)

if __name__ == '__main__':
    train()