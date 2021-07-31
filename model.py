import math
import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, stride):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class ConvBlock2(nn.Module):
    def __init__(self, channels_in, channels_out, kernel, stride, padding):
        super(ConvBlock2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel, stride, padding),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(992, 992, 3, 1, 1, bias=False),
            nn.BatchNorm2d(992),
            nn.Conv2d(992, 992, 3, 1, 1, bias=False),
            nn.BatchNorm2d(992),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return x + self.layers(x)


class Resblocks(nn.Module):
    def __init__(self):
        super(Resblocks, self).__init__()
        self.layers =  nn.Sequential(*[Resblock() for i in range(4)])

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, channels_in):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            ConvBlock(channels_in, 16, 1),
            ConvBlock(16, 16, 1),
            ConvBlock(16, 32, 2),
            ConvBlock(32, 32, 1),
            ConvBlock(32, 64, 2),
            ConvBlock(64, 64, 1),
            ConvBlock(64, 128, 2),
            ConvBlock(128, 128, 1),
            ConvBlock(128, 256, 2),
            ConvBlock(256, 256, 1)
        ])
        self.pool = nn.AdaptiveAvgPool2d(output_size=(16, 16))

    def forward(self, x):
        feature_maps = []
        down_feature_maps = []

        # ConvBlock forward pass
        for i in range(10):
            x = self.layers[i](x)
            feature_maps.append(x)

        # Downsample feature maps
        for i in range(8):
            down_feature_maps.append(self.pool(feature_maps[i]))
        down_feature_maps.append(feature_maps[8])
        down_feature_maps.append(feature_maps[9])

        V = torch.cat(down_feature_maps, dim=1)
    
        return V, feature_maps


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # [Block, cat_number]
        self.layers = nn.ModuleList([
            ConvBlock(992 + 992, 256, 1),
            ConvBlock(256, 256, 1),
            ConvBlock(256 + 128, 128, 1),
            ConvBlock(128, 128, 1),
            ConvBlock(128 + 64, 64, 1),
            ConvBlock(64, 64, 1),
            ConvBlock(64 + 32, 32, 1),
            ConvBlock(32, 32, 1),
            ConvBlock(32 + 16, 16, 1),
            ConvBlock(16, 16, 1),
            ConvBlock(16, 3, 1),
        ])
        self.concat_feature = [-1, None, 7, None, 5, None, 3, None, 1, None, None]
        self.tanh = nn.Tanh()

    def forward(self, x, features):
        for i in range(11):
            if self.concat_feature[i] != None:
                if self.concat_feature[i] != -1:
                    x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
                x = torch.cat([features[self.concat_feature[i]], x], dim=1)
            x = self.layers[i](x)

        return self.tanh(x)


class SCFT_Module(nn.Module):
    def __init__(self, is_train):
        super(SCFT_Module, self).__init__()
        self.d_v = 992
        self.W_v = nn.Linear(self.d_v, self.d_v, bias=False)
        self.W_k = nn.Linear(self.d_v, self.d_v, bias=False)
        self.W_q = nn.Linear(self.d_v, self.d_v, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.is_train = is_train

    def forward(self, V_r, V_s):
        batchs, d_v, h, w = V_r.size()
        V_r = V_r.reshape((batchs, h * w, d_v))     # (batchs, 256, 992)
        V_s = V_s.reshape((batchs, h * w, d_v))

        W_v_V_r = self.W_v(V_r)     # (batchs, 256, 992)
        W_k_V_r = self.W_k(V_r)
        W_q_V_s = self.W_q(V_s)

        dot = torch.bmm(W_q_V_s, W_k_V_r.permute(0, 2, 1)) / math.sqrt(self.d_v)  # (batchs, 256, 256)
        attention_matrix = self.softmax(dot)
        V_star = torch.bmm(attention_matrix, W_v_V_r)
        C = (V_star + V_s).reshape(batchs, d_v, h, w)

        if self.is_train:
            neg_sample = torch.cat([torch.arange(1, batchs), torch.tensor([0])])
            W_v_V_r_neg = W_v_V_r[neg_sample]
            dot_neg = torch.bmm(W_q_V_s, W_v_V_r_neg.permute(0, 2, 1)) / math.sqrt(self.d_v)
            return C, [dot, dot_neg]

        return C, None


class Generator(nn.Module):
    def __init__(self, is_train=True):
        super(Generator, self).__init__()
        self.encoder_r = Encoder(3)
        self.encoder_s = Encoder(1)
        self.scft = SCFT_Module(is_train)
        self.resblocks = Resblocks()
        self.decoder= Decoder()

    def forward(self, image_r, image_s):
        V_r, _ = self.encoder_r(image_r)
        V_s, feature_maps = self.encoder_s(image_s)
        C, dots = self.scft(V_r, V_s)
        Res_C = self.resblocks(C)

        image_gen = self.decoder(Res_C, feature_maps + [C])
        return image_gen, dots


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock2(4, 64, 4, 2, 1),
            ConvBlock2(64, 128, 4, 2, 1),
            ConvBlock2(128, 256, 4, 2, 1),
            ConvBlock2(256, 512, 4, 1, 1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.layers(x)