import torch
import os
import tqdm
import numpy as np

from torch.utils.data import DataLoader
from torch import nn, optim

from tqdm import tqdm

from model import Generator, Discriminator
from utils import init_device_seed, load_args
from dataset import Dataset
from losses import VGGLoss


BATCH_SIZE = 16
W_ADV = 1
W_REC = 30
W_TR = 1
W_PERC = 0.01
W_STYLE = 50
MARGIN_TR = 12

def train():
    args = load_args()
    device = init_device_seed(1234, args.cuda_visible)

    dataset = Dataset('./data/danbooru', ['color', 'sketch'])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    os.makedirs('./model', exist_ok=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    vgg_loss = VGGLoss().to(device)

    epoch = 0

    if args.load_model:
        checkpoint = torch.load('./model/model_dict', map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['model_dict'])
        epoch = checkpoint['epoch']

    optimizer_gen = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    criterion_mae = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    while epoch <= 100:
        epoch += 1

        generator.train()
        discriminator.train()

        pbar = tqdm(range(len(dataloader)))
        pbar.set_description('Epoch {}'.format(epoch))
        total_loss_gen = .0
        total_loss_con = .0
        total_loss_tr = .0
        total_loss_disc = .0

        for idx, images in enumerate(dataloader):
            image_s = images[0][0].to(device, dtype=torch.float32)
            image_r = images[0][1].to(device, dtype=torch.float32)
            image_gt = images[1].to(device, dtype=torch.float32)

            # Discriminator loss and update
            optimizer_disc.zero_grad()
            image_gen = generator(image_r, image_s).detach()
            label_gen = discriminator(torch.cat([image_gen, image_s], dim=1))
            label_gt = discriminator(torch.cat([image_gt, image_s], dim=1))

            loss_gen_disc = criterion_mse(label_gen, torch.zeros_like(label_gen))
            loss_gt_disc = criterion_mse(label_gt, torch.ones_like(label_gt))
            loss_disc = W_ADV * (loss_gen_disc + loss_gt_disc)

            loss_disc.backward()
            optimizer_disc.step()

            # Generator loss and update
            optimizer_gen.zero_grad()
            image_gen = generator(image_r, image_s)
            label_gen = discriminator(torch.cat([image_gen, image_s], dim=1))

            loss_rec = criterion_mae(image_gen, image_gt)
            loss_adv_gen = criterion_mse(label_gen, torch.ones_like(label_gen))
            loss_perc, loss_style = vgg_loss(image_gen, image_gt)
            loss_tr = .0

            loss_gen = W_TR * loss_tr + W_REC * loss_rec + W_ADV * loss_adv_gen + W_PERC * loss_perc + W_STYLE * loss_style

            loss_gen.backward()
            optimizer_gen.step()
            optimizer_gen.zero_grad()

            # Loss display
            total_loss_gen += W_ADV * loss_adv_gen.item()
            total_loss_con += W_REC * loss_rec.item() + W_PERC * loss_perc.item() + W_STYLE * loss_style.item()
            # total_loss_tr += W_TR * loss_tr.item()
            total_loss_disc += loss_disc.item()
            pbar.set_postfix_str('G_GAN: {}, G_Content: {}, G_tr: {}, D: {}'.format(
                np.around(total_loss_gen / (idx + 1), 4),
                np.around(total_loss_con / (idx + 1), 4),
                np.around(total_loss_tr / (idx + 1), 4),
                np.around(total_loss_disc / (idx + 1), 4)))
            pbar.update()

        # Save checkpoint per epoch
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'epoch': epoch,
        }, './model/model_dict')

if __name__ == '__main__':
    train()