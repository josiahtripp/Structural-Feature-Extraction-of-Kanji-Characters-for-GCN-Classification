import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import SegmenterTraining as st
from scipy.ndimage import label

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.set_defaults(
    batch_size=16,
    dataset_name="KanjiVG",
    checkpoint_interval=5,
    channels=1,
    img_height=64,
    img_width=64,
    lr=0.0001
)
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 175
lambda_cluster = 3

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

'''
    Custom dataset integration for segmenter
'''
class SegmenterTrainingDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {"A": self.X[idx], "B": self.Y[idx]}

characters, stroke_groups = st.MakeTensors(regenerate=False, 
                                           regenerate_valid_characters=False, 
                                           regenerate_kanji_characters=False)

full_dataset = SegmenterTrainingDataset(characters, stroke_groups)

val_ratio = 0.1
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=opt.batch_size,
    shuffle=True
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=opt.batch_size,
    shuffle=True
)

# Minimum size of allowed pixel clusters in output (computed in loss and removed in sample)
min_cluster_size = 7

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["A"].type(Tensor))
    real_B = Variable(imgs["B"].type(Tensor))
    fake_B = generator(real_A)

    fake_B = (fake_B > 0.5).float()

    fake_B = torch.cat(torch.unbind(fake_B, dim=1), dim=1).unsqueeze(1)
    real_B = torch.cat(torch.unbind(real_B, dim=1), dim=1).unsqueeze(1)

    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)

    B, C, H, W = img_sample.shape
    assert C == 1, "Function assumes single-channel images"
    
    cleaned = []
    
    for i in range(B):
        img = img_sample[i, 0]  # shape [H, W]
        arr = img.cpu().numpy().astype(np.uint8)
        
        # Label connected components (8-connectivity)
        labeled, num_features = label(arr, structure=np.ones((3,3)))
        
        if num_features == 0:
            cleaned.append(img)
            continue
        
        # Count pixels per component
        counts = np.bincount(labeled.ravel())
        
        # Create mask of small clusters
        remove_mask = np.zeros_like(arr, dtype=bool)
        for j, c in enumerate(counts):
            if j == 0:
                continue  # skip background
            if c < min_cluster_size:
                remove_mask[labeled == j] = True
        
        arr[remove_mask] = 0
        cleaned.append(torch.from_numpy(arr).to(img_sample.device, dtype=img_sample.dtype))
    
    # Stack back into [B, 1, H, W]
    img_sample = torch.stack(cleaned, dim=0).unsqueeze(1)

    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=16, normalize=True)
1


# ----------
#  Training
# ----------

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Model inputs
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)

        # Pixel-wise loss
        weights = torch.where(real_B > 0, torch.ones_like(real_B), 0.1*torch.ones_like(real_B))
        loss_pixel = (weights * torch.abs(fake_B - real_B)).mean()

        # Small cluster loss
        loss_cluster = small_cluster_loss(fake_B.detach(), min_cluster_size)

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel + loss_cluster * lambda_cluster

        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
