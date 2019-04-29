import pandas as pd
import os
import torch
import random
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class OccMapDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.examples = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        rgb = io.imread(self.examples.iloc[item, 0])
        map = io.imread(self.examples.iloc[item, 1])

        example = {'rgb': rgb,
                   'map': map,
                  }
        if self.transform:
            example = self.transform(example)

        return example


class ToTensor(object):

    def __call__(self, sample):
        rgb = sample['rgb']
        map = np.expand_dims(sample['map'], 0)

        rgb = rgb.transpose((2, 0, 1))
        rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(torch.from_numpy(rgb))
        map = torch.from_numpy(map)
        return {'rgb': rgb,
                'map': map}


class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        rgb = sample['rgb']
        map = sample['map']

        rgb = transform.resize(rgb, self.output_size, mode='constant', preserve_range=False, anti_aliasing=False)

        return {'rgb': rgb,
                'map': map}


class Img_distro(object):

    def __init__(self, rot_deg, pix_offset):
        self.rot_deg = rot_deg
        self.pix_offset = pix_offset

    def __call__(self, sample):
        rgb = sample['rgb']
        map = sample['map']

        tran_mat = transform.AffineTransform(translation=(0, self.pix_offset))
        shifted = transform.warp(rgb, tran_mat, preserve_range=True)

        rotated = transform.rotate(shifted, self.rot_deg)

        return {'rgb': rotated,
                'map': map}



class Normalize(object):

    def __call__(self, sample):
        rgb = sample['rgb']
        map = sample['map']
        rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(rgb)
        return {'rgb': rgb,
                'map': map}


if __name__ == '__main__':
    val_set = OccMapDataset('dataset/Cityscapes/CS_val_64.csv',
                            transform=transforms.Compose([Rescale((256, 512)), Img_distro(0., 0), ToTensor()]))
    print('number of val examples:', len(val_set))
    print(val_set[0]['rgb'].shape)
    print(val_set[0]['map'].shape)


    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=8)
    print('show 3 examples')
    for i, temp_batch in enumerate(val_loader):
        if i == 0:
            print(temp_batch['rgb'])
            print(temp_batch['map'])
        break

