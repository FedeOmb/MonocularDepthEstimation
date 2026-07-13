import os
import torch
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset


class DepthDataset(Dataset):
    TRAIN = 0
    VAL = 1
    TEST = 2

    def __init__(self, data_dir, train=TRAIN, transform=None):

        self.data_dir = data_dir
        if train==DepthDataset.TRAIN:
            rgb_img_dir = os.path.join(self.data_dir, 'rgb', 'train')
            depth_img_dir = os.path.join(self.data_dir, 'depth', 'train')
        elif train==DepthDataset.VAL:
            rgb_img_dir = os.path.join(self.data_dir, 'rgb', 'val')
            depth_img_dir = os.path.join(self.data_dir, 'depth', 'val')
        elif train==DepthDataset.TEST:
            rgb_img_dir = os.path.join(self.data_dir, 'rgb', 'test')
            depth_img_dir = os.path.join(self.data_dir, 'depth', 'test')
        else:
            raise ValueError('The train parameter is not valid')

        self.rgb_img_path = []
        self.depth_img_path = []
        for i in os.listdir(rgb_img_dir):
            self.rgb_img_path.append(os.path.join(rgb_img_dir, i))
            self.depth_img_path.append(os.path.join(depth_img_dir,  i[:-4] + 'npy'))

        self.transform = transform

    def __len__(self):
        return len(self.rgb_img_path)

    def __getitem__(self, idx):
        rgb_img = read_image(self.rgb_img_path[idx])
        rgb_img = rgb_img/255
        depth_img = torch.from_numpy(np.load(self.depth_img_path[idx]))
        depth_img = depth_img.unsqueeze(0).float().clip(min=0.0, max=20.0)
        if self.transform:
            rgb_img = self.transform(rgb_img)
            depth_img = self.transform(depth_img)
        return rgb_img, depth_img
