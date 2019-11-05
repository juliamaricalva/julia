import os
import json
import torch
import torchvision
import numpy as np
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''

        self.mode = mode
        self.dirimg = 'hw2_data/' + mode + '/img'
        self.data_dirimg = os.listdir(self.dirimg)
        self.img_dir = [self.dirimg + '/' + photo for photo in self.data_dirimg]

        ''' set up image trainsform '''
        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):

        ''' get data '''
        img_path = self.img_dir[idx]

        ''' read image '''
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        # img = torch.from_numpy(img)

        return self.transform(img)
        #to get the mask need os.path.basename(img_path)
