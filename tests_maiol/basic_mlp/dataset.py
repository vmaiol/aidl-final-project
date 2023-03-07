import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torchmetrics.functional import r2_score

#import matplotlib.pyplot as plt
import netCDF4 as nc
import glob
import numpy.ma as ma

from sklearn.model_selection import train_test_split #https://www.sharpsightlabs.com/blog/scikit-train_test_split/

#POINT = './p2_2000_182.nc' #no la utilizo, solo era para tests
#POINT_DATA = nc.Dataset(POINT) #no la utilizo, solo era para tests

class Reanalysisdata(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.label = y
        self.transform = transform
        self.length = self.x.shape[0]

    def __getitem__(self,idx):
        if self.transform:
            x = self.transform(self.x[idx])
        return x, self.label[idx]

    def __len__(self):
        return self.length
