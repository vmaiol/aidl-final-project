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

class Downscaling_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files_path = [root_dir+"/"+file for file in os.listdir(root_dir)]
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        path_file= self.files_path[index]
        point_data = nc.Dataset(path_file, mode="r")
        #print("POINT DATAAAA!!")
        #print(point_data)
        #exit()
        input, target = point_data['input'][0][:].data, point_data['target'][:].data
        target = torch.tensor(target)
        input = torch.tensor(input)
        if self.transform:
            input = self.transform(input)
        point_data.close()
        return input, target

    def __len__(self):
        return len(self.files_path)
