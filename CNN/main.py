import netCDF4 as nc
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np


from model import *
from dataset import *
from resources import *

path = "../mini_dataset"

torch.manual_seed(0)

hparams = {
    'batch_size':20,
    'num_epochs':10,
    'val_batch_size':20,
    'hidden_size':128,
    'num_classes':1,
    'learning_rate':1e-4,
    'log_interval':2,
    'early:stop': True
}

#comment if you want to use gpu
hparams['device'] = 'cpu'

#Comment if you don't want to use gpu even if it's available
#hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

files = os.listdir(path)
dataset = Downscaling_dataset(path)

train, val, test = torch.utils.data.random_split(dataset, [0.7,0.2,0.1])

data_loader = torch.utils.data.DataLoader(dataset, batch_size=hparams['batch_size'], drop_last=True, shuffle=True)
train_loader = torch.utils.data.DataLoader(train, batch_size=hparams['batch_size'], drop_last=True, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=hparams['batch_size'], drop_last=True, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=hparams['batch_size'], drop_last=True, shuffle=True)

model = Downscaling_model1().to(hparams['device'])
optimizer = torch.optim.Adam(model.parameters(), hparams['learning_rate'])
criterion = nn.MSELoss()

train_losses = []
val_losses = []

model , train_losses, val_losses = fit(model, train_loader, val_loader, hparams, optimizer, criterion )

save_model(hparams['num_epochs'], model, optimizer, criterion)
save_plots(train_losses,val_losses)
