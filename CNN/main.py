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

path = "data_sample"

torch.manual_seed(0)

hparams = {
    'batch_size':10,
    'num_epochs':20,
    'test_batch_size':10,
    'hidden_size':128,
    'num_classes':1,
    'learning_rate':1e-4,
    'log_interval':2,
    'early:stop': True
}

hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

files = os.listdir(path)
dataset = Downscaling_dataset(path)

train, test, val = torch.utils.data.random_split(dataset, [0.7,0.2,0.1])

data_loader = torch.utils.data.DataLoader(dataset, batch_size=hparams['batch_size'], drop_last=True, shuffle=True)
train_loader = torch.utils.data.DataLoader(train, batch_size=hparams['batch_size'], drop_last=True, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=hparams['batch_size'], drop_last=True, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=hparams['batch_size'], drop_last=True, shuffle=True)

model = Downscaling_model1().to(hparams['device'])
optimizer = torch.optim.Adam(model.parameters(), hparams['learning_rate'])
criterion = nn.MSELoss()

train_losses = []
test_losses = []
test_accs = []

# For each epoch
for epoch in range(1, hparams['num_epochs'] + 1):

  # Compute & save the average training loss for the current epoch
  train_loss = train_epoch(train_loader, model, optimizer, criterion, hparams, epoch)
  train_losses.append(train_loss)

  # TODO: Compute & save the average test loss & accuracy for the current epoch
  # HELP: Review the functions previously defined to implement the train/test epochs
  test_loss, test_accuracy = test_epoch(test_loader, model, criterion, hparams)

  test_losses.append(test_loss)
  test_accs.append(test_accuracy)
  break
