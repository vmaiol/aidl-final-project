import numpy as np
import torch
import os
import netCDF4 as nc

from torch.utils.data import Dataset


class Downscaling_dataset(Dataset):
  """ Downscaling dataset """

  def __init__(self, root_dir, transform=None):
    self.files_path = [root_dir+"/"+file for file in os.listdir(root_dir)]
    self.root_dir = root_dir
    self.transform = transform

  def get_data(self,position):
    path_file= self.files_path[position]
    file1 = nc.Dataset(path_file)
    input, target = file1.variables['input'], file1.variables['target']
    target = torch.tensor(target[:])
    input = torch.tensor(input[:])
    return input, target

  def __getitem__(self, position):
    path_file= self.files_path[position]
    file1 = nc.Dataset(path_file)
    input, target = file1.variables['input'], file1.variables['target']
    target = torch.tensor(target[:])
    input = torch.tensor(input[:])
    input = input.view(8,201,201) #5 is because we are only getting the first 5 variable: 'prep' precipitation
    return input, target

  def __len__(self):
    return len(self.files_path)

  def split_dataset(self, size=[1000,500,326]):
    return random_split(self.files_path,size)



  def get_vars(self, position):
    path_file= self.files_path[position]
    file1 = nc.Dataset(path_file)
    input = file1.variables['input']
    input = torch.tensor(input[:])
    return torch.split(input,5)
