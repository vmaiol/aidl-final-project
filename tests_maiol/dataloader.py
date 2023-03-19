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

def main_dataloader():
    #a tener en cuenta
    #https://gis.stackexchange.com/questions/317735/read-multiple-netcdf4-using-python-3

    '''GETTING FILES AND DATA'''
    dir = "./data_sample_victor/"
    list_of_paths = sorted(filter(os.path.isfile, glob.glob(dir+'*.nc', recursive=True)))

    #print(list_of_paths)
    input_x = []
    input_y = []
    for point in list_of_paths:
        point_data = nc.Dataset(point)
        input=point_data['input'][0][:].data # 0 -> precipitation
        #input=point_data['input'][:].data # 0 -> precipitation
        #print(input[0])
        #no estoy seguro si se necesita el IF aqui
        if float(point_data['target'][:].data)>0:
            target = float(point_data['target'][:].data)
        else:
            #target = float(point_data['target'][:].data)
            target = float(0)

        input_x.append(input)
        input_y.append(target)

    '''LIST TO NUMPY ARRAY TYPE FLOAT32'''
    #en principio los valores creo que ya estan en float, just in case¿?
    np_array_input_x = np.asarray(input_x).astype("float32")
    np_array_input_y = np.asarray(input_y).astype("float32")

    '''INPUT ARRAY TO TENSOR FLOAT, TARGET AS TENSOR OF FLOATS'''
    tensor_x = torch.tensor(np_array_input_x)
    tensor_y = np_array_input_y

    '''SPLITTING TRAIN AND TEST SETS'''
    #como estoy haciendo pruebas con 2 ficheros no me preocupo de dividir bien la data
    #[N:] o [:N] -> opción para dividir
    #x_val = tensor_x[0:]
    #x_train = tensor_x[0:]
    #y_val = tensor_y[0:]
    #y_train = tensor_y[0:

    (x_train_val, x_test, y_train_val, y_test) = train_test_split(tensor_x, tensor_y, test_size = .2)
    # Split train into train and val
    (x_train, x_val, y_train, y_val) = train_test_split(x_train_val, y_train_val, test_size=0.1)
    #print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)

    transform = transforms.Compose([
        # center-crop
        transforms.CenterCrop(64),
    ])

    '''CREATING TRAINING AND TESTING SETS'''
    #trainset
    trainset = Reanalysisdata(x_train, y_train, transform)
    print(x_train.shape, y_train.shape)
    #valset
    valset = Reanalysisdata(x_val, y_val, transform)
    print(x_val.shape, y_val.shape)
    #testset
    testset = Reanalysisdata(x_test, y_test, transform)
    print(x_test.shape, x_test.shape)

    hparams = {
        'batch_size' : 64,
        'test_batch_size':64
    }

    '''DATALOADERS'''
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=hparams['batch_size'],
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=1, #since we have a small valset... maybe 1?
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=1, #since we have a small testset... maybe 1?
        shuffle=False)

    #print(train_loader.__dict__)
    #print(test_loader.__dict__)

    '''print("--------------------------------------------------------------------------")
    for batch_idx, (data, target) in enumerate(train_loader):
        print("BATCH INDEX: "+str(batch_idx))
        print("INPUT DATA:")
        print(data.shape)
        print(data)
        print("TARGET/LABEL")
        print(target)
        print("\n\n")
        #data, target = data.to(device), target.to(device)'''

    return train_loader, val_loader, test_loader, hparams

#if __name__ == "__main__":
#main_dataloader()