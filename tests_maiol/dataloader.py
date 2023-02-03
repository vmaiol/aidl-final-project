import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#import matplotlib.pyplot as plt
import netCDF4 as nc
import glob
import numpy.ma as ma

POINT = './p2_2000_182.nc' #no la utilizo, solo era para tests
POINT_DATA = nc.Dataset(POINT) #no la utilizo, solo era para tests

class Reanalysisdata(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.label = y
        self.length = self.x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx], self.label[idx]

    def __len__(self):
        return self.length

def main():
    #a tener en cuenta
    #https://gis.stackexchange.com/questions/317735/read-multiple-netcdf4-using-python-3

    '''GETTING FILES AND DATA'''
    list_of_paths = glob.glob('./*.nc', recursive=True)
    print(list_of_paths)
    input_x = []
    input_y = []
    for point in list_of_paths:
        point_data = nc.Dataset(point)
        input=point_data['input'][:].data
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
    x_val = tensor_x[0:]
    x_train = tensor_x[0:]
    y_val = tensor_y[0:]
    y_train = tensor_y[0:]

    '''CREATING TRAINING AND TESTING SETS'''
    trainset = Reanalysisdata(x_train, y_train)
    testset = Reanalysisdata(x_val, y_val)

    hparams = {
        'batch_size' : 1,
        'test_batch_size':1
    }

    '''DATALOADERS'''
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=hparams['batch_size'],
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=hparams['test_batch_size'],
        shuffle=False)

    #print(train_loader.__dict__)
    #print(test_loader.__dict__)

    print("--------------------------------------------------------------------------")
    for batch_idx, (data, target) in enumerate(train_loader):
        print("BATCH INDEX: "+str(batch_idx))
        print("INPUT DATA:")
        print(data.shape)
        print(data)
        print("TARGET/LABEL")
        print(target)
        print("\n\n")
        #data, target = data.to(device), target.to(device)


if __name__ == "__main__":
    main()
