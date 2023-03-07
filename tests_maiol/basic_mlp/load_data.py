import os
from torchmetrics.functional import r2_score
import numpy as np
import torch
import torch.nn as nn

#import matplotlib.pyplot as plt
import netCDF4 as nc
import glob
import numpy.ma as ma

from sklearn.model_selection import train_test_split #https://www.sharpsightlabs.com/blog/scikit-train_test_split/

def load_data(dir):
    #a tener en cuenta
    #https://gis.stackexchange.com/questions/317735/read-multiple-netcdf4-using-python-3

    '''GETTING FILES AND DATA'''
    #dir = "./data_sample_victor/"
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

        input_x.append(input) #q es 201 x 201 x 1, 1 pq nomes precipitacion
        input_y.append(target) #nomes un valor, target prep

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

    sets = [x_train, x_val, x_test, y_train, y_val, y_test]
    return sets
