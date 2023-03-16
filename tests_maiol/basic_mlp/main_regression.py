from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import datetime
import shutil

#from load_data import * no se utiliza load_datapy
from dataset import *
from hyperparameter_config import *
from model import *
from collections import OrderedDict
import matplotlib.pyplot as plt

'''TRAINING ON THE GPU'''
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")
#print(torch.cuda.device_count())
#print(torch.cuda.current_device())
#print(torch.cuda.get_device_name(0))

def create_dataloader(config, sets):
    '''-----NO SE UTILIZAAA!!!!!!!!!----'''
    transform = transforms.Compose([
        # center-crop
        transforms.CenterCrop(config['img_size']),
    ])

    x_train = sets[0]
    x_val = sets[1]
    x_test = sets[2]
    y_train = sets[3]
    y_val = sets[4]
    y_test = sets[5]

    '''CREATING TRAINING AND TESTING SETS'''
    #trainset
    trainset = Reanalysisdata(x_train, y_train, transform)
    #print(x_train.shape, y_train.shape)
    #valset
    valset = Reanalysisdata(x_val, y_val, transform)
    #print(x_val.shape, y_val.shape)
    #testset
    testset = Reanalysisdata(x_test, y_test, transform)
    #print(x_test.shape, y_test.shape)

    '''DATALOADERS'''
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config['batch_size'],
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=config['batch_size'],
        shuffle=True)

    #NO HACE FALTA HACER EL DATALOADER DEL TEST, YA QUE CON LA MEJOR CONFIG ENTRENADA EN TRAIN Y VAL, DECIDIREMOS QUE BATCH utilizar
    #test_loader = torch.utils.data.DataLoader(
        #testset,
        #batch_size=config['val_batch_size'], #since we have a small testset... maybe 1?
        #shuffle=False)

    return train_loader, val_loader #,test_loader

def check_loss_list(list, val):
    for x in list:
        if val<x:
            return True
    return False

def plots_learning_curves(plot_type, loss_stats, dir=None):
    '''plots de las loss, accuracy...'''
    #loss_stats contiene -> loss_stats ={'train':[], 'val':[], 'test':[]}
    if plot_type == "train_val":
        plot_name = "./plot_train_val.png"
        plt.figure(figsize=(10, 8))
        plt.subplot(2,1,1)
        plt.xlabel('Epoch')
        plt.ylabel('MSELoss')
        plt.plot(loss_stats['train'], label='train')
        plt.plot(loss_stats['val'], label='val')
        plt.legend()
    elif plot_type == "test":
        plot_name = "./plot_test_best_config.png"
        plt.figure(figsize=(10, 8))
        plt.subplot(2,1,1)
        plt.xlabel('Epoch')
        plt.ylabel('MSELoss')
        plt.plot(loss_stats['test'], label='test')

    plt.savefig(plot_name, bbox_inches='tight')

def save_model(model, path):
    torch.save(model.state_dict(), path)

def layers_config(img_size, img_vars):
    #para construir una arquitectura,
    #minimo de la layer output de 16
    #me lo invento un poco...
    n_layers = 0
    layers = []
    layers.append([img_size*img_size, int((img_size*img_size)/2), img_vars])
    img_size = img_size * img_size
    contador = 0
    if img_size > 16:
        while int(img_size/2) != 1:
            #print(img_size)
            img_size = int(img_size/2)
            next_size = int(img_size/2)
            if next_size < 16:
                next_size = 16
            if img_size >16: #16 como limite
                layers.append([img_size, next_size, img_vars])
                contador += 1
    else:
        n_layers = 3
        layers.append([128, 16, 1]) #2nda capa, la primera es 16x16, 16x16/2

    #last layer, porque si
    layers.append([16, 1, 1])
    #print("\n")
    #print (layers)
    #print(len(layers))
    return layers

def train_eval_epoch(epoch, model, optimizer, loss_fn, train_loader, val_loader, loss_stats):
    print("Training epoch...")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    #print(f"Training on device {device}.")

    #for epoch in range(1, n_epochs + 1):
    '''TRAIN EPOCH SECTION!!!'''
    loss_train = 0.0
    model.to(device)
    model.train()

    #for data, target in train_loader:
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = data.shape[0]
        data = data.view(batch_size, -1)

        data = data.to(device)
        #print("DATA SHAAAAPE")
        #print(data.shape)

        target = target.unsqueeze(1)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = loss_fn(output, target)

        loss.backward()
        optimizer.step()

        #Computing accuracy
        #acc += accuracy(output, target, 0.10)
        loss_train += loss.item()

    #guardando las losses de train
    loss_stats['train'].append(loss_train/len(train_loader)) #for each epoch

    if epoch == 1 or epoch % 10 == 0:
        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch,
            loss_train / len(train_loader)))
        #print("Accuracy (within 0.10) on train data = %0.4f" % acc)
        #avg_acc = 100. * acc / len(train_loader.dataset)


    '''EVAL EPOCH SECTION!!!'''
    print("Validating epoch...")
    loss_val = 0.0
    with torch.no_grad(): #no backprop en validation, ergo, no hay optimizador
        model.eval()  # Optional creo
        for batch_idx, (data, target) in enumerate(val_loader):
            batch_size = data.shape[0]
            data = data.view(batch_size, -1)
            data = data.to(device)
            target = target.unsqueeze(1)
            target = target.to(device)
            output = model(data)
            loss = loss_fn(output, target)

            loss_val += loss.item()

    #guardando las losses de val
    #print("loss_stats")
    #print(loss_stats)
    loss_stats['val'].append(loss_val/len(val_loader)) #for each epoch

    if epoch == 1 or epoch % 10 == 0:
        print('{} Epoch {}, Validation loss {}'.format(
            datetime.datetime.now(), epoch,
            loss_val / len(val_loader)))
        print("\n")

    #stats.save_stats(model.state_dict(), loss_stats['val'][-1])
    #guardamos el modelo con los parametros q dan la loss mas baja. Nse si podemos solamente tomar esta decision... Los dias con 0 lluvia afectan mucho creo
    resp = check_loss_list(loss_stats['val'], loss_val/len(val_loader))
    if epoch == 1:
        save_model(model, "./mlp_model.pth") #cuidado, el dir esta alterado por RayTune. La ruta relativa es en funcion de RayTune
    elif resp==True:
        save_model(model, "./mlp_model.pth")

    #viene a ser un append a tune de la loss que hemos obtenido para que despues analizamos los resultados
    #importante hacerlo lo ultimo del loop, porque sino en la ultima iteracion de todas, raytune termina donde hace el report
    tune.report(val_loss=loss_stats['val'][-1])
    return loss_stats

def test_epoch(epoch, model, loss_fn, test_loader, loss_stats):
    '''Test/evaluation loop. Testing with data not seen'''
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    #print(f"Training on device {device}.")
    model.to(device)
    model.eval()
    loss_test = 0
    epoch_loss = 0
    #print("----------------------------------")
    #print(len(test_loader))
    with torch.no_grad():
        loss_train = 0.0
        for batch_idx, (data, target) in enumerate(test_loader):
            batch_size = data.shape[0]
            data = data.view(batch_size, -1)
            #print("In test looop!!!")
            #print(data.shape)
            #print(data[0][0])
            #print(data[0][2000])
            target = target.unsqueeze(1)
            #print(target.shape)
            #print(target[10][0])
            data, target = data.to(device), target.to(device)
            #print("target:")
            #print(target)
            output = model(data)
            #print("output")
            #print(output)
            loss_test += loss_fn(output, target)
            #print("loss")
            #print(loss_test)
            #loss_test += loss.item()

    loss_stats['test'].append(loss_test/len(test_loader)) #for each epoch

    if epoch == 1 or epoch % 10 == 0:
        print('{} Epoch {}, Test loss {}'.format(
            datetime.datetime.now(), epoch,
            loss_test / len(test_loader)))
        #print("\n")

    return loss_stats

def test(config, model, test_set, n_epochs):
    print("Creating the dataloader...")
    loss_stats = {
        "train": [],
        "val": [],
        "test":[]
    }

    #haciendo el crop de la imagen (matriz), con la teorica mejor img_size entrenada
    transform = transforms.Compose([
        # center-crop
        transforms.CenterCrop(config['img_size']),
    ])
    #x_test = sets[2]
    #y_test = sets[5]
    #testset
    #testset = Reanalysisdata(x_test, y_test, transform)
    #print(x_test.shape, y_test.shape)

    test_set.dataset.transform = transform
    test_loader = torch.utils.data.DataLoader(test_set,
                                                batch_size=config['batch_size'],
                                                shuffle=True)

    loss_fn = nn.MSELoss() #MSE is the default loss function for most Pytorch regression problems.

    print("Test epochs loop...")
    for epoch in range(1, n_epochs + 1):
        loss_stats = test_epoch(epoch, model, loss_fn, test_loader, loss_stats)

    plots_learning_curves("test", loss_stats)

def train(config, train_set=None, val_set=None):
    '''CREATING DATALOADER'''
    loss_stats = {
        "train": [],
        "val": [],
        "test":[]
    }

    print("Creating the dataloader...")
    #lo hago dentro del train con la config de Ray porque consideramos que el tamaño de la imagen es un hyperparametro. Asi puede ser variable
    transform = transforms.Compose([
        # center-crop
        transforms.CenterCrop(config['img_size']),
    ])
    train_set.dataset.transform = transform
    val_set.dataset.transform = transform
    train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=config['batch_size'],
                                                shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,
                                            batch_size=config['batch_size'],
                                            shuffle=True)


    #esta create_dataloader function ya no funciona
    #train_loader, val_loader = create_dataloader(config, sets) #solo creo train y val loader porque el test_loader lo creare a partir de resultados de la "mejor" config entrenada

    layers = layers_config(config['img_size'], config['img_vars'])
    #model = Net(layers) -> no lo utilizaremos de momento
    model = BasicMlp(layers)
    print(model)

    optimizer = optim.SGD(model.parameters(), config['lr'], weight_decay=config["l2"])
    loss_fn = nn.MSELoss() #MSE is the default loss function for most Pytorch regression problems.

    print("\nInitializing epochs loop...")
    for epoch in range(1, config['n_epochs'] + 1):
        loss_stats = train_eval_epoch(epoch, model, optimizer, loss_fn, train_loader, val_loader, loss_stats)
        #eval_epoch(epoch, model, optim, loss_fn, val_loader) - no era del todo practica separarlo en 2 funciones train y val

    plots_learning_curves("train_val", loss_stats)

def main():
    '''LOADING HYPERPARAMETER CONFIG'''
    #primero borro los ficheros anteriores que ha creado rayTune, ya que no nos interesan mucho al ser pruebas y ocupan espacio
    if os.path.isdir('./outs_mlp_train/'):
        shutil.rmtree('./outs_mlp_train/') #solo comentar si no queremos que borre...

    print("\nLoading the HYPERPARAMETER CONFIG...")
    #config_owner = "maiol" config_type = "mlp" config_search = "other" #grid_search, random_search, y other
    max_n_epochs, reporter, config = load_hyper_conf(conf_vars['config_owner'], conf_vars['config_type'], conf_vars['config_search'])

    '''LOADING THE DATA'''
    print("\nLoading the data...")
    #sets = load_data(maiol_conf_vars['DATA_DIR'])
    data_dir = os.path.abspath(conf_vars['DATA_DIR'])
    dataset = Downscaling_dataset(data_dir)
    print(len(dataset))
    train_set, test_set, val_set = torch.utils.data.random_split(dataset, [0.7,0.2,0.1])
    print(len(train_set))
    print(len(test_set))
    print(len(val_set))

    '''INITIALIZING RAY TUNE ETC'''
    print("\nInitializing ray tune with config to train...")
    ray.init(configure_logging=False, include_dashboard=False, num_gpus=1) #creo que no hace nada...
    results = tune.run(
            partial(train, train_set=train_set, val_set=val_set),
            config=config, #definido en el conf del hyperparameter_config
            num_samples=conf_vars['NUM_SAMPLES'], #importante este valor sobretodo si es random_search
            metric="val_loss", #metrica para obtener la mejor config, y el mode seguidamente
            mode="min",
            progress_reporter=reporter,  #definido en el conf del hyperparameter_config
            local_dir='./outs_mlp_train/', #donde se van a guardar los outputs etc
            resources_per_trial={"cpu": conf_vars['cpu'], "gpu": conf_vars['gpu']} #por la GPU!!!
        )

    ''' ----------- TRAINING RESULTS!! ----------------'''
    best_trial = results.get_best_trial("val_loss", "min") #la metrica y el min definidos antes
    #print(best_trial)
    print("Best config: {}".format(best_trial.config))
    best_config = best_trial.config
    f = open("./best_config.txt", "a")
    f.write(str(results.best_logdir)+"\n")
    f.write(str(best_config)+"\n\n")
    f.close()
    #print("Best validation loss: {}".format(best_trial.last_result["val_loss"]))
    dir_best_results = results.best_logdir #lo utilizaremos para cargar el modelo guardado
    dir_best_results_model = dir_best_results+"/mlp_model.pth"
    #print(dir_best_results_model)
    #Best trial config: {'img_size': 64, 'img_vars': 1, 'train_batch_size': 64, 'val_batch_size': 64, 'n_epochs': 10, 'lr': 0.011850997181721131}
    #Best trial final validation loss: 13.202382882436117


    '''-------------- TESTING BEST RESUUUUUULTS------------'''
    '''we select the best model with an optimal combination of hyperparameters'''
    '''creamos el modelo con los mejores hyperparameters'''
    #volvemos a llamar la función para obtener la config del model a partir de la mejor config
    layers = layers_config(best_config['img_size'], best_config['img_vars'])
    best_trained_model = BasicMlp(layers)
    best_trained_model.to(device)

    '''cargamos el modelo guardado con sus weights'''
    #model_weights
    model_state = torch.load(dir_best_results+"/mlp_model.pth")
    best_trained_model.load_state_dict(model_state)
    print("Best trained model:")
    print(best_trained_model)

    '''lo testeamos'''
    test(best_config, best_trained_model, test_set, best_config['n_epochs'])
    '''---------------------------------------------'''

if __name__ == "__main__":
    main()
