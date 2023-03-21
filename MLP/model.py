import torch
import torch.nn as nn
from collections import OrderedDict

#models, so se utiliza
class MlpNet(nn.Module):
    #no lo utilizo
    def __init__(self, layers):
        super(MyModule, self).__init__()
        #in features es matrix height x width x 1 -> 1 porque de momento lo hacemos con solo precipitacion
        #p.ej, 64 x 64 x 1
        valor = in_features/2
        print(valor)
        self.input_size = in_features
        self.hidden_size  = hidden_size
        self.hid1 = nn.Linear(in_features, self.hidden_size)
        self.hid2 = nn.Linear(2048, 1024)
        self.hid3 = nn.Linear(1024, 512)
        #self.hid4 = nn.Linear(512, 256)
        #self.hid5 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)
        #self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.relu(self.hid1(x))
        z = self.relu(self.hid2(z))
        z = self.relu(self.hid3(z))
        #z = self.relu(self.hid4(z))
        #z = self.relu(self.hid5(z))
        #z = self.relu(self.hid6(z))
        z = self.out(z)  # no activation
        return z

#este es el BUENO
class BasicMlp(nn.Module):
    def __init__(self, layers):
        super().__init__()
        layers_dict = OrderedDict()
        idx=0
        idx_aux=1
        #dict con la estructura que tendra en Sequential
        for layer in layers:
            if idx_aux < len(layers):
                layers_dict[str(idx)] = nn.Linear(layer[0],layer[1])
                idx += 1
                layers_dict[str(idx)] = nn.ReLU()
            else:
                layers_dict[str(idx)] = nn.Linear(layer[0],layer[1]) #sin activacion fn para hacerlo lineal
            idx += 1
            idx_aux += 1

        self.mlp = nn.Sequential(layers_dict)

        #print(self)

    def forward(self, x):
        z = self.mlp(x) # no activation
        return z
