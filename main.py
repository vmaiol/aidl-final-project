import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import matplotlib.pyplot as plt

from dataset import NameDatasetClass
from model import NameModelClass
from utils import accuracy, save_model
from conf import *

def load_data():
    #torch.utils.data.DataLoader...
    pass

def train_single_epoch():
    pass

def eval_single_epoch():
    pass

def train_model():
    #loop
    #train_single_epoch(...)
    #eval_single_epoch(...)
    pass

def test_model():
    pass

def main():
    #train_ds, val_ds, test_ds = load_data()
    #...
    print("Basic project structure loaded!!")

if __name__ == "__main__":
    main()
