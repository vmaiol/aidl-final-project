import torch

#constants
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
confif = {}
