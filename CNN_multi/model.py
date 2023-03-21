import torch.nn as nn
import torch.nn.functional as F
import torch
class Downscaling_model1(nn.Module):
  def __init__(self):
    super(Downscaling_model1, self).__init__()
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16 , kernel_size = 3, stride = 2 , padding = 1) # out --> 
    self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size= 3, stride = 2, padding= 1) # out
    self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size= 3, stride = 2, padding= 1) # out
    self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size= 3, stride = 2, padding= 1) # out
    self.fc = nn.Linear(in_features=128,out_features = 1)
    self.maxpool = nn.MaxPool2d(kernel_size=2)
    self.dropout = nn.Dropout(0)

  def forward(self, x: torch.Tensor):
    out = self.maxpool(F.relu(self.dropout(self.conv1(x))))
    out = self.maxpool(F.relu(self.dropout(self.conv2(out))))
    out = self.maxpool(F.relu(self.dropout(self.conv3(out))))
    out = self.maxpool(F.relu(self.dropout(self.conv4(out))))
    bsz,chn,w,h = out.shape
    out = out.view(bsz,-1)
    out = self.fc(out)

    return out

class Downscaling_model2(nn.Module):
  def __init__(self):
    super(Downscaling_model2, self).__init__()
    self.conv1 = nn.Conv2d(in_channels = 8, out_channels = 16 , kernel_size = 2, stride = 2 , padding = 1) # out --> 
    self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size= 2, stride = 2, padding= 1) # out
    self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size= 2, stride = 2, padding= 1) # out
    self.fc = nn.Linear(in_features=576,out_features = 1)
    self.maxpool = nn.MaxPool2d(kernel_size=2)
    self.dropout = nn.Dropout(0)

  def forward(self, x: torch.Tensor):
    out = self.maxpool(F.relu(self.dropout(self.conv1(x))))
    out = self.maxpool(F.relu(self.dropout(self.conv2(out))))
    out = self.maxpool(F.relu(self.dropout(self.conv3(out))))
    bsz,chn,w,h = out.shape
    out = out.view(bsz,-1)
    out = self.fc(out)

    return out