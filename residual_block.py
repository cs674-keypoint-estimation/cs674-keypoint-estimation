import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import numpy as np
import einops
from torch.autograd import Variable


#INSTANCE NORM:

class residual_block_instance_norm(nn.Module):
    def __init__(self, in_, out_):
        super(residual_block_instance_norm, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_, out_channels=in_,kernel_size=1)
        self.instant_norm1 = nn.InstanceNorm1d(num_features=in_)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=in_, out_channels=out_,kernel_size=1)
        self.instant_norm2 = nn.InstanceNorm1d(num_features=out_)

    def forward(self, input):
        x1 = self.conv1(input)
        x1 = self.instant_norm1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(input)
        x2 = self.instant_norm2(x2)

        x3 = self.conv2(x1)
        x3 = self.instant_norm2(x3)

        x3 += x2

        x = self.relu1(x3)


#GROUP NORM:

class residual_block_group_norm(nn.Module):
    def __init__(self, in_, out_):
        super(residual_block_group_norm, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_, out_channels=in_,kernel_size=1)
        self.instant_norm1 = nn.GroupNorm(num_features=in_)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=in_, out_channels=out_,kernel_size=1)
        self.instant_norm2 = nn.GroupNorm(num_features=out_)

    def forward(self, input):
        x1 = self.conv1(input)
        x1 = self.instant_norm1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(input)
        x2 = self.instant_norm2(x2)

        x3 = self.conv2(x1)
        x3 = self.instant_norm2(x3)

        x3 += x2

        x = self.relu1(x3)


        return x

#LAYER NORM:

class residual_block_layer_norm(nn.Module):
    def __init__(self, in_, out_):
        super(residual_block_group_norm, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_, out_channels=in_,kernel_size=1)
        self.instant_norm1 = nn.GroupNorm(num_features=in_)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=in_, out_channels=out_,kernel_size=1)
        self.instant_norm2 = nn.GroupNorm(num_features=out_)

    def forward(self, input):
        x1 = self.conv1(input)
        x1 = self.instant_norm1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(input)
        x2 = self.instant_norm2(x2)

        x3 = self.conv2(x1)
        x3 = self.instant_norm2(x3)

        x3 += x2

        x = self.relu1(x3)


        return x