
import torch 
import torch.nn as nn 
import numpy as np 
from torch.utils.data import Dataset
import glob
import cv2
import os 
from torch.autograd import Variable


def return_conv_block(input_size, output_size, stride_conv=1, stride_pool=2, kernel_size_conv=3, kernel_size_pool=2):
    return nn.Sequential(
        nn.Conv2d(input_size, output_size, kernel_size=(kernel_size_conv, kernel_size_conv), stride=stride_conv),
        nn.BatchNorm2d(output_size),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=kernel_size_pool, stride=2 )
    )
        


class CNN(nn.Module):
    """
        This implements basic cnn module
    """
    def __init__(self, num_classes = 2):
        super().__init__()
        self.conv1 = return_conv_block(input_size=3, output_size=20)
        self.conv2 = return_conv_block(input_size=20, output_size=40)
        self.conv3 = return_conv_block(input_size=40, output_size=80)
        self.fc = nn.Linear(23 * 23 * 80, num_classes)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1) # Flatten the output for linear layer 
        return self.fc(out)


