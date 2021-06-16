#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:26:11 2020

@author: mig-arindam
"""


import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import torchvision.transforms.functional as TF 
from PIL import Image
import torchvision.transforms as transforms
 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



# torchvision.transforms.RandomAffine(degree) 

class myDataset_val(Dataset):
    def __init__(self, images, labels, masks):
        # print(images.shape, labels.shape)
        
        self.X = images
        self.y = labels
        self.z = masks

        
        

         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        datax = self.X[i, :]
        datay = self.y[i,:]
        dataz = self.z[i,:]

            
        return datax, datay, dataz
  

       
 

 
