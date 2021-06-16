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
 
 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset





class myDataset(Dataset):
    def __init__(self, images):

            
        self.X = images

         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        datax = self.X[i, :]

            
        return datax
  

       
 
