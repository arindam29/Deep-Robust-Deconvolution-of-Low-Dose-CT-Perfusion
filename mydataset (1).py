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

class myDataset(Dataset):
    def __init__(self, images, labels, do_trn):
        # print(images.shape, labels.shape)
        
        self.X = images
        self.y = labels
        self.do_trn = do_trn
         
        
    def my_transform(self, src, tgt, angle):
        if self.do_trn == True:
            # if torch.rand(1) < 0.5:
            #     src = TF.hflip(src)
            #     tgt = TF.hflip(tgt)
            src = TF.affine(src, angle, scale=1., translate=(0,0), shear=(0,0), resample=Image.BILINEAR)
            tgt = TF.affine(tgt, angle, scale=1., translate=(0,0), shear=(0,0), resample=Image.BILINEAR)


        src = TF.to_tensor(np.array(src)).type(torch.FloatTensor)
        tgt = TF.to_tensor(np.array(tgt)).type(torch.FloatTensor)
        # print(src.shape, tgt.shape)

        return src, tgt

    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        datax = self.X[i, :]
        datay = self.y[i, :]
        
        
        
        # print(datax.shape, datay.shape)
        datax_new = np.zeros_like(datax)
        datay_new = np.zeros_like(datay)    #write dtype
        degrees = (-25,25)
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        
        for i in range(datax.shape[0]):
            datax_new[i,:,:], datay_new[0,:,:] = self.my_transform(Image.fromarray(datax[i,:,:]), Image.fromarray(datay[0,:,:]), angle)

        return datax_new, datay_new
        # return datax, datay
 
