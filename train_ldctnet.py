#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:07:48 2020

@author: mig-arindam
"""
import os
abspath = os.path.abspath('/media/cds/storage/DATA-1/ARIDAM') ## String which contains absolute path to the script file
os.chdir(abspath) ## Setting up working directory
#%%
import time
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
from datetime import datetime
from config import Config
from mydataset import myDataset
import h5py as h5
from model_cbf import DCNet
from archs import NestedUNet

from perceptual import perceptual
config  = Config()
print ('*************************************************')
#%%
start_time = time.time()
saveDir = 'savedModels/'
cwd = os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%P_")+'model_parameters'
if not os.path.exists(directory): 
    os.makedirs(directory)



# load the training data
trainimages = h5.File(config.traindire+'data/'  +'Vn_train.h5','r')['Vn_train'].value
trainlabels = h5.File(config.traindire+'labels/'+'cbf_train.h5','r')['cbf_train'].value

# load the validation data
valiimages = h5.File(config.validdire+'data/'  +'Vn_val.h5','r')['Vn_val'].value
valilabels = h5.File(config.validdire+'labels/'+'cbf_val.h5','r')['cbf_val'].value




transform = transforms.ToTensor()

# make the data iterator for training data
train_data = myDataset(trainimages, trainlabels, transform, False)
trainloader = torch.utils.data.DataLoader(train_data, config.batchsize, shuffle=False, num_workers=2)


# make the data iterator for validation data

valid_data = myDataset(valiimages, valilabels, transform, False)
validloader = torch.utils.data.DataLoader(valid_data, config.batchsize, shuffle=False, num_workers=2)








#%% for LDCT-Net
#Create the object for the network

if config.gpu == True:
    net = DCNet()
    net.cuda(config.gpuid)
else:
    net = DCNet()
#%% for LDCT-Net++
# if config.gpu == True:
#     net = NestedUNet(num_classes =1)
#     net.cuda(config.gpuid)
# else:
#     net = NestedUNet(num_classes=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

                                  
parameters = count_parameters(net)
print(parameters)
#%%



optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [5,50,100,150], gamma=0.1, last_epoch=-1)

# Define the loss function
criterion = nn.MSELoss()
perc_loss = perceptual()


a = 1 			# weight for mse loss
b = 0.0003		# weight for L2-Perceptual


train_Loss = np.zeros(config.epochs, dtype = float)
val_Loss = np.zeros(config.epochs, dtype = float)
#%%
print('#Start_Training#')
# Iterate over the training dataset

for j in range(config.epochs):
    #print(j)
    # Start epochs
   
    Loss = []
    net = net.train() 
    for i,data in enumerate(trainloader): 
        # start iterations
        images,labels = data[0],data[1]

        images = images.permute((0,2,1,3))
        labels = labels.permute((0,2,1,3))
        

        labels[labels<=0] = 0
        labels = labels[:,0,:,:].unsqueeze(dim=1)

        # ckeck if gpu is available
        if config.gpu == True:
              images = images.cuda(config.gpuid)
              labels = labels.cuda(config.gpuid)
              
        # make forward pass      
        output_train = net(images)
    
        #compute loss
        loss_mse   = criterion(output_train, labels)
        loss_perc  = perc_loss(output_train, labels)
        
        
        loss = a*loss_mse + b*loss_perc
 
        # make gradients zero
        optimizer.zero_grad()
        # back propagate
        loss.backward()
        Loss.append(loss.item())
        # update the parameters
        optimizer.step()
        #ema.update(net.parameters())
    # print loss after every epoch

    print('epoch {}, train_loss:{:.6f}'.format(j+1, np.mean(Loss)))
    train_Loss[j] = np.mean(Loss)
    
    net = net.eval() 
    
    Loss_v = []
    for i,data in enumerate(validloader): 
        # start iterations
        images,labels = data[0],data[1]

        images = images.permute((0,2,1,3))
        labels = labels.permute((0,2,1,3))
        labels[labels<=0] = 0
        labels = labels[:,0,:,:].unsqueeze(dim=1)
        # ckeck if gpu is available
        if config.gpu == True:
              images = images.cuda(config.gpuid)
              labels = labels.cuda(config.gpuid)
        # make forward pass      
        output_val = net(images)
        #compute loss
        loss_mse   = criterion(output_val, labels)
        loss_perc  = perc_loss(output_val, labels)
        
        loss = a*loss_mse + b*loss_perc
        Loss_v.append(loss.item())
        
        scheduler.step()
    # print loss after every epoch
        
   # save model parameters
      
    torch.save(net.state_dict(), os.path.join(directory, 'net_epoch_%d.pth' % (j+1)))   
    print('epoch {}, val_loss:{:.6f}'.format(j+1, np.mean(Loss_v)))
    val_Loss[j] = np.mean(Loss_v)
        
    


