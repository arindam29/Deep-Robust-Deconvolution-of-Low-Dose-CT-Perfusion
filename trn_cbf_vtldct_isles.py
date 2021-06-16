#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:07:48 2020

@author: mig-arindam
"""
import os
abspath = os.path.abspath('/media/cds/storage/DATA-1/ARIDAM/VT_LDCT_net') ## String which contains absolute path to the script file
os.chdir(abspath) ## Setting up working directory

import time
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from mydataset_vali import myDataset_val
import torch.nn as nn
from datetime import datetime
from config import Config
from mydataset_combined import myDataset
import h5py as h5
from model_cbf_3d_2d_concat_isles import ldct_32d_net
from torch.nn.parallel import DistributedDataParallel
from compute_metric import compute_ssim
from perceptual import perceptual
config  = Config()
print ('*************************************************')
# combined dataset training
torch.manual_seed(5)

#%%


# load the training data
#trainimages = h5.File(config.traindire+'data/'  +'Vn_train.h5','r')['Vn_train'].value

#trainlabels = h5.File(config.traindire+'labels/'+'cbf_train.h5','r')['cbf_train'].value

traindata = h5.File(config.traindire + 'train_combined.h5', 'r')

# load the validation data
valiimages = h5.File(config.valdire+'data/'  +'Vn_val.h5','r')['Vn_val'].value

valilabels = h5.File(config.valdire+'labels/'+'cbf_val.h5','r')['cbf_val'].value

valimasks = h5.File(config.valdire+'masks/'+'Mask_val.h5','r')['Mask_val'].value


#%%

# train_data = myDataset(trainimages, trainlabels, True)
# trainloader = torch.utils.data.DataLoader(train_data, config.batchsize, shuffle=False, num_workers=1)
# print(len(trainloader))

# #%%

# img, label = next(iter(trainloader))
# print(img.shape)
# f, ax = plt.subplots(1,2, figsize=(9,5))
# ax[0].imshow(img[0][22,:,:], plt.cm.gray)
# ax[0].get_xaxis().set_visible(False)
# ax[0].get_yaxis().set_visible(False)
# ax[1].imshow(label[0][0,:,:])
# ax[1].get_xaxis().set_visible(False)
# ax[1].get_yaxis().set_visible(False)
# plt.ion()
# plt.show()


# one conv_middle block

 #%%
# make the data iterator for training data
#train_data = myDataset(trainimages, trainlabels, True)
#trainloader = torch.utils.data.DataLoader(train_data, config.batchsize, shuffle= True, num_workers=2)

train_data = myDataset(traindata, True, train = True)
trainloader = torch.utils.data.DataLoader(train_data, config.batchsize, shuffle=False, num_workers=2)


# make the data iterator for validation data

valid_data = myDataset_val(valiimages, valilabels, valimasks)
validloader = torch.utils.data.DataLoader(valid_data, config.batchsize, shuffle=False, num_workers=2)

# trained on patches after entrire training data is made 0 mean and unit std.


#%%
#Create the object for the network

if config.gpu == True:
    net = ldct_32d_net()
    net = net.cuda(config.gpuid)
    par = torch.nn.DataParallel(net, device_ids = [0,1])
else:
    net = ldct_32d_net()

#%%

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

                                  
parameters = count_parameters(net)
print(parameters)
#%%


optimizer = torch.optim.Adam(net.parameters(), lr=0.00075, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)


scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [25], gamma= 0.1, last_epoch=-1)
#scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [150,175], gamma= 0.1, last_epoch=-1)

#print('torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [40,75,100,150,200,250,300], gamma=0.1, last_epoch=-1)')
# Define the loss function
criterion = nn.MSELoss()
perc_loss = perceptual()
criterion_l1 = nn.L1Loss()

a = 1
b = 0.0025  #L1
c = 0
#%%
train_Loss = np.zeros(config.epochs, dtype = float)
val_Loss = np.zeros(config.epochs, dtype = float)
ssim_values =  np.zeros(config.epochs, dtype = float)
#%%
print('#Start_Training#')
start_time = time.time()
saveDir = 'savedModels/'
cwd = os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b%Y_%H%M_")+'model_parameters'
if not os.path.exists(directory): 
    os.makedirs(directory)
    
print(directory)
#%%


# Iterate over the training dataset
# torch.save(net.state_dict(), '/media/cds/storage/DATA-1/ARIDAM/directory/net_epoch_%d.pth' % (1))
for j in range(config.epochs):
    #print(j)
    # Start epochs
   
    Loss = []
    net = net.train() 
    for i,data in enumerate(trainloader): 
        # start iterations
        images,labels = data[0],data[1]
        images = images.unsqueeze(1)
        
        images = images.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        #labels = torch.max(labels, 1, True)[0]


        # ckeck if gpu is available
        if config.gpu == True:
              images = images.cuda(config.gpuid)
              labels = labels.cuda(config.gpuid)
              
        # make forward pass      
        output_train = par(images)

        #compute loss
        loss_mse   = criterion(output_train, labels)
        loss_perc  = perc_loss(output_train, labels)
        loss_l1 = criterion_l1(output_train, labels)
        
        loss = a*loss_mse + b*loss_perc + c*loss_l1
        #print(loss)
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
    ssim_v = []
    Loss_v = []
    for i,data in enumerate(validloader): 
        # start iterations
        images,labels,val_masks = data[0],data[1],data[2]
        
        images = images.unsqueeze(1)

        images = images.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        val_masks = val_masks.type(torch.FloatTensor)
        val_masks = torch.mean(val_masks,1).unsqueeze(1)
       
        # ckeck if gpu is available
        if config.gpu == True:
              images = images.cuda(config.gpuid)
              labels = labels.cuda(config.gpuid)
              val_masks = val_masks.cuda(config.gpuid)
        # make forward pass      
        output_val = par(images)


        # no normalization was done here!
        #compute loss
        loss_mse   = criterion(output_val, labels)
        loss_perc  = perc_loss(output_val, labels)
        loss_l1 = criterion_l1(output_val, labels)
        
        loss = a*loss_mse + b*loss_perc + c*loss_l1
        Loss_v.append(loss.item())
        ssim_val = compute_ssim(output_val,labels,val_masks)
        ssim_v.append(ssim_val.item())
        #ssim_v = []
    scheduler.step()
    # print loss after every epoch
    val_Loss[j] = np.mean(Loss_v)
    ssim_values[j] = np.mean(ssim_v)
   # save model parameters
    if (j>0 and ssim_values[j] > np.max(ssim_values[0:j])):  
        torch.save(net.state_dict(), os.path.join(directory, 'net_epoch_%d.pth' % (j+1)))   
    #print('epoch {}, val_loss:{:.6f}'.format(j+1, np.mean(Loss_v)))

    print('epoch {}, val_ssim:{:.6f}'.format(j+1, np.mean(ssim_v)))
#%%
abspath = os.path.abspath(directory) ## String which contains absolute path to the script file
os.chdir(abspath) ## Setting up working directory
#%%
from numpy import save
save('Val_ssim.npy',ssim_values)
save('Train_loss.npy',train_Loss)
x = np.argsort(ssim_values)
y = x+1
best_epoch = y[-1]

print('best epoch {}'.format(best_epoch))
np.save('best_epoch.npy',best_epoch)

#%%
plot1 = plt. figure(1)
plt.plot(np.arange(config.epochs), ssim_values)
plt.legend("Val_ssim")
plt.savefig('Val_ssim_curve.png')
plot2 = plt. figure(2)
plt.plot(np.arange(config.epochs), train_Loss)
plt.legend("Train_Loss")
plt.savefig('Train_Loss_curve.png')
#%%
import shutil

original = r'/media/cds/storage/DATA-1/ARIDAM/VT_LDCT_net/trn_cbf_3d_isles.py'
target = abspath

shutil.copy(original, target)

#%%
abspath = os.path.abspath('/media/cds/storage/DATA-1/ARIDAM/VT_LDCT_net') ## String which contains absolute path to the script file
os.chdir(abspath) ## Setting up working directory

