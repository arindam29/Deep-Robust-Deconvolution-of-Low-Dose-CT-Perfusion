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
#os,time
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
from datetime import datetime
from config_test import Config
from mydataset import myDataset
import h5py as h5
from model_cbf import DCNet
from archs import NestedUNet
from perceptual import perceptual

#%%
print ('*************************************************')

config  = Config()


# load the testing data
testimages = h5.File(config.testdirec+'data/'  +'Vn_test.h5','r')['Vn_test'].value
testlabels = h5.File(config.testdirec+'labels/'+'cbf_test.h5','r')['cbf_test'].value


# load the new testing data
#testimages = h5.File(config.testdirec+'data/'  +'Vn_test_new.h5','r')['Vn_test_new'].value
#testlabels = h5.File(config.testdirec+'labels/'+'cbf_test_new.h5','r')['cbf_test_new'].value


transform = transforms.Compose([transforms.ToTensor()])


# make the data iterator for testing data
test_data = myDataset(testimages, testlabels, transform, False)
testloader  = torch.utils.data.DataLoader(test_data, config.batchsize, shuffle=False, num_workers=2)

print('###########################')
#%%
#Create the object for the network
if config.gpu == True:
    net = DCNet()
    net.cuda(config.gpuid)
    par = torch.nn.DataParallel(net, device_ids = [0,1])
else:
      net = DCNet()



        
    
#%%
net.load_state_dict(torch.load('')) #location of epoch

net = net.eval()

Loss_t = []
for i,data in enumerate(testloader): 
    # start iterations
    images,labels = data[0],data[1]

    images = images.permute((0,2,1,3))
    labels = labels.permute((0,2,1,3))
    
    labels[labels<=0] = 0
    
    labels = labels[:,0,:,:].unsqueeze(dim=1)
    
        # ckeck if gpu is available
    
    # ckeck if gpu is availabl
    if config.gpu == True:
          images = images.cuda(config.gpuid)
          labels = labels.cuda(config.gpuid)
    # make forward pass      
    output_test = par(images)

    output_test = torch.div(output_test,torch.max(output_test))
    output_test[output_test < 0] = 0
    output_test = torch.div(output_test,0.01)
    



test_residue = output_test.cpu()
test_residue = test_residue.detach().numpy()

from numpy import save
save('test_residue.npy',test_residue)

#%%


from scipy.io import savemat

Residue_test = np.load('test_residue.npy')


adict = {}
adict['test_residue'] = Residue_test
savemat('test_residue.mat',adict)
#%%
print('done!!!')
#%%



