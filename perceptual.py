#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 10:09:52 2020

@author: aridam
"""

import torch 
import torch.nn as nn
from torchvision.models.vgg import vgg16
from config import Config



class perceptual(nn.Module):
    def __init__(self):
        super(perceptual, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        config = Config()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network.cuda(config.gpuid)        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, out_images, target_images):        
        out = torch.cat((out_images,out_images,out_images),dim=1)
        tar = torch.cat((target_images,target_images,target_images),dim=1)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out), self.loss_network(tar))
        return perception_loss 