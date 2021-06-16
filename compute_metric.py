#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 21:01:22 2021

@author: aridam
"""

import torch 
import torch.nn as nn
from config import Config
from pytorch_msssim import ssim





        
def compute_ssim(out_images,target_images,val_mask):
    
    out_images = torch.mul(out_images,val_mask)
    
    ssim_val = []
    
    for i in range(out_images.size(0)):
        
        output_i = out_images[i,:,:,:]
        target_i = target_images[i,:,:,:]
        
        output_i = output_i.unsqueeze(0)
        target_i = target_i.unsqueeze(0)
        
        output_i = torch.mul(torch.div(output_i,torch.max(output_i)),100)
        target_i = torch.mul(torch.div(target_i,torch.max(target_i)),100)
        
        
        ssim_value = ssim(output_i, target_i, data_range=100, size_average=False, nonnegative_ssim = True)
        
        ssim_val.append(ssim_value)
        
    
    ssim_val = torch.stack(ssim_val)
    
        
    return torch.mean(ssim_val)


