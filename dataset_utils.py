#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:59:26 2023

A python implementation of ARDMO (Abstract Relational Decision-making MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""

import numpy as np
import random
import torch
import torchvision
from torchvision import transforms
import os
import os.path as osp

import params

def load_cifar100(input_dim):
    root_path = osp.join(params.downloaded_datasets_root, 'cifar100')
    ds = torchvision.datasets.CIFAR100(root=root_path, train=True, download=True)
    images = torch.tensor(ds.data).float() / 255
    targets = ds.targets
    nimages, xsize, ysize, zsize = images.size()
    image_dim = xsize * ysize * zsize
    A = images.reshape(nimages, image_dim)
    U, S, V = torch.pca_lowrank(A, q=input_dim)
    U_norm = U.norm(dim=1)
    S = S * U_norm.mean()
    U = U / U_norm.unsqueeze(1)
    image_center = images.mean(dim=0)
    image_basis = (V * S.unsqueeze(0)).reshape(xsize, ysize, zsize, input_dim)
    
    image_pcs_dict = dict()
    for i in range(len(ds)):
        try:
            image_pcs_dict[targets[i]].append(U[i])
        except KeyError:
            image_pcs_dict[targets[i]] = [U[i]]
    for k in image_pcs_dict.keys():
        image_pcs_dict[k] = torch.stack(image_pcs_dict[k], dim=0)
    image_pcs_dict = image_pcs_dict
    
    return image_pcs_dict, image_center, image_basis

