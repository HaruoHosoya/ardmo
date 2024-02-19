#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:17:58 2022

A python implementation of ARDMO (Abstract Relational Decision-making MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""


import os as os
import os.path as osp

import torch
import argparse

import datasets1d as D
import params

#%%

datasets_root = 'datasets'

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_root', type=str, default=datasets_root, help="dataset root path")

opts = parser.parse_args()

### CIFAR100 images
#%%

ds = D.Dataset(data_size=1000, 
               env_size=3, 
               input_dim=700, 
               relation_dim=2,
               input_type='cifar100')

path = 'ds1d_cifar_{:d}_{:d}_{:d}.pt'.format(ds.data_size, ds.env_size, ds.input_dim)
torch.save(ds, osp.join(opts.datasets_root, path))

#%%

ds = D.Dataset(data_size=1000, 
               env_size=5, 
               input_dim=700, 
               relation_dim=2,
               input_type='cifar100')

path = 'ds1d_cifar_{:d}_{:d}_{:d}.pt'.format(ds.data_size, ds.env_size, ds.input_dim)
torch.save(ds, osp.join(opts.datasets_root, path))

#%%

ds = D.Dataset(data_size=1000, 
               env_size=7, 
               input_dim=700, 
               relation_dim=2,
               input_type='cifar100')

path = 'ds1d_cifar_{:d}_{:d}_{:d}.pt'.format(ds.data_size, ds.env_size, ds.input_dim)
torch.save(ds, osp.join(opts.datasets_root, path))

#%%

ds = D.Dataset(data_size=1000, 
               env_size=10, 
               input_dim=700, 
               relation_dim=2,
               input_type='cifar100')

path = 'ds1d_cifar_{:d}_{:d}_{:d}.pt'.format(ds.data_size, ds.env_size, ds.input_dim)
torch.save(ds, osp.join(opts.datasets_root, path))




