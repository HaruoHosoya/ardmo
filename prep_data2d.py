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

import datasets2d as D

#%%

datasets_root = 'datasets'

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_root', type=str, default=datasets_root, help="dataset root path")

opts = parser.parse_args()

#%%

ds = D.Dataset(data_size=1000, 
               env_size=3,
               input_dim=700, 
               input_type='cifar100')

path = 'ds2d_cifar_{:d}_{:d}_{:d}.pt'.format(ds.data_size, ds.env_size, ds.input_dim)
torch.save(ds, osp.join(opts.datasets_root, path))

#%%

ds = D.Dataset(data_size=1000, 
               env_size=4,
               input_dim=700, 
               input_type='cifar100')

path = 'ds2d_cifar_{:d}_{:d}_{:d}.pt'.format(ds.data_size, ds.env_size, ds.input_dim)
torch.save(ds, osp.join(opts.datasets_root, path))

#%%

ds = D.Dataset(data_size=1000, 
               env_size=5,
               input_dim=700, 
               input_type='cifar100')

path = 'ds2d_cifar_{:d}_{:d}_{:d}.pt'.format(ds.data_size, ds.env_size, ds.input_dim)
torch.save(ds, osp.join(opts.datasets_root, path))

#%%

ds = D.Dataset(data_size=1000, 
               env_size=6,
               input_dim=700, 
               input_type='cifar100')

path = 'ds2d_cifar_{:d}_{:d}_{:d}.pt'.format(ds.data_size, ds.env_size, ds.input_dim)
torch.save(ds, osp.join(opts.datasets_root, path))

