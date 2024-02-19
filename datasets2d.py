#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:06:51 2022

A python implementation of ARDMO (Abstract Relational Decision-making MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""

import numpy as np
import random
import torch
import torch.utils.data
from torch.nn.functional import one_hot
import torchvision
from torchvision import transforms
import os
import os.path as osp
from tqdm import tqdm

import params
import dataset_utils

class World():
    def __init__(self, dataset, split='train', shuffle=False, random_reward=False):
        self.dataset = dataset
        self.env_size = dataset.env_size
        self.relation_dim = self.env_size * 2 - 1
        self.random_reward = random_reward

        self.split = split
        if split == 'train':
            self.env_indices_orig = torch.tensor(self.dataset.train_indices)
        elif split == 'val':
            self.env_indices_orig = torch.tensor(self.dataset.val_indices)
        else:
            self.env_indices_orig = torch.tensor(self.dataset.test_indices)
        self.shuffle = shuffle
        self.init_env_series()
        
    def init_env_series(self):
        if self.shuffle:
            self.env_indices = self.env_indices_orig[torch.randperm(len(self.env_indices_orig))]
        else:
            self.env_indices = self.env_indices_orig
        self.env_series_pos = 0
        
    def next_env(self, batch_size):
        self.batch_size = batch_size
        self.envi = self.env_indices[torch.arange(self.env_series_pos, self.env_series_pos + batch_size)]
        self.env_series_pos += batch_size
        if self.env_series_pos + batch_size >= len(self.env_indices):
            self.init_env_series()
            
    def get_data(self, rel_axis):
        # rel: batch x relation (x or y)
        # pos: batch x coordinate (x,y) x stimulus_numbers (0,1)
        rel = torch.randint(2, (self.batch_size, ))
        pos = torch.zeros(self.batch_size, 2, 2, dtype=torch.long)
        inp1 = torch.zeros(self.batch_size, self.dataset.input_dim)
        inp2 = torch.zeros(self.batch_size, self.dataset.input_dim)

        for i in range(self.batch_size):
            env = self.dataset.env_data[self.envi[i]]
            p = torch.randint(self.env_size-1, (1,))
            p2 = [p+1,p] if rel[i] == 0 else [p,p+1]
            if rel_axis == 0:
                pos[i,0,:] = torch.tensor(p2)
                pos[i,1,:] = torch.randint(self.env_size, (2,))
            else:
                pos[i,0,:] = torch.randint(self.env_size, (2,))
                pos[i,1,:] = torch.tensor(p2)
            inp1[i,:] = env[pos[i,0,0], pos[i,1,0]]
            inp2[i,:] = env[pos[i,0,1], pos[i,1,1]]
                    
        def reward_func(given_rel):
            if self.random_reward:
                return torch.randint(2, (self.batch_size,)).float()
            else:
                rb = [ torch.equal(given_rel[b], rel[b]) for b in range(self.batch_size) ]
                r = torch.tensor(rb).float()
                return r
        
        truth = rel, pos
        
        return inp1, inp2, reward_func, truth
    
    def get_test_data(self, rel_axis):
        rel = torch.zeros(self.batch_size, dtype=torch.long)
        pos = torch.zeros(self.batch_size, 2, 2, dtype=torch.long)
        inp1 = torch.zeros(self.batch_size, self.dataset.input_dim)
        inp2 = torch.zeros(self.batch_size, self.dataset.input_dim)
        for i in range(self.batch_size):
            env = self.dataset.env_data[self.envi[i]]
            if rel_axis == 0:
                pos[i,0,:] = torch.randperm(self.dataset.env_size)[:2]
                pos[i,1,:] = torch.randi(self.dataset.env_size, (2,))
                rel[i] = 0 if pos[i,0,0] < pos[i,0,1] else 1
            else:
                pos[i,0,:] = torch.randi(self.dataset.env_size, (2,))
                pos[i,1,:] = torch.randperm(self.dataset.env_size)[:2]
                rel[i] = 0 if pos[i,1,0] < pos[i,1,1] else 1
            inp1[i,:] = env[pos[i,0,0], pos[i,1,0]]
            inp2[i,:] = env[pos[i,0,1], pos[i,1,1]]
                    
        def reward_func(given_rel):
            rb = [ torch.equal(given_rel[b], rel[b]) for b in range(self.batch_size) ]
            r = torch.tensor(rb).float()
            return r
        
        truth = rel, pos
        
        return inp1, inp2, reward_func, truth


    
class Dataset():
    def __init__(self, data_size, env_size, input_dim, 
                 input_type='cifar100', train_ratio=0.6, val_ratio=0.2):
    
        self.data_size = data_size
        self.env_size = env_size
        self.input_dim = input_dim
        self.input_type = input_type

        # generate environment data
        
        self.env_data = []
        self.env_class_data = []
        
        if input_type == 'cifar100':
            self.image_pcs_dict, self.image_basis, self.image_center = \
                dataset_utils.load_cifar100(self.input_dim)        
        
        for i in range(data_size):
            if input_type == 'cifar100':
                env_class = torch.randint(100, (1,)).item()
                image_pcs = self.image_pcs_dict[env_class]
                env = image_pcs[torch.randperm(image_pcs.size(0))[:env_size**2]]
                env = env.reshape(env_size, env_size, self.input_dim)
            else:
                raise Exception('unsupported input type:', input_type)

            self.env_data.append(env)
            self.env_class_data.append(env_class)
            
        if input_type == 'cifar100': 
            nclass = 100
            ntrain_class = int(nclass * train_ratio)
            nval_class = int(nclass * val_ratio)
            self.train_indices = list(filter(lambda i:self.env_class_data[i] < ntrain_class, range(data_size)))
            self.val_indices = list(filter(lambda i:self.env_class_data[i] >= ntrain_class and self.env_class_data[i] < ntrain_class+nval_class, range(data_size)))
            self.test_indices = list(filter(lambda i:self.env_class_data[i] >= ntrain_class+nval_class, range(data_size)))
        else:
            ntrain = int(data_size * train_ratio)
            nval = int(data_size * val_ratio)
            self.train_indices = list(range(0, ntrain))
            self.val_indices = list(range(ntrain, ntrain+nval))
            self.test_indices = list(range(ntrain+nval, data_size))

    def generate_image(self, pcs):
        if pcs.ndim == 1:
            pcs = pcs.unsqueeze(1)
        imgs = torch.matmul(self.image_basis, pcs.permute(1, 0)) + self.image_center.unsqueeze(3)
        return imgs.permute(3, 2, 0, 1)
            
        
        