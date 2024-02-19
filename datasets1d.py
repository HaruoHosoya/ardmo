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
    def __init__(self, dataset, split='train', shuffle=False, rand_nonadj=False, tokenize=False, walk_around=False):
        self.dataset = dataset
        self.split = split
        if split == 'train':
            self.env_indices_orig = torch.tensor(self.dataset.train_indices)
        elif split == 'val':
            self.env_indices_orig = torch.tensor(self.dataset.val_indices)
        else:
            self.env_indices_orig = torch.tensor(self.dataset.test_indices)
        self.shuffle = shuffle
        self.rand_nonadj = rand_nonadj
        self.tokenize = tokenize
        self.walk_around = walk_around
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
        if self.walk_around:
            self.last_pos = torch.randint(self.dataset.env_size, (self.batch_size,))
        if self.env_series_pos + batch_size >= len(self.env_indices):
            self.init_env_series()
            
    def get_data(self):
        if self.tokenize:
            inp1 = torch.zeros(self.batch_size, 1, dtype=torch.long)
            inp2 = torch.zeros(self.batch_size, 1, dtype=torch.long)
        else:
            inp1 = torch.zeros(self.batch_size, self.dataset.input_dim)
            inp2 = torch.zeros(self.batch_size, self.dataset.input_dim)
        pos = torch.zeros(self.batch_size, 2, dtype=torch.long)
        if self.walk_around:
            rel = torch.zeros(self.batch_size, dtype=torch.long)
            for i in range(self.batch_size):
                x1 = self.last_pos[i]                
                rel_ = torch.randint(self.dataset.relation_dim, (1,))
                if x1 == self.dataset.env_size-1: 
                    rel_ = 0
                elif x1 == 0:
                    rel_ = 1
                if rel_ == 0:
                    x2 = x1 - 1
                elif rel_ == 1:
                    x2 = x1 + 1
                rel[i] = rel_
                pos[i,0] = x1
                pos[i,1] = x2
                self.last_pos[i] = x2
        else:
            rel = torch.randint(self.dataset.relation_dim, (self.batch_size, ))
            for i in range(self.batch_size):
                if rel[i] == 0:
                    x1 = torch.randint(self.dataset.env_size-1, (1,)) + 1
                    x2 = x1 - 1
                elif rel[i] == 1:
                    x1 = torch.randint(self.dataset.env_size-1, (1,))
                    x2 = x1 + 1
                pos[i,0] = x1
                pos[i,1] = x2
        for i in range(self.batch_size):
            if self.tokenize:
                inp1[i,0] = self.dataset.env_tokens[self.envi[i]][pos[i,0]]
                inp2[i,0] = self.dataset.env_tokens[self.envi[i]][pos[i,1]]
            else:
                inp1[i,:] = self.dataset.env_data[self.envi[i]][pos[i,0]]
                inp2[i,:] = self.dataset.env_data[self.envi[i]][pos[i,1]]
                    
        def reward_func(given_rel):
            rb = [ torch.equal(given_rel[b], rel[b]) for b in range(self.batch_size) ]
            r = torch.tensor(rb).float()
            return r
        
        truth = rel, pos
        
        return inp1, inp2, reward_func, truth
    
    def get_test_data(self):
        rel = torch.zeros(self.batch_size, dtype=torch.long)
        pos = torch.zeros(self.batch_size, 2, dtype=torch.long)
        inp1 = torch.zeros(self.batch_size, self.dataset.input_dim)
        inp2 = torch.zeros(self.batch_size, self.dataset.input_dim)
        for i in range(self.batch_size):
            pos[i,:] = torch.randperm(self.dataset.env_size)[:2]
            inp1[i,:] = self.dataset.env_data[self.envi[i]][pos[i,0]]
            inp2[i,:] = self.dataset.env_data[self.envi[i]][pos[i,1]]
        rel[pos[:,0] > pos[:,1]] = 0
        rel[pos[:,0] < pos[:,1]] = 1
                    
        def reward_func(given_rel):
            rb = [ torch.equal(given_rel[b], rel[b]) for b in range(self.batch_size) ]
            r = torch.tensor(rb).float()
            return r
        
        truth = rel, pos
        
        return inp1, inp2, reward_func, truth
        


    
class Dataset():
    def __init__(self, data_size, env_size, input_dim, relation_dim,
                 input_type='onehot', train_ratio=0.6, val_ratio=0.2, tokenize=False):
    
        self.data_size = data_size
        self.env_size = env_size
        self.input_dim = input_dim
        self.relation_dim = relation_dim
        self.input_type = input_type
        self.tokenize = tokenize

        # generate environment data
        
        self.env_data = []
        self.env_class_data = []
        if tokenize:
            self.env_tokens = []
        
        if input_type == 'cifar100':
            self.image_pcs_dict, self.image_basis, self.image_center = \
                dataset_utils.load_cifar100(self.input_dim)        
        
        for i in range(data_size):
            if input_type == 'onehot':
                env_inp = torch.randperm(input_dim)[:env_size]
                env = one_hot(env_inp, input_dim).float()
                env_class = 0
            elif input_type == 'gaussian':
                env = torch.randn(env_size, input_dim)
                env = env / torch.norm(env, dim=1).unsqueeze(1)
                env_class = 0
            elif input_type == 'cifar100':
                env_class = torch.randint(100, (1,)).item()
                image_pcs = self.image_pcs_dict[env_class]
                if tokenize:
                    while(True):
                        idx = torch.randperm(image_pcs.size(0))[:env_size]
                        tokens = idx
                        # tokens = torch.tensor([(torch.stack(sorted(idx), dim=0) == idx1).nonzero().item() for idx1 in idx])
                        if not(any([ torch.equal(tokens_, tokens) for tokens_ in self.env_tokens ])):
                            break
                else:
                    idx = torch.randperm(image_pcs.size(0))[:env_size]
                env = image_pcs[idx]
            else:
                raise Exception('unsupported input type:', input_type)

            self.env_data.append(env)
            self.env_class_data.append(env_class)
            if tokenize:
                self.env_tokens.append(tokens)
            
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
            
        
        