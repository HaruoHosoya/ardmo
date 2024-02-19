#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:07:58 2022

A python implementation of ARDMO (Abstract Relational Decision-making MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import utils
import scipy

from sklearn.manifold import MDS
from sklearn.decomposition import PCA


def plot_performance_adjacent(logs, block_len, fig=None, show_block=False):
    rew = torch.concat([log['reward'] for log in logs], dim=1)
    batch_size = rew.size(1)
    nstep = rew.size(0)
    nblock = int(nstep / block_len)
    rew = rew.reshape(nblock, block_len, batch_size)
    perf = rew.mean(dim=1)
    
    if fig is None:
        fig = plt.figure(figsize=(5,3.5))
    if show_block:
        x = torch.arange(1, nblock+1)
        xlabel = 'block#'
    else:
        x = torch.arange(1, nblock+1) * block_len
        xlabel = 'step#'

    plt.errorbar(x, perf.mean(dim=1), utils.sem(perf, dim=1))
    plt.xlabel(xlabel)
    plt.ylabel('performance')
    
    return { 'performance': perf, 'nblock' : nblock, 'block_len': block_len }
    
    
def plot_performance_nonadjacent(logs, block_len, show_conf=False, show_block=False):
    tls = [log['testlog'] for log in logs]
    rew = torch.concat([torch.stack([ l['reward'] for l in tl ], dim=0)
                        for tl in tls], dim=2)
    nblock = rew.size(0)
    nsample = rew.size(1)
    batch_size = rew.size(2)
    perf = rew.mean(dim=1)
    confc = torch.concat([torch.stack([ l['conf'] for l in tl ], dim=0)
                          for tl in tls], dim=2)
    conf = torch.floor(confc * 2.9999 + 1)
    inf_score = (rew * conf).mean(dim=1)
    mean_conf = conf.mean(dim=1)
    
    fig = plt.figure(figsize=(10,7))
    if show_block:
        x = torch.arange(1, nblock+1)
        xlabel = 'block#'
    else:
        x = torch.arange(1, nblock+1) * block_len
        xlabel = 'step#'
    if show_conf:
        ax = fig.add_subplot(2, 2, 1)
    else:
        ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(x, perf.mean(dim=1), utils.sem(perf, dim=1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('performance')

    if show_conf:
        ax = fig.add_subplot(2, 2, 2)
        ax.errorbar(x, mean_conf.mean(dim=1), utils.sem(mean_conf, dim=1))
        ax.set_xlabel(xlabel)
        ax.set_ylabel('confidence')

        ax = fig.add_subplot(2, 2, 3)
        ax.errorbar(x, inf_score.mean(dim=1), utils.sem(inf_score, dim=1))
        ax.set_xlabel(xlabel)
        ax.set_ylabel('inference score')
        ax.set_ylim([0, 3])

    return { 'performance': perf, 'confidence': conf, 'inference_score': inf_score,
             'nblock' : nblock, 'block_len': block_len }


def get_state_repr(agent, logs, val_world):
    agent.eval()
    agent.cpu()
    
    sts_l = []
    for log in logs:    
        memory_value = log['memory_value'].detach().cpu()   
        inp1 = log['inp1'].detach().cpu()
        inp2 = log['inp2'].detach().cpu()
        pos = log['pos'].detach().cpu()
    
        ds = val_world.dataset
        batch_size = memory_value.size(0)
        for i in range(batch_size):
            st_l = []
            for j in range(ds.env_size):
                p1 = (pos[:,i,0] == j).nonzero(as_tuple=True)[0]
                if len(p1) != 0:
                    inp = inp1[p1[0],i]
                else:
                    p2 = (pos[:,i,1] == j).nonzero(as_tuple=True)[0]
                    inp = inp2[p2[0],i]
                inp = inp.unsqueeze(0)
                st = agent.memory_infer(memory_value[i:i+1], inp)
                st_l.append(st)
            sts_l.append(torch.cat(st_l, dim=0))        
    state_repr = torch.stack(sts_l, dim=0).detach()

    return state_repr
    
def plot_state_repr(state_repr, num=5, show_mean=False):
    if show_mean:
        mean_state_repr = state_repr.mean(dim=0)
        state_repr = torch.concat((state_repr[0:num, :, :], mean_state_repr.unsqueeze(0)), dim=0)
        num += 1

    env_size = state_repr.size(1)
    num = min(state_repr.size(0), num)
    state_repr = state_repr[:num, :, :]

    fig = plt.figure()
    fig.set_tight_layout(True)
    for i in range(num):
        ax = fig.add_subplot(int(np.ceil(num / 2)), 2, i+1)
        im = plt.imshow(state_repr[i,:,:], cmap='jet')
        fig.colorbar(im, shrink=1)
        ax.set_xlabel('state')
        ax.set_ylabel('position')    
    
def plot_low_dim_state_space(state_repr, num=5, display='2d', method='mds', show_mean=False):
    if show_mean:
        mean_state_repr = state_repr.mean(dim=0)
        state_repr = torch.concat((state_repr[0:num, :, :], mean_state_repr.unsqueeze(0)), dim=0)
        num += 1
    
    env_size = state_repr.size(1)
    num = min(state_repr.size(0), num)
    state_repr = state_repr[:num, :, :]

    sts = state_repr.numpy().reshape(env_size * num, -1)
    if display == '2d':
        if method == 'mds':
            emb = MDS(n_components=2)
        elif method == 'pca':
            emb = PCA(n_components=2)
        X = emb.fit_transform(sts)
        X = X.reshape(num, env_size, -1)
        fig = plt.figure()
        if show_mean:
            for j in range(env_size):
                plt.scatter(X[:-2,j,0], X[:-2,j,1], label=j, s=100, marker='o')            
            for j in range(env_size):
                plt.scatter(X[-1,j,0], X[-1,j,1], label=j, s=100, marker='^')            
        else:
            for j in range(env_size):
                plt.scatter(X[:,j,0], X[:,j,1], label=j, s=100, marker='o')            
        for i in range(num):
            plt.plot(X[i,:,0], X[i,:,1], color='k', linewidth=1)        
        plt.legend()
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')

    elif display == '3d': 
        if method == 'mds':
            emb = MDS(n_components=3)
        elif method == 'pca':
            emb = PCA(n_components=3)
        X = emb.fit_transform(sts)
        X = X.reshape(num, env_size, -1)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')     
        if show_mean:
            for j in range(env_size):
                ax.scatter(X[:-2,j,0], X[:-2,j,1], X[:-2,j,2], label=j, s=50, marker='o')            
            for j in range(env_size):
                plt.scatter(X[-1,j,0], X[-1,j,1], X[-1,j,2], label=j, s=100, marker='^')            
        else:
            for j in range(env_size):
                ax.scatter(X[:,j,0], X[:,j,1], X[:,j,2], label=j, s=50, marker='o')                        
        for i in range(num):
            ax.plot(X[i,:,0], X[i,:,1], X[i,:,2], color='k', linewidth=0.5)        
        plt.legend()
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')

def show_example_env(dataset, i=0):
    imgs = dataset.generate_image(dataset.env_data[i])
    return imgs
    
    
    