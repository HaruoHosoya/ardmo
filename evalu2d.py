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

from sklearn.linear_model import LinearRegression


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
        

def get_state_repr(agent, logs, val_world, nlast=None, zscore=False):
    agent.eval()
    agent.cpu()
    
    sts_l = []
    mis_l = []
    for log in logs:    
        if nlast is None:
            memory_value = log['memory_value'].detach().cpu()   
        else:
            memory_value = log['all_memory_value'][-nlast].detach().cpu() 
        inp1 = log['inp1'].detach().cpu()
        inp2 = log['inp2'].detach().cpu()
        pos = log['pos'].detach().cpu()
    
        ds = val_world.dataset
        batch_size = memory_value.size(0)
        for i in range(batch_size):
            st_l = []
            mi_l = []
            for j in range(ds.env_size):
                for k in range(ds.env_size):
                    p1 = ((pos[:,i,0,0] == j) & (pos[:,i,1,0] == k)).nonzero(as_tuple=True)[0]
                    if len(p1) != 0:
                        inp = inp1[p1[0],i]
                    else:
                        p2 = ((pos[:,i,0,1] == j) & (pos[:,i,1,1] == k)).nonzero(as_tuple=True)[0]
                        inp = inp2[p2[0],i]
                    inp = inp.unsqueeze(0)
                    st, mi = agent.memory_infer(memory_value[i:i+1], inp, return_interm=True)
                    st_l.append(st)
                    mi_l.append(mi)
            sts_l.append(torch.cat(st_l, dim=0).reshape(ds.env_size, ds.env_size, -1))        
            mis_l.append(torch.cat(mi_l, dim=0).reshape(ds.env_size, ds.env_size, -1))        
    state_repr = torch.stack(sts_l, dim=0).detach()
    interm_repr = torch.stack(mis_l, dim=0).detach()

    if zscore:
        state_repr = (state_repr - state_repr.mean(dim=(0,1,2,3))) / state_repr.std(dim=(0,1,2,3))
        interm_repr = (interm_repr - interm_repr.mean(dim=(0,1,2,3))) / interm_repr.std(dim=(0,1,2,3))

    return state_repr, interm_repr
    
def plot_state_repr(state_repr):
    env_size = state_repr.size(0)
    state_dim = state_repr.size(2)

    fig = plt.figure(figsize=(10,10))
    fig.set_tight_layout(True)
    for i in range(state_dim):
        ht = int(np.sqrt(state_dim))
        ax = fig.add_subplot(ht+1, state_dim // ht, i+1)
        # im = plt.imshow(state_repr[:,:,i], cmap='jet')
        vmax = state_repr[:,:,i].reshape(-1).abs().max()
        # im = plt.imshow(state_repr[:,:,i], cmap='coolwarm', vmax=vmax, vmin=-vmax)
        im = plt.imshow(state_repr[:,:,i], cmap='coolwarm')
        fig.colorbar(im, shrink=0.5)
        ax.set_title('dim #{}'.format(i))
        ax.set_xlabel('x')
        ax.set_ylabel('y')    

def get_state_dir_repr(state_repr):
    env_size = state_repr.size(0)
    state_dim = state_repr.size(2)
    sdr = []
    for x1 in range(env_size):
        for y1 in range(env_size):
            for x2 in range(env_size):
                for y2 in range(env_size):
                    if x1 == x2 and y1 == y2: continue
                    s1 = state_repr[x1,y1,:]
                    s2 = state_repr[x2,y2,:]
                    direction = np.arctan2(y1 - y2, x1 - x2)
                    mean_rel = (s1.mean() + s2.mean()) / 2
                    sdr.append((direction, mean_rel))
    return sdr

def plot_state_dir_repr(sdr, prd=6, fixed_phase=None, adjust_phase=False, skip_plot=False, show_err=True):
    # adjust phase
    if adjust_phase:
        X = [(np.cos(dr * prd), np.sin(dr * prd)) for dr, ma in sdr ]
        y = [ma for dr, ma in sdr]
        reg = LinearRegression().fit(X, y)
        beta_cos = reg.coef_[0]
        beta_sin = reg.coef_[1]
        phase = np.arctan2(beta_sin, beta_cos) / prd
        sdr = [(dr - phase, ma) for dr, ma in sdr]
    elif not fixed_phase is None:
        sdr = [(dr - fixed_phase, ma) for dr, ma in sdr]
    
    d = { i:[] for i in range(prd * 2) }
    for dr, ma in sdr:       
        i = ((dr + np.pi/(prd*2)) // (np.pi/prd)) % (prd * 2)
        d[i].append(ma)
    
    xs = [i * np.pi / prd for i in range(prd * 2) ]
    ys = [np.nanmean(d[i]) if len(d[i]) > 0 else np.nan for i in range(prd * 2)  ]
    es = [utils.sem(d[i]) if len(d[i]) > 0 else np.nan for i in range(prd * 2)  ]
    
    bl = np.min([np.nanmean(d[i]) for i in range(prd * 2) if len(d[i]) > 0])
    if not skip_plot:
        if show_err:
            plt.bar(xs, [y-bl for y in ys], yerr=es)
        else:
            plt.bar(xs, [y-bl for y in ys])
    return xs, ys, es

def get_state_dir_phases(state_repr, prd=6):
    env_size = state_repr.size(0)
    state_dim = state_repr.size(2)
    sdr = [ [] for i in range(state_dim) ]
    for x1 in range(env_size):
        for y1 in range(env_size):
            for x2 in range(env_size):
                for y2 in range(env_size):
                    if x1 == x2 and y1 == y2: continue
                    s1 = state_repr[x1,y1,:]
                    s2 = state_repr[x2,y2,:]
                    direction = np.arctan2(y1 - y2, x1 - x2)
                    mean_rel = (s1 + s2) / 2
                    for i in range(state_dim):
                        sdr[i].append((direction, mean_rel[i]))
    phases = []
    for i in range(state_dim):
        X = [(np.cos(dr * prd), np.sin(dr * prd)) for dr, ma in sdr[i] ]
        y = [ma for dr, ma in sdr[i]]
        reg = LinearRegression().fit(X, y)
        beta_cos = reg.coef_[0]
        beta_sin = reg.coef_[1]
        phase = np.arctan2(beta_sin, beta_cos) / prd
        phases.append(phase)

    return torch.tensor(phases)
    


    
def get_dist_repr(rep):
    env_size = rep.size(0)
    interm_dim = rep.size(2)
    dr = []
    for x1 in range(env_size):
        for y1 in range(env_size):
            for x2 in range(env_size):
                for y2 in range(env_size):
                    if x1 == x2 and y1 == y2: continue
                    s1 = rep[x1,y1,:]
                    s2 = rep[x2,y2,:]
                    dist = (y1 - y2) ** 2 + (x1 - x2) ** 2
                    dissim = torch.sqrt(torch.sum((s1 - s2) ** 2))
                    dr.append((dist, dissim))
    return dr
 
def plot_dist_repr(dr, skip_plot=False):
    dists = sorted(list(np.unique([ dist for (dist, dissim) in dr ])))
    d = {}
    for dist in dists:
        d[dist] = []
    for dist, dissim in dr:
        d[dist].append(dissim)
        
    xs = dists
    ys = [np.nanmean(d[dist]) for dist in dists]
    es = [utils.sem(d[dist]) for dist in dists]
    
    if not skip_plot:
        plt.bar(xs, ys, yerr=es)

    return { 'xs': xs, 'ys': ys, 'es': es }
        
    
    
def show_example_env(dataset, i=0):
    imgs = dataset.generate_image(dataset.env_data[i])
    return imgs
    
    
    