#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:40:56 2022

A python implementation of ARDMO (Abstract Relational Decision-making MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""

import os as os
import os.path as osp
import numpy as np
import numpy.random as random
import scipy.stats

from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
import shutil
import re

import argparse
from distutils.util import strtobool

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data 
import torchvision
from torch.utils.tensorboard import SummaryWriter

import agent2d
import train
import datasets2d
import evalu2d

import utils
import params

#%%

def save_model(agent, epoch, opts):
    file_name = 'model.{}.pt'.format(epoch)
    agent2d.save_agent(agent, osp.join(opts.results_root, opts.model_id, file_name))
    shutil.copyfile(osp.join(opts.results_root, opts.model_id, file_name), 
                    osp.join(opts.results_root, opts.model_id, 'model.last.pt'))

def save_log(log, epoch, opts, train=True):
    train_str = 'train' if train else 'test'
    file_name = 'log.{}.{}.pt'.format(train_str, epoch)
    torch.save(log, osp.join(opts.results_root, opts.model_id, file_name))    
    shutil.copyfile(osp.join(opts.results_root, opts.model_id, file_name), 
                    osp.join(opts.results_root, opts.model_id, 'log.{}.last.pt'.format(train_str)))
        
def load_model(opts, epoch=None):
    if epoch is None:
        file_name = 'model.last.pt'
    else:
        file_name = 'model.{}.pt'.format(epoch)
    model = agent2d.load_agent(osp.join(opts.results_root, opts.model_id, file_name), opts=opts)
    return model
    
#%%

torch.backends.cudnn.benchmark = True

#%%

datasets_root = 'datasets'
results_root = 'results'
logs_root = 'logs'

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_root', type=str, default=datasets_root, help="dataset root path")
parser.add_argument('--results_root', type=str, default=results_root, help="results root path")
parser.add_argument('--logs_root', type=str, default=logs_root, help="logs root path")
parser.add_argument('--dataset', type=str, help="dataset name")
parser.add_argument('--model_id', type=str, help="model id")

parser.add_argument('--gpu', type=str, default=None, help="gpu id or 'cpu'")
parser.add_argument('--mode', type=str, default='train', help="'train' or 'test' or show'")

parser.add_argument('--state_dim', type=int, default=20, help="state dimension")
parser.add_argument('--memory_dim', type=int, default=50, help="memory dimension")
parser.add_argument('--trans_interm_dim', type=int, default=10, help="transition intermediate layer dimension")
parser.add_argument('--trans_nonlin', type=str, default='normalize', help="transition nonlinearity")
parser.add_argument('--trans_prior', type=bool, default=True, help="learn transition prior")

parser.add_argument('--axis_dim', type=int, default=None, help="number of relational axes")

parser.add_argument('--alpha', type=float, default=0.7, help="alpha (memory update rate)")
parser.add_argument('--k_prior', type=float, default=1.0, help="transition prior regularization (coefficient)")
parser.add_argument('--seq_len', type=int, default=2000, help="sequence length (single environment)")
parser.add_argument('--block_len', type=int, default=25, help="block length (truncate interval)")

parser.add_argument('--random_reward', type=bool, default=False, help="random reward (only training)")

parser.add_argument('--batch_size', type=int, default=5, help="batch size")
parser.add_argument('--epochs', type=int, default=10000, help="number of epochs")

parser.add_argument('--save_interval', type=int, default=100, help="save interval for model training")

opts = parser.parse_args()

print(opts)

#%%

if opts.mode == 'train':
    os.makedirs(opts.logs_root, exist_ok=True)
    os.makedirs(osp.join(opts.results_root, opts.model_id), exist_ok=True)
else:
    if not os.path.exists(osp.join(opts.results_root, opts.model_id)):
        print('Result directoriy does not exist')
        exit(1)
        
#%%

if opts.gpu == None or opts.gpu == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + opts.gpu)

#%%

ds = torch.load(osp.join(opts.datasets_root, opts.dataset + '.pt'))
train_world = datasets2d.World(ds, split='train', random_reward=opts.random_reward)
val_world = datasets2d.World(ds, split='val')
test_world = datasets2d.World(ds, split='test')

#%%

if opts.mode == 'train':
    opts.log_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.' + opts.model_id
    torch.save(opts, osp.join(opts.results_root, opts.model_id, 'options.pt'))    
    axis_dim = 2 if opts.axis_dim is None else opts.axis_dim
    agent = agent2d.Agent(ds.input_dim, 
                             opts.state_dim, 
                             opts.memory_dim, 
                             relation_dim=2, 
                             axis_dim=axis_dim,
                             trans_nonlin=opts.trans_nonlin,
                             trans_prior=opts.trans_prior,
                             trans_interm_dim=opts.trans_interm_dim)
    optimizer = optim.Adam(params=utils.get_params(agent), lr=0.001)
    epoch = 0
    train_hist = []
    test_hist = []

elif opts.mode == 'continue' or ('eval' in opts.mode) or opts.mode == 'show':
    opts_ = opts
    opts = torch.load(osp.join(opts.results_root, opts.model_id, 'options.pt'))
    opts.results_root = opts_.results_root
    opts.logs_root = opts_.logs_root
    opts.epochs = opts_.epochs
    opts.mode = opts_.mode
    opts.model_id = opts_.model_id
    agent = load_model(opts)
    optimizer = optim.Adam(params=utils.get_params(agent), lr=0.001)
    train_hist = torch.load(osp.join(opts.results_root, opts.model_id, 'train_hist.pt'))
    test_hist = torch.load(osp.join(opts.results_root, opts.model_id, 'test_hist.pt'))
    epoch = train_hist[-1][0] if train_hist != [] else 0
    if epoch >= opts.save_interval:
        agent2 = load_model(opts, epoch=epoch-opts.save_interval)
    else:
        agent2 = agent
    
#%%

writer = SummaryWriter(log_dir=osp.join(opts.logs_root, opts.log_id))
    
#%%

if opts.mode == 'train' or opts.mode == 'continue':
    #%%
    save_model(agent, -100, opts)
    
    if epoch == 0:
        save_model(agent, 0, opts)
        torch.save(train_hist, osp.join(opts.results_root, opts.model_id, 'train_hist.pt'))        
    
    #%%    
    
    while(True):
        # test once and save
    
        loss, rewards, logs = \
            train.run_model(
                agent, 
                val_world, 
                [], 
                num_epochs=1, 
                seq_len=opts.seq_len, 
                start_epoch=epoch,
                train=False, 
                alpha=opts.alpha, 
                k_prior=opts.k_prior,
                batch_size=opts.batch_size, 
                device=device, 
                writer=writer, 
                )
        log = logs[-1]
        
        test_hist += [(log['epoch'], log['loss'], log['mean_total_reward']) for log in logs]
        torch.save(test_hist, osp.join(opts.results_root, opts.model_id, 'test_hist.pt'))

        if epoch >= opts.epochs:
            break
    
        # training for some epochs and save
    
        loss, rewards, logs = \
            train.run_model(
                agent, 
                train_world, 
                optimizer, 
                num_epochs=opts.save_interval, 
                seq_len=opts.seq_len, 
                start_epoch=epoch+1,                              
                train=True,
                truncate=opts.block_len, 
                alpha=opts.alpha, 
                k_prior=opts.k_prior,
                batch_size=opts.batch_size, 
                device=device, 
                writer=writer, 
                )
        epoch += opts.save_interval

        log = logs[-1]
        save_model(agent, epoch, opts)
        
        train_hist += [(log['epoch'], log['loss'], log['mean_total_reward']) for log in logs]
        torch.save(train_hist, osp.join(opts.results_root, opts.model_id, 'train_hist.pt'))        
    
#%%    
elif opts.mode == 'eval-full':
    test_size = 100
    
    test_world.init_env_series()

    loss, total_rew, logs = \
        train.run_model(
            agent, 
            test_world, 
            [], 
            num_epochs=test_size//opts.batch_size, 
            seq_len=opts.seq_len, 
            start_epoch=0,
            train=False, 
            save_all_memory_value=True,
            truncate=opts.block_len, 
            alpha=opts.alpha, 
            batch_size=opts.batch_size, 
            device=device, 
            writer=None, 
            )
    torch.save(logs, osp.join(opts.results_root, opts.model_id, 'log.eval.full.pt'))            
    
    test_epochs = [e for e,l,r in test_hist]

    
    
#%%
elif opts.mode == 'show':

    # loss/reward trace
    
    print('generating files for loss/reward')
    
    fig = plt.figure(figsize=(6,3))
    fig.set_tight_layout(True)

    train_epochs = torch.tensor([e for e,l,r in train_hist])
    test_epochs = torch.tensor([e for e,l,r in test_hist])
    train_loss = torch.tensor([l for e,l,r in train_hist])
    test_loss = torch.tensor([l for e,l,r in test_hist])
    train_reward = torch.tensor([r for e,l,r in train_hist])
    test_reward = torch.tensor([r for e,l,r in test_hist])

    ax = fig.add_subplot(1, 2, 1)
    plt.plot(train_epochs, train_loss)    
    plt.plot(test_epochs, test_loss)    
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(['training', 'test'])
    
    ax = fig.add_subplot(1, 2, 2)
    plt.plot(train_epochs, train_reward)    
    plt.plot(test_epochs, test_reward)    
    ax.set_xlabel('epoch')
    ax.set_ylabel('total reward')
    ax.legend(['training', 'test'])

    plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_loss_reward.pdf'))
    rec = { 'train_epochs': train_epochs, 'train_loss': train_loss, 'train_reward': train_reward,
            'test_epochs': test_epochs, 'test_loss': test_loss, 'test_reward': test_reward }
    torch.save(rec, osp.join(opts.results_root, opts.model_id, 'rec_loss_reward.pt'))

    # performance 
    
    print('generating files for performance')

    logs = torch.load(osp.join(opts.results_root, opts.model_id, 'log.eval.full.pt'))            
    # logs2 = torch.load(osp.join(opts.results_root, opts.model_id, 'log2.eval.full.pt'))            

    fig = plt.figure()
    fig.set_tight_layout(True)
    rec = evalu2d.plot_performance_adjacent(logs, 25, show_block=False)
    plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_performance_adj.pdf'))
    torch.save(rec, osp.join(opts.results_root, opts.model_id, 'rec_performance_adj.pt'))
    
    # state representation
    
    print('generating files for representation')

    sr, mr = evalu2d.get_state_repr(agent, logs, test_world, zscore=True)
    sr2, mr2 = evalu2d.get_state_repr(agent, logs, test_world, nlast=2, zscore=True)
    
    # state representation
    fig = plt.figure(figsize=(10,8))
    fig.set_tight_layout(True)
    evalu2d.plot_state_repr(sr.mean(dim=0))
    plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_state_repr.pdf'))
    
    # state direction representation
    fig = plt.figure(figsize=(10,8))
    fig.set_tight_layout(True)
    yss = []
    l = min(len(sr), 10)
    for i in range(l):
        sdr = evalu2d.get_state_dir_repr(sr[i])
        fig.add_subplot(2, l // 2, i+1)
        xs, ys, es = evalu2d.plot_state_dir_repr(sdr, prd=6, adjust_phase=True, show_err=False)
        yss.append(ys)
    plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_state_dir_repr.pdf'))

    # distance representation
    fig = plt.figure(figsize=(10,8))
    fig.set_tight_layout(True)
    for i in range(min(10, len(mr))):
        dr = evalu2d.get_dist_repr(mr[i])
        fig.add_subplot(2, l // 2, i+1)
        evalu2d.plot_dist_repr(dr, skip_plot=False)
    plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_interm_dist_repr.pdf'))

    fig = plt.figure(figsize=(5,4))
    fig.set_tight_layout(True)
    cs = []    
    recs = []
    for i in range(len(mr)):
        dr = evalu2d.get_dist_repr(mr[i])
        rec = evalu2d.plot_dist_repr(dr, skip_plot=True)
        recs.append(rec)
        cs.append(np.corrcoef(rec['xs'], rec['ys'])[0,1])
    yss = torch.tensor([ rec['ys'] for rec in recs])
    plt.errorbar(recs[0]['xs'], yss.mean(dim=0), yerr=utils.sem(yss, dim=0))
    plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_interm_dist_repr_comb.pdf'))
    torch.save(recs, osp.join(opts.results_root, opts.model_id, 'recs_interm_dist.pt'))

    fig = plt.figure(figsize=(5,4))
    fig.set_tight_layout(True)
    plt.hist(cs, bins=np.linspace(-1, 1, 11))
    plt.xlabel('correlation')
    plt.ylabel('# of domains')
    plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_interm_dist_repr_corr.pdf'))

    fig = plt.figure(figsize=(5,4))
    fig.set_tight_layout(True)
    recs = []
    cs = []
    for i in range(len(sr)):
        dr = evalu2d.get_dist_repr(sr[i])
        rec = evalu2d.plot_dist_repr(dr, skip_plot=True)
        recs.append(rec)
        cs.append(np.corrcoef(rec['xs'], rec['ys'])[0,1])
    yss = torch.tensor([ rec['ys'] for rec in recs])
    plt.errorbar(recs[0]['xs'], yss.mean(dim=0), yerr=utils.sem(yss, dim=0))
    plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_state_dist_repr_comb.pdf'))
    torch.save(recs, osp.join(opts.results_root, opts.model_id, 'recs_state_dist.pt'))

    fig = plt.figure(figsize=(5,4))
    fig.set_tight_layout(True)
    plt.hist(cs, bins=np.linspace(-1, 1, 11))
    plt.xlabel('correlation')
    plt.ylabel('# of domains')
    plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_state_dist_repr_corr.pdf'))
    

    # periodicity
    prds = [4, 5, 6, 7, 8]
    dlts_mean = []
    dlts_sem = []
    ys_mean = []
    ys_sem = []
    xss = []
    phases_l = []
    dct = {}
    for prd in prds:
        yss = []
        dct[prd] = []
        phases = torch.stack([ evalu2d.get_state_dir_phases(sr2[i], prd=prd) for i in range(len(sr2))], dim=0)
        phases_l.append(phases)
        for i in range(len(sr)):
            sdr = evalu2d.get_state_dir_repr(sr[i])
            mean_phase = scipy.stats.circmean(phases[i], high=np.pi*2/prd)
            xs, ys, es = evalu2d.plot_state_dir_repr(sdr, prd=prd, fixed_phase=mean_phase, skip_plot=True)
            yss.append(ys)
            dct[prd].append((xs, ys, es))
        xss.append(xs)
        ys_mean.append(torch.tensor(yss).nanmean(dim=0).numpy())
        ys_sem.append(utils.sem(torch.tensor(yss),dim=0).numpy())
        dlt = [(np.mean(ys[0::2]) - np.mean(ys[1::2])) for ys in yss]
        dlts_mean.append(np.mean(dlt))
        dlts_sem.append(utils.sem(dlt))

    torch.save(dct, osp.join(opts.results_root, opts.model_id, 'rec_periodicity.pt'))

    fig = plt.figure(figsize=(6,3))
    fig.set_tight_layout(True)
    plt.errorbar(xss[2], ys_mean[2], yerr=ys_sem[2])
    plt.xlabel('direction (deg)')
    plt.ylabel('activity')
    plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_dir_activity.pdf'))
    
    fig = plt.figure(figsize=(6,3))
    fig.set_tight_layout(True)
    plt.errorbar(prds, dlts_mean, yerr=dlts_sem)
    plt.xlabel('periodicity')
    plt.ylabel('activity delta')
    plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_periodicity.pdf'))

    # phase
    fig = plt.figure(figsize=(5,5))
    fig.set_tight_layout(True)
    # phases = sum([ evalu2d.get_state_dir_phases(sr[i]) for i in range(len(sr))], [])
    phases = [ (p % (np.pi / 6)) * 12 for p in torch.flatten(phases_l[2]) ]
    # phases = [ (p % (np.pi / 6)) * 12 for p in phases ]
    h, bins = np.histogram(phases, np.linspace(0, np.pi * 2, 13))
    ax = fig.add_subplot(projection='polar')
    ax.bar(bins[:-1], h, width=np.pi / 6 * 0.9, bottom=0.0, 
           tick_label=['0', '', '', 'π/12', '', '', 'π/6', '', '', '3π/12', '', ''])
    plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_phase.pdf'))

