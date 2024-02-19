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

import agent1d 
import train 
import datasets1d 
import utils
import evalu1d
import params

#%%

def save_model(agent, epoch, opts):
    file_name = 'model.{}.pt'.format(epoch)
    agent1d.save_agent(agent, osp.join(opts.results_root, opts.model_id, file_name))
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
    model = agent1d.load_agent(osp.join(opts.results_root, opts.model_id, file_name), opts=opts)
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

parser.add_argument('--alpha', type=float, default=0.7, help="alpha (memory update rate)")
parser.add_argument('--k_prior', type=float, default=1.0, help="transition prior regularization (coefficient)")
parser.add_argument('--seq_len', type=int, default=3000, help="sequence length (single environment)")
parser.add_argument('--block_len', type=int, default=25, help="block length (truncate interval)")
parser.add_argument('--no_reward_update', type=bool, default=False, help="update when no reward")

parser.add_argument('--batch_size', type=int, default=5, help="batch size")
parser.add_argument('--epochs', type=int, default=10000, help="number of epochs")

parser.add_argument('--save_interval', type=int, default=100, help="save interval for model training")

opts = parser.parse_args()

#%%

os.makedirs(opts.logs_root, exist_ok=True)
os.makedirs(osp.join(opts.results_root, opts.model_id), exist_ok=True)

#%%

if opts.gpu == None or opts.gpu == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + opts.gpu)

#%%

ds = torch.load(osp.join(opts.datasets_root, opts.dataset + '.pt'))
train_world = datasets1d.World(ds, split='train')
val_world = datasets1d.World(ds, split='val')
test_world = datasets1d.World(ds, split='test')

#%%

if opts.mode == 'train':
    opts.log_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.' + opts.model_id
    torch.save(opts, osp.join(opts.results_root, opts.model_id, 'options.pt'))    
    agent = agent1d.Agent(ds.input_dim, 
                          opts.state_dim, 
                          opts.memory_dim, 
                          ds.relation_dim, 
                          trans_nonlin=opts.trans_nonlin,
                          trans_prior=opts.trans_prior,
                          trans_interm_dim=opts.trans_interm_dim)
    optimizer = optim.Adam(params=utils.get_params(agent), lr=0.001)
    epoch = 0
    train_hist = []
    test_hist = []

elif opts.mode == 'continue' or ('eval' in opts.mode) or ('show' in opts.mode):
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
    epoch = train_hist[-1][0]
    
#%%

writer = SummaryWriter(log_dir=osp.join(opts.logs_root, opts.log_id))
    
#%%

if opts.mode == 'train' or opts.mode == 'continue':
    #%%
    save_model(agent, -100, opts)
    
    if epoch == 0:
        save_model(agent, 0, opts)
    
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
                no_reward_update=opts.no_reward_update,
                k_prior=opts.k_prior,
                batch_size=opts.batch_size, 
                device=device, 
                writer=writer, 
                )
        log = logs[-1]
        # save_log(log, epoch, opts, train=False)
        
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
                no_reward_update=opts.no_reward_update,
                k_prior=opts.k_prior,
                batch_size=opts.batch_size, 
                device=device, 
                writer=writer, 
                )
        epoch += opts.save_interval

        log = logs[-1]
        save_model(agent, epoch, opts)
        # save_log(log, epoch, opts, train=True)
        
        train_hist += [(log['epoch'], log['loss'], log['mean_total_reward']) for log in logs]
        torch.save(train_hist, osp.join(opts.results_root, opts.model_id, 'train_hist.pt'))        
    
#%%    
elif opts.mode == 'eval-full':
    test_size = 100

    loss, total_rew, logs = \
        train.run_model(
            agent, 
            test_world, 
            [], 
            num_epochs=test_size//opts.batch_size, 
            seq_len=1000, 
            start_epoch=0,
            train=False, 
            testing=True, 
            truncate=opts.block_len, 
            alpha=opts.alpha, 
            no_reward_update=opts.no_reward_update,
            batch_size=opts.batch_size, 
            device=device, 
            writer=None, 
            )
    torch.save(logs, osp.join(opts.results_root, opts.model_id, 'log.eval.full.pt'))            
    
    test_epochs = [e for e,l,r in test_hist]

    # intv = opts.epochs // 10
    # epcs = np.arange(intv, test_epochs[-1]+1, intv)

    last_epoch = test_epochs[-1]
    intv = test_epochs[1] - test_epochs[0]
    epcs = list(np.arange(1, 6) * intv) + list(np.arange(10, last_epoch/intv, 10, dtype=int) * intv)
    if epcs[-1] != last_epoch: 
        epcs += [last_epoch]
    
    # epcs = [100, 300, 500, 1000, last_epoch]    
    epcs = [100]    
    for epc in epcs:
        agent_interm = load_model(opts, epc)
        loss, total_rew, logs_interm = \
            train.run_model(
                agent_interm, 
                test_world, 
                [], 
                num_epochs=test_size//opts.batch_size, 
                seq_len=1000, 
                start_epoch=0,
                train=False, 
                testing=False, 
                truncate=opts.block_len, 
                alpha=opts.alpha, 
                no_reward_update=opts.no_reward_update,
                batch_size=opts.batch_size, 
                device=device, 
                writer=None, 
                )
        torch.save(logs_interm, osp.join(opts.results_root, opts.model_id, 'log.eval.{}.pt'.format(epc)))            

#%%

elif opts.mode == 'eval-last':
    test_size = 100
    test_epochs = [e for e,l,r in test_hist]
    intv = opts.epochs // 10
    epc = test_epochs[-1]
    agent_interm = load_model(opts, epc)
    loss, total_rew, logs = \
        train.run_model(
            agent_interm, 
            test_world, 
            [], 
            num_epochs=test_size//opts.batch_size, 
            seq_len=1000, 
            start_epoch=0,
            train=False, 
            testing=False, 
            truncate=opts.block_len, 
            alpha=opts.alpha, 
            no_reward_update=opts.no_reward_update,
            batch_size=opts.batch_size, 
            device=device, 
            writer=None, 
            )
    torch.save(logs, osp.join(opts.results_root, opts.model_id, 'log.eval.{}.pt'.format(epc)))            
    
#%%
elif opts.mode == 'show' or opts.mode == 'show-last':

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

    if opts.mode == 'show-last':
        logs = torch.load(osp.join(opts.results_root, opts.model_id, 'log.eval.{}.pt'.format(train_epochs[-1])))
    else:           
        logs = torch.load(osp.join(opts.results_root, opts.model_id, 'log.eval.full.pt'))            

    fig = plt.figure()
    fig.set_tight_layout(True)
    rec = evalu1d.plot_performance_adjacent(logs, 25, show_block=False)
    plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_performance_adj.pdf'))
    torch.save(rec, osp.join(opts.results_root, opts.model_id, 'rec_performance_adj.pt'))
    
    if not opts.mode == 'show-last':
        fig = plt.figure()
        fig.set_tight_layout(True)
        rec = evalu1d.plot_performance_nonadjacent(logs, 25, show_conf=True)
        plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_performance_nonadj.pdf'))
        torch.save(rec, osp.join(opts.results_root, opts.model_id, 'rec_performance_nonadj.pt'))
        
        epcs = [re.findall('log.eval.(\d*).pt',f) for f in os.listdir(osp.join(opts.results_root, opts.model_id))]
        epcs = sorted([int(e[0]) for e in epcs if len(e) != 0])
        fig = plt.figure()
        fig.set_tight_layout(True)
        recs = []
        for epc in epcs:
            logs_interm = torch.load(osp.join(opts.results_root, opts.model_id, 'log.eval.{}.pt'.format(epc)))
            rec = evalu1d.plot_performance_adjacent(logs_interm, 25, fig=fig)     
            rec['epoch'] = epc
            recs.append(rec)
        plt.legend(['{} epcs'.format(epc) for epc in epcs])
        plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_performance_adj_hist.pdf'))
        torch.save(recs, osp.join(opts.results_root, opts.model_id, 'recs_performance_adj_hist.pt'))
        
    
        # state representation
        
        print('generating files for representation')
    
        fig = plt.figure()
        fig.set_tight_layout(True)
        state_repr = evalu1d.get_state_repr(agent, logs, test_world)
        evalu1d.plot_low_dim_state_space(state_repr, num=5)
        plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_state_repr_mds.pdf'))
        
        evalu1d.plot_state_repr(state_repr, num=5)
        plt.savefig(osp.join(opts.results_root, opts.model_id, 'fig_state_repr_ind.pdf'))
        
