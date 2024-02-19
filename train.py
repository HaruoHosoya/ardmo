#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:50:54 2022

A python implementation of ARDMO (Abstract Relational Decision-making MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.linalg as L
import utils
import scipy
import time


def run_model(agent, world, optimizer, num_epochs, seq_len,
              start_epoch=0, **kwargs):
                
    logs = []
    print('Run for seq_len={}'.format(seq_len), 'with:', kwargs)
    for epoch in range(num_epochs):
        epoch_num = start_epoch + epoch
        print('Epoch #{} ({}/{})'.format(epoch_num, epoch + 1, num_epochs))
        start_time = time.time()
        
        epoch_loss, mean_total_reward, log = run_model_one_epoch(
            agent, world, optimizer, epoch_num, seq_len,
            **kwargs)

        log['epoch'] = epoch_num
        logs.append(log)
        end_time = time.time()
        print('Elapsed time: {:.4f}s'.format(end_time - start_time))

    return epoch_loss, mean_total_reward, logs

#%%

def run_model_one_epoch(agent, world, optimizer, epoch_num, seq_len,
                        batch_size=1, train=True, testing=False, 
                        k_prior=1.0, 
                        alpha=0.1, truncate=np.Inf, eps=1e-10, 
                        no_reward_update=False,
                        save_all_memory_value=False,
                        device='cpu', writer=None):

    agent.to(device)    

    if train:
        agent.train()
    else:
        agent.eval()
    
    sd = agent.state_dict()
    
    world.next_env(batch_size)
    # print('env #{}'.format(world.envi.numpy()))

    inp1_l = []
    inp2_l = []
    memory_value_l = []
    state1_l = []
    state2_l = []
    reward_l = []
    pos_l = []
    rel_l = []
    testlog_l = []

    total_reward = torch.zeros(batch_size)
            
    memory_value = agent.new_memory_value(batch_size)
    memory_value = memory_value.to(device)        
        
    truncate = min(truncate, seq_len)
    
    epoch_loss = 0.0
    epoch_loss_poster = 0.0
    epoch_loss_prior = 0.0

    with torch.set_grad_enabled(train): 

        for t0 in range(0, seq_len, truncate):

            # print('seq #{}'.format(t0))
            valid = True
            memory_value = memory_value.detach()
            loss = torch.tensor(0.0).to(device)
            loss_poster = torch.tensor(0.0).to(device)
            loss_prior = torch.tensor(0.0).to(device)

            if train:
                optimizer.zero_grad()
            
            for t in range(t0, min(t0 + truncate, seq_len)):
                
                # memory_value_l = []     # to save space for memory value log
                rel_axis = torch.randint(agent.axis_dim, (1,))
                
                if agent.axis_dim == 1:
                    inp1, inp2, reward_func, truth = world.get_data()
                else:
                    inp1, inp2, reward_func, truth = world.get_data(rel_axis)

                inp1 = inp1.to(device)
                inp2 = inp2.to(device)
                
                # infer the states from the two inputs
                
                state1_inf = agent.memory_infer(memory_value, inp1)
                state2_inf = agent.memory_infer(memory_value, inp2)
                
                # select the relation 
                # according to the transition probabilities between the states
    
                log_prob = torch.zeros(batch_size, agent.relation_dim).to(device)
                
                if agent.axis_dim == 1:
                    state2_cand_mu, state2_cand_sig = agent.transit_all(state1_inf)
                else:
                    state2_cand_mu, state2_cand_sig = agent.transit_all(state1_inf, rel_axis)
                log_prior = agent.transit_log_prior_all(state1_inf)
                for a in range(agent.relation_dim):
                    f = - torch.log(state2_cand_sig[a] + eps) - 0.5 * ((state2_inf - state2_cand_mu[a]) / (state2_cand_sig[a] + eps)) ** 2 
                    log_prob[:, a] = torch.sum(f, dim=1) + log_prior[:, a]
                                    
                rel_prob = F.softmax(log_prob, dim=1)
                rel_prior = F.softmax(log_prior, dim=1)
                reli = torch.multinomial(rel_prob, 1).squeeze(1)
                                
                rel_prob_ = torch.zeros(batch_size).to(device)
                rel_prior_ = torch.zeros(batch_size).to(device)
                for b in range(batch_size):
                    rel_prob_[b] = rel_prob[b, reli[b]]
                    rel_prior_[b] = rel_prior[b, reli[b]]

                # get reward
                
                reward = reward_func(reli.detach().cpu())
                total_reward += reward
                
                # reward maximization
                
                err_poster = torch.sum((reward.to(device) - rel_prob_) ** 2, 0)
                loss = loss + err_poster
                loss_poster = loss_poster + err_poster

                # prior regularization
                
                err_prior = torch.sum((reward.to(device) - rel_prior_) ** 2, 0)  
                loss = loss + k_prior * err_prior
                loss_prior = loss_prior + k_prior * err_prior
                    
                # update

                if agent.relation_dim == 2 and no_reward_update:
                    update_rate = torch.ones(batch_size).to(device) * alpha
                else:
                    reward_obtained = (reward > 0).float().to(device)
                    update_rate = reward_obtained * alpha
    
                inp_prev1 = agent.memory_retrieve(memory_value, state1_inf)

                state2_pred_mu = torch.zeros_like(state1_inf)
                state2_pred_sig = torch.zeros_like(state1_inf)
                for i in range(batch_size):
                    a = reli[i]
                    if agent.relation_dim == 2 and no_reward_update and reward[i] == 0:
                        a = 1 - reli[i]
                    state2_pred_mu[i] = state2_cand_mu[a,i] 
                    state2_pred_sig[i] = state2_cand_sig[a,i]
                state2_pred = utils.sample_normal(state2_pred_mu, state2_pred_sig)
                
                inp_pred2 = agent.memory_retrieve(memory_value, state2_pred)                                    

                memory_value = agent.memory_update(memory_value, inp1, inp_prev1, state1_inf, update_rate)                
                memory_value = agent.memory_update(memory_value, inp2, inp_pred2, state2_pred, update_rate)

                # loss check
    
                if np.isnan(loss.item()) or np.isinf(loss.item()):
                    print('nan/inf loss: sequence discarded')
                    valid = False
                    break
                
                # logging
                
                reward_l.append(reward)
                inp1_l.append(inp1.detach().cpu())
                inp2_l.append(inp2.detach().cpu())
                state1_l.append(state1_inf.detach().cpu())
                state2_l.append(state2_inf.detach().cpu())
                memory_value_l.append(memory_value.detach().cpu())
                pos_l.append(truth[1])
                rel_l.append(reli.detach().cpu())
    
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                print('nan/inf loss: sequence discarded')
                valid = False
                break

            if not valid:
                continue
            
            if train:
                loss.backward() 
                optimizer.step()
                
            if testing:
                # print('update: {}'.format(sum(reward_l[-truncate-1:-1])/float(truncate)))
                testlog = test_model(agent, world, memory_value, 1, batch_size, eps=1e-10, device=device)
                testlog_l.append(testlog)
    
            epoch_loss += loss.item()
            epoch_loss_poster += loss_poster.item()
            epoch_loss_prior += loss_prior.item()
                    
        epoch_loss = epoch_loss / (seq_len * batch_size)
        epoch_loss_poster = epoch_loss_poster / (seq_len * batch_size)
        epoch_loss_prior = epoch_loss_prior / (seq_len * batch_size)
        
    mean_total_reward = total_reward.mean().item()
                    
    if train:
        print('train loss: {:.4f}, rew: {:.1f}'.format(epoch_loss, mean_total_reward))
        if not writer is None:
            writer.add_scalar('train loss', epoch_loss, epoch_num)
            writer.add_scalar('train total reward', mean_total_reward, epoch_num)
            writer.add_scalar('train loss posterior', epoch_loss_poster, epoch_num)
            writer.add_scalar('train loss prior', epoch_loss_prior, epoch_num)
    else:
        print('Test loss: {:.4f}, rew: {:.1f}'.format(epoch_loss, mean_total_reward))
        if not writer is None:
            writer.add_scalar('test loss', epoch_loss, epoch_num)
            writer.add_scalar('test total reward', mean_total_reward, epoch_num)

    log = {'reward': torch.stack(reward_l, dim=0), 
            'mean_total_reward': mean_total_reward,
            'loss': epoch_loss,
            'inp1': torch.stack(inp1_l, dim=0), 
            'inp2': torch.stack(inp2_l, dim=0),
            'state1': torch.stack(state1_l, dim=0), 
            'state2': torch.stack(state2_l, dim=0), 
            'memory_value': memory_value_l[-1],
            'all_memory_value': memory_value_l[-1::-truncate][::-1] if save_all_memory_value else [],
            'pos': torch.stack(pos_l, dim=0),
            'rel': torch.stack(rel_l, dim=0),
            'testlog': testlog_l}
    
    return epoch_loss, mean_total_reward, log

#%%

def test_model(agent, world, memory_value, seq_len, batch_size, nsample=1, eps=1e-10, device='cpu'):

    agent.to(device)    

    inp1_l = []
    inp2_l = []
    reward_l = []
    pos_l = []
    rel_l = []
    conf_l = []
    
    total_reward = torch.zeros(batch_size)
            
    memory_value = memory_value.to(device)        
        
    start_time = time.time()

    for t0 in range(0, seq_len):

        valid = True

        inp1, inp2, reward_func, truth = world.get_test_data()
        inp1 = inp1.to(device)
        inp2 = inp2.to(device)
        
        # infer the states from the two inputs
        
        state1_inf = agent.memory_infer(memory_value, inp1)
        state2_inf = agent.memory_infer(memory_value, inp2)
        
        # perform transitive inference for given number of samples and
        # determine the most likely relation 

        max_ninf = world.dataset.env_size - 1     
        loglik_tbl = torch.zeros(batch_size, nsample, agent.relation_dim, max_ninf).to(device) 
        prior_tbl = torch.zeros(batch_size, nsample, agent.relation_dim, max_ninf).to(device) 
        memory_value_aug = memory_value.unsqueeze(1).repeat(1, nsample, 1, 1).reshape(memory_value.size(0) * nsample, memory_value.size(1), memory_value.size(2))
        for a in range(agent.relation_dim):   
            state_cur = state1_inf.unsqueeze(1).repeat(1, nsample, 1)
            state_dst = state2_inf.unsqueeze(1).repeat(1, nsample, 1)
            for i in range(max_ninf):        
                state_next_mu, state_next_sig = agent.transit_all(state_cur)
                log_prior = agent.transit_log_prior_all(state_cur)
                # log_likeli = torch.zeros(batch_size, nsample, agent.relation_dim).to(device) 
                f = - torch.log(state_next_sig[a] + eps) - 0.5 * ((state_dst - state_next_mu[a]) / (state_next_sig[a] + eps)) ** 2 
                loglik_tbl[:,:,a,i] = torch.sum(f, dim=2) 
                prior_tbl[:,:,a,i] = torch.softmax(log_prior, dim=2)[:,:,a]
                
                # for next iteration
                
                # state_cur = utils.sample_normal(state_next_mu[a], state_next_sig[a])
                state_cur = state_next_mu[a]
                state_cur = agent.clean_state(memory_value_aug, state_cur.reshape(batch_size * nsample, -1)).reshape(batch_size, nsample, -1)
                
        mx = loglik_tbl.reshape(batch_size, -1).max(dim=1).values.reshape(batch_size, 1, 1, 1)        
        ti_likeli_tbl = torch.exp(loglik_tbl - mx)
        ti_prior_tbl = torch.zeros(batch_size, nsample, agent.relation_dim, max_ninf).to(device) 
        for a in range(agent.relation_dim):   
            ti_prior_cur = torch.ones(batch_size, nsample).to(device) 
            for i in range(max_ninf):        
                ti_prior_cur = ti_prior_cur * prior_tbl[:,:,a,i] 
                ti_prior_tbl[:,:,a,i] = ti_prior_cur
                
        ti_prior_tbl = ti_prior_tbl / torch.sum(ti_prior_tbl, dim=2, keepdim=True) # normalize over relation
        ti_poster = ti_likeli_tbl * ti_prior_tbl                  
        ti_poster2 = ti_poster.mean(dim=1) # average over sample
        ti_poster3 = ti_poster2.sum(dim=2) # sum over iteration
        ti_poster4 = ti_poster3 / ti_poster3.sum(dim=1, keepdim=True) # normalize over relation
        conf, rel = torch.max(ti_poster4, dim=1)            

        # get reward

        reward = reward_func(rel.detach().cpu())
                                    
        # print('reward: {:.4f}, inference score: {:.4f}'.format(reward.item(), reward.item() * conf.item()))

        # logging
        
        reward_l.append(reward.detach().cpu())
        inp1_l.append(inp1.detach().cpu())
        inp2_l.append(inp2.detach().cpu())
        pos_l.append(truth[1])
        rel_l.append(rel.detach().cpu())
        conf_l.append(conf.detach().cpu())
        
    # mean_total_reward = sum([r.item() for r in reward_l]) / seq_len
    # mean_conf = sum([c.item() for c in conf_l]) / seq_len
    # mean_inf_score = sum([r.item() * c.item() for (r,c) in zip(reward_l, conf_l)]) / seq_len
                    
    end_time = time.time()
    
    # print('Elapsed time: {:.4f}s'.format(end_time - start_time))
    # print('total reward: {:.4f}, inference score: {:.4f}'.format(mean_total_reward, mean_inf_score))
    # print('{}, {}, {}, '.format(mean_total_reward, mean_conf, mean_inf_score))
    # print('{},'.format(mean_inf_score))

    log = {'reward': torch.stack(reward_l, dim=0), 
           'inp1': torch.stack(inp1_l, dim=0), 
           'inp2': torch.stack(inp2_l, dim=0),
           'memory_value': memory_value.detach().cpu(),
           'pos': torch.stack(pos_l, dim=0),
           'rel': torch.stack(rel_l, dim=0),
           'conf': torch.stack(conf_l, dim=0)}

    return log

#%%

def plot_reward_hist(reward_l):
    z = scipy.signal.convolve2d(np.stack(reward_l,axis=0).squeeze(), np.ones((200,1))/200,mode='valid')
    plt.plot(np.arange(0,len(z)),z)
    
def plot_single_norm_hist(norm_l):
    plt.plot(np.arange(0,len(norm_l)),np.array([n.item() for n in norm_l]))
    
def plot_multi_norm_hist(norm_l):
    plt.plot(np.arange(0,len(norm_l)),np.array([n.norm(dim=1).detach().numpy() for n in norm_l]))
    
    
    