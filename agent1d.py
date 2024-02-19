#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:26:47 2022

A python implementation of ARDMO (Abstract Relational Decision-making MOdel),
written by Haruo Hosoya.

@author: hahosoya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, dim=x.ndim-1)

class Constant(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return x * 0 + self.val
    
class Agent(nn.ModuleDict):

    def __init__(self, input_dim, state_dim, memory_dim, relation_dim, 
                 state_comp='inner', trans_nonlin='normalize',
                 trans_prior=True, trans_interm_dim = 10,
                 single_sig=True,
                 eps=1e-10):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.memory_dim = memory_dim
        self.axis_dim = 1
        self.relation_dim = relation_dim
        
        self.trans_interm_dim = trans_interm_dim
        
        self.state_comp = state_comp
        self.trans_nonlin = trans_nonlin
        self.trans_prior = trans_prior
        self.single_sig = single_sig
        
        self.eps = eps
        
        self.info = dict()

        self.add_transition()
        self.add_memory()
        
    def add_transition(self):
        
        self.transition_mu = nn.ModuleList()
        self.transition_sig = nn.ModuleList()
        
        if self.single_sig:
            self.global_sig = nn.Parameter(torch.randn(1))
        
        for i in range(self.relation_dim):
            if self.trans_nonlin == 'normalize':
                nl = Normalize()
            elif self.trans_nonlin == 'sigmoid':
                nl = nn.Sigmoid()
            self.transition_mu.append(nn.Sequential(
                nn.Linear(self.state_dim, self.state_dim),
                nl
                ))
            if self.single_sig:
                self.transition_sig.append(nn.Sequential(
                    Constant(self.global_sig),
                    nn.Softplus()
                    ))
            else:
                self.transition_sig.append(nn.Sequential(
                    nn.Linear(self.state_dim, self.trans_interm_dim),
                    nn.Sigmoid(),
                    nn.Linear(self.trans_interm_dim, self.state_dim),
                    nn.Softplus()
                    ))
        
        if self.trans_prior:
            self.transition_log_prior = nn.Sequential(
                nn.Linear(self.state_dim, self.trans_interm_dim),
                nn.Sigmoid(),
                nn.Linear(self.trans_interm_dim, self.relation_dim))
        
    def add_memory(self):
        
        self.memory_key = nn.Parameter(
            torch.sigmoid(torch.randn(self.memory_dim, self.state_dim) * 1))

    def transit_all(self, state):

        next_state_mu_l = []
        next_state_sig_l = []
        
        for reli in range(self.relation_dim):
            next_state_mu1 = self.transition_mu[reli](state)
            next_state_mu_l.append(next_state_mu1)
            next_state_sig1 = self.transition_sig[reli](state)
            next_state_sig_l.append(next_state_sig1)
            
        next_state_mu = torch.stack(next_state_mu_l, dim=0)
        next_state_sig = torch.stack(next_state_sig_l, dim=0)
        
        return (next_state_mu, next_state_sig)
    
    def transit_log_prior_all(self, state):
        if self.trans_prior:
            return self.transition_log_prior(state)
        else:
            return torch.zeros(state.size(0), relation_dim) 
                                
    def new_memory_value(self, batch_size, scale=0.001):
        return torch.randn(batch_size, self.input_dim, self.memory_dim, requires_grad=True) * scale    
    
    def memory_infer(self, memory_value, inp):
        mi = torch.matmul(memory_value.permute(0, 2, 1), inp.unsqueeze(2))
        mis = (mi + self.eps).softmax(dim=1)
        st = torch.matmul(self.memory_key.unsqueeze(0).permute(0, 2, 1), mis).squeeze(2)
        return st

    def memory_retrieve(self, memory_value, state):
        if self.state_comp == 'inner':
            ms = torch.matmul(self.memory_key.unsqueeze(0), state.unsqueeze(2))
        elif self.state_comp == 'L2':
            ms = - torch.sum((self.memory_key.unsqueeze(0) - state.unsqueeze(2).permute(0, 2, 1)) ** 2, dim=2).unsqueeze(2)
        elif self.state_comp == 'norm_inner':
            ms = torch.matmul(F.normalize(self.memory_key.unsqueeze(0), dim=2), state.unsqueeze(2))

        mss = (ms + self.eps).softmax(dim=1)
        return torch.matmul(memory_value, mss).squeeze(2)
        
    def memory_update(self, memory_value, inp, inp_old, state, alpha):
        p = (inp - inp_old).unsqueeze(2)
        if self.state_comp == 'inner':
            q = torch.matmul(self.memory_key.unsqueeze(0), state.unsqueeze(2))
        elif self.state_comp == 'L2':            
            q = - torch.sum((self.memory_key.unsqueeze(0) - state.unsqueeze(2).permute(0, 2, 1)) ** 2, dim=2).unsqueeze(2)
        elif self.state_comp == 'norm_inner':
            q = torch.matmul(F.normalize(self.memory_key.unsqueeze(0), dim=2), state.unsqueeze(2))
            
        qs = (q + self.eps).softmax(dim=1)
        memory_value_out = memory_value + alpha.unsqueeze(1).unsqueeze(2) * torch.matmul(p, qs.permute(0, 2, 1)) 
        return memory_value_out

    def clean_state(self, memory_value, state):
        inp = self.memory_retrieve(memory_value, state)
        return self.memory_infer(memory_value, inp)
    
    

def save_agent(agent, path):
    agent.cpu()
    s = { 'input_dim': agent.input_dim,
          'state_dim': agent.state_dim,
          'memory_dim': agent.memory_dim,
          'relation_dim': agent.relation_dim,
          'state_comp': agent.state_comp,
          'trans_nonlin': agent.trans_nonlin,
          'state_dict': agent.state_dict(),
          'trans_interm_dim': agent.trans_interm_dim,
          'trans_prior': agent.trans_prior,
          'info': agent.info,
          }    
    torch.save(s, path)

def load_agent(path, opts=None):
    s = torch.load(path)
    agent = Agent(s['input_dim'], 
                  s['state_dim'], 
                  s['memory_dim'], 
                  s['relation_dim'],
                  state_comp=s['state_comp'],
                  trans_nonlin=s['trans_nonlin'],
                  trans_interm_dim=s['trans_interm_dim'],
                  trans_prior=s['trans_prior'])
    agent.info = s['info']
    agent.load_state_dict(s['state_dict'])
    return agent

