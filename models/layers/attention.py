# -*- coding: utf-8 -*-
# file: attention.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class NoQueryAttention(Attention):
    '''q is a parameter'''
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1, dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)


epsilon = 1e-5
"""
缺少mask
带有距离信息的self-attention
"""
class SelfAttention(nn.Module):
    def __init__(self, input_size,use_opinion):
        super(SelfAttention, self).__init__()
        self.use_opinion = use_opinion
        self.W = nn.Linear(input_size, input_size)

    def forward(self, x,gold_opinion,predict_opinion,gold_prob):
        device = x.device
        steps = x.size(-2)
        x_tran = self.W(x)
        x_transpose = x.permute(0, 2, 1)
        weights = torch.matmul(x_tran, x_transpose)
        location = np.abs(
            np.tile(np.array(range(steps)), (steps, 1)) - np.array(range(steps)).reshape(steps, 1))
        location = torch.FloatTensor(location).to(device)
        loc_weights = 1.0/(location+epsilon)
        weights = weights * loc_weights

        if self.use_opinion:
            #gold_opinion:0,1,2:O,BP，IP
            #predict_opinion:0,1,2,3,4:O,BA,IA,BP,IP
            gold_opinion_ = gold_opinion[:,:,1]+gold_opinion[:,:,2]
            predict_opinion_ = predict_opinion[:,:,3]+predict_opinion[:,:,4]
            opinion_weights = gold_prob*gold_opinion_ + (1-gold_prob)*predict_opinion_
            opinion_weights = opinion_weights.unsqueeze(-1)
            weights = weights*opinion_weights

        weights = torch.tanh(weights)
        weights = weights.exp()
        weights = weights * torch.Tensor(np.eye(steps) == 0).to(device)
        weights = weights/(torch.sum(weights,dim=-1,keepdim=True)+epsilon)
        return torch.matmul(weights,x)

class DotSentimentAttention(nn.Module):
    def __init__(self,input_size):
        super(DotSentimentAttention,self).__init__()
        self.W = nn.Parameter(torch.Tensor(input_size,))
        nn.init.normal_(self.W.data)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        eij = torch.sum(x*self.W,dim=-1)+self.bias
        a_exp = eij.exp()
        a_sigmoid = torch.sigmoid(a_exp)
        a_softmax = torch.softmax(eij,dim=-1)
        return a_softmax,a_sigmoid
