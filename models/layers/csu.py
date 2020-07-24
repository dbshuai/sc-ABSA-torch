"""
用于计算两个表示的交互后各自的表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Cross_Shared_Unit(nn.Module):
    def __init__(self, m_hidden_size, g_hidden_size):
        super(Cross_Shared_Unit, self).__init__()
        self.m_hidden_size = m_hidden_size
        self.g_hidden_size = g_hidden_size
        self.G_aspect_polarity = nn.Parameter(torch.randn(g_hidden_size,m_hidden_size, g_hidden_size))
        self.G_polarity_aspect = nn.Parameter(torch.randn(g_hidden_size, m_hidden_size,g_hidden_size))
        self.G_vector_polarity = nn.Parameter(torch.randn(m_hidden_size,1))
        self.G_vector_aspect = nn.Parameter(torch.randn(m_hidden_size,1))

    def forward(self,aspect_hidden,polarity_hidden,max_pooling=True):
        batch = aspect_hidden.size()[0]
        seq_len = aspect_hidden.size()[1]
        G_aspect_polarity = self.G_aspect_polarity.view(self.g_hidden_size,-1) #g_hidden_size,m_hidden_size*g_hidden_size
        G_aspect_polarity = G_aspect_polarity.unsqueeze(0).repeat(batch,1,1) # batch,g_hidden_size,m_hidden_size*g_hidden_size
        shared_hidden_aspect_polarity = torch.matmul(aspect_hidden, G_aspect_polarity) #batch,seq_len,g_hidden_size x batch,g_hidden_size,m_hidden_size*g_hidden_size
        #batch,seq_len,m_hidden_size*g_hidden_size
        shared_hidden_aspect_polarity = shared_hidden_aspect_polarity.view(-1,seq_len*self.m_hidden_size,self.g_hidden_size)
        polarity_hidden_transpose = polarity_hidden.permute(0,2,1) #batch,g_hidden_size,seq_len
        shared_hidden_aspect_polarity = torch.tanh(torch.matmul(shared_hidden_aspect_polarity, polarity_hidden_transpose))
        #batch,seq_len*m_hidden_size,seq_len
        shared_hidden_aspect_polarity = shared_hidden_aspect_polarity.view(-1,seq_len,self.m_hidden_size,seq_len)
        if max_pooling:
            shared_hidden_aspect_polarity,_ = torch.max(shared_hidden_aspect_polarity, dim=-2)
        else:
            shared_hidden_aspect_polarity = shared_hidden_aspect_polarity.permute(0,1,3,2)
            shared_hidden_aspect_polarity = shared_hidden_aspect_polarity.contiguous().view(-1,seq_len*seq_len,self.m_hidden_size)
            G_vector_aspect = self.G_vector_aspect.unsqueeze(0).repeat(batch,1,1)
            #batch,m_hidden_size,1
            shared_hidden_aspect_polarity = torch.matmul(shared_hidden_aspect_polarity, G_vector_aspect)
            #batch,seq_len*seq_len,1
        aspect_vector = shared_hidden_aspect_polarity.view(-1,seq_len,seq_len)

        G_polarity_aspect = self.G_polarity_aspect.view(self.g_hidden_size,-1)
        G_polarity_aspect = G_polarity_aspect.unsqueeze(0).repeat(batch,1,1)
        shared_hidden_polarity_aspect = torch.matmul(polarity_hidden, G_polarity_aspect)
        shared_hidden_polarity_aspect = shared_hidden_polarity_aspect.view(-1,seq_len*self.m_hidden_size,self.g_hidden_size)
        aspect_hidden_transpose = aspect_hidden.permute(0,2,1)
        shared_hidden_polarity_aspect = torch.tanh(torch.matmul(shared_hidden_polarity_aspect, aspect_hidden_transpose))
        shared_hidden_polarity_aspect = shared_hidden_polarity_aspect.view(-1,seq_len,self.m_hidden_size,seq_len)
        if max_pooling:
            shared_hidden_polarity_aspect,_ = torch.max(shared_hidden_polarity_aspect, dim=-2)
        else:
            shared_hidden_polarity_aspect = shared_hidden_polarity_aspect.permute(0, 1, 3, 2)
            shared_hidden_polarity_aspect = shared_hidden_polarity_aspect.contiguous().view(-1,seq_len*seq_len,self.m_hidden_size)
            G_vector_polarity = self.G_vector_polarity.unsqueeze(0).repeat(batch,1,1)
            shared_hidden_polarity_aspect = torch.matmul(shared_hidden_polarity_aspect, G_vector_polarity)
        polarity_vector = shared_hidden_polarity_aspect.view(-1,seq_len,seq_len)

        # Get attention vector
        aspect_attention_vector = F.softmax(aspect_vector, dim=-1)
        polarity_attention_vector = F.softmax(polarity_vector, dim=-1)

        aspect_hidden_v = torch.matmul(aspect_attention_vector, polarity_hidden)
        polarity_hidden_v = torch.matmul(polarity_attention_vector, aspect_hidden)

        aspect_hidden = aspect_hidden + aspect_hidden_v
        polarity_hidden = polarity_hidden + polarity_hidden_v

        aspect_hidden = aspect_hidden.view(-1,seq_len,self.g_hidden_size)
        polarity_hidden = polarity_hidden.view(-1,seq_len,self.g_hidden_size)

        return aspect_hidden, polarity_hidden


