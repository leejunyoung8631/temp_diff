import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange


def exists(val):
        return val is not None

def default(val, d):
    return val if exists(val) else d  

'''

bring from informer git

'''






class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        pe = pe.permute(0,2,1) # for transposing
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:,:,:x.size(2)]



class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        # x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        x = self.tokenConv(x)
        return x

    

class ScalarEmbedding(nn.Module):
    def __init__(self, config_vocab_size, dim):
        super().__init__()
        self.config_vocab_size = config_vocab_size
        self.dim = dim
        self.hypercfg_embedding = nn.Embedding(
                config_vocab_size + 1, dim)
        nn.init.normal_(self.hypercfg_embedding.weight, std=0.02)

    def forward(self, x):
        # for sw_use
        # b, _, _ = x.shape
        # x = x.reshape(b, -1)
        
        mask = torch.isnan(x)
        # Assign the token for each element by its index and for nan elements by token 0

        ind = torch.arange(x.shape[1], device=x.device).add_(1).unsqueeze(0).expand(x.shape[0], -1)
        token = ind.clone().detach()
        token[mask] = 0

        # Remove the nan elements
        x_ = x.clone().detach()
        x_[mask] = 0
        

        return self.hypercfg_embedding(token) * x_.unsqueeze(-1)
    
    


class MeanTokenEmbed(nn.Module):
    def __init__(self, d_embed):
        super(MeanTokenEmbed, self).__init__()
        self.d_embed = d_embed
        n = 101 # For each channel, 0 to 100
        self.embed = nn.Embedding(num_embeddings=n, embedding_dim=d_embed)
        
        self.first_cls = nn.Parameter(torch.zeros(1, 1, d_embed))
        self.second_cls = nn.Parameter(torch.zeros(1, 1, d_embed))
        self.eos = nn.Parameter(torch.zeros(1, 1, d_embed))
        
        torch.nn.init.normal_(self.first_cls)
        torch.nn.init.normal_(self.second_cls)
 
    
    def forward(self, x, fw_cls=True, sw_cls=False):
        x = self.embed(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        b, n, d, = x.shape  # b, n, d = batch, n_token, dimension
        
        # concat second cls token
        if sw_cls:
            x = torch.concat([x[:, :(n//2)], self.second_cls.expand(b, -1, -1), x[:, (n//2):]], axis=1)
        # concat first cls token
        if fw_cls:
            x = torch.concat([self.first_cls.expand(b, -1, -1), x], axis=1)
            
        return x
    

        

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, timeemb=False):
        super(DataEmbedding, self).__init__()
        self.timeemb = timeemb

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, t_prev=None, t_next=None):
        out = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(out)
    
    
    
