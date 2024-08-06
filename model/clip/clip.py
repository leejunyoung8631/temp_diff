import numpy as np

import torch
from torch import nn
from torch import nn, einsum

from model.clip.clip_components import *
from model.clip.clip_embed import DataEmbedding, MeanTokenEmbed, ScalarEmbedding



# normal clip
class CLIP_XAV(nn.Module):
    def __init__(self,
                  width = 128, # dimension of latent vector
                  layers = 4, # the number of ResidualAttension block 
                  heads = 4, # the number of head of attension
                  length = 128, # the range of timesteps
                  m_in = 4, # input channel size of metric
                  c_in = 5, # input channel size of counter
                 ):
        super(CLIP_XAV, self).__init__()
        
        self.n_channels = c_in
        self.width = width
        
        # embed model
        # value embedding & positional embedding for each step
        self.m_embedding = DataEmbedding(m_in, width)
        self.c_embedding = DataEmbedding(c_in, width)
        
        # Transformer model
        self.m_transformer = Transformer(width, layers, heads) # Topdown metric transformer
        self.c_transformer = Transformer(width, layers, heads) # event count transformer
        
        # logit_scale for contrastive learning
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # some weight init
        self.initialize_parameters()
        
        
    def initialize_parameters(self):
        # embed weight init
        # nn.init.normal_(self.m_embedding.value_embedding.weight, std=0.02)
        # nn.init.normal_(self.m_embedding.position_embedding.weight, std=0.01)
        # nn.init.normal_(self.c_embedding.value_embedding.weight, std=0.02)
        # nn.init.normal_(self.c_embedding.position_embedding.weight, std=0.01)
        
        # linear weight init
        # 일단 conv1d는 init 안하고 linear만 골라서 init.
        proj_std = (self.m_transformer.width ** -0.5) * ((2 * self.m_transformer.layers) ** -0.5)
        attn_std = self.m_transformer.width ** -0.5
        fc_std = (2 * self.m_transformer.width) ** -0.5
        
        for block in self.m_transformer.resblocks:
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.c_transformer.resblocks:
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        
        # if isinstance(self.visual, ModifiedResNet):
        #     if self.visual.attnpool is not None:
        #         std = self.visual.attnpool.c_proj.in_features ** -0.5
        #         nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

        #     for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
        #         for name, param in resnet_block.named_parameters():
        #             if name.endswith("bn3.weight"):
        #                 nn.init.zeros_(param)

        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # if self.hypercfg_projection is not None:
        #     nn.init.normal_(self.hypercfg_projection, std=self.transformer.width ** -0.5)
        
        
    def embed_data(self, metric, count):
        embeded_metric = self.m_embedding(metric)
        embeded_count = self.c_embedding(count)
        
        return embeded_metric, embeded_count
        
        
    def forward(self, metric, count):
        embeded_metric, embeded_count = self.embed_data(metric, count)
        
        encoded_metric = self.m_transformer(embeded_metric)
        encoded_count = self.c_transformer(embeded_count)
        
        # b, timesteps, dim / ex) 64 64 128
        encoded_metric = einsum('ijk -> ikj', encoded_metric)
        encoded_count = einsum('ijk -> ikj', encoded_count) 
        
        encoded_metric_one, encoded_count_one = encoded_metric[:,-1,:], encoded_count[:,-1,:]
        
        return encoded_metric, encoded_count, encoded_metric_one, encoded_count_one
    
    
    
    # need for prior
    # return [projected, not projected]
    def encode_metric(self, metric, return_no_proj=True):
        embeded_metric = self.m_embedding(metric) # 4 4 64
        encoded_metric = self.m_transformer(embeded_metric) # 4 128 2 -> 4 128 128 ->
        
        encoded_metric = einsum('ijk -> ikj', encoded_metric)
        
        return encoded_metric[:,-1,:], encoded_metric
    
    
    def encode_event(self, count):
        embeded_count = self.c_embedding(count)
        encoded_count = self.c_transformer(embeded_count)
        
        encoded_count = einsum('ijk -> ikj', encoded_count) 
        
        return encoded_count[:,-1,:], encoded_count
    

    def embed_text_with_l2norm(self, metric):
        hypercfg_embed, hypercfg_encodings = self.encode_metric(metric, return_no_proj=True)
        return l2norm(hypercfg_embed.float()), l2norm(hypercfg_encodings.float())
    
    
    
    # the below is legacy
    def embed_text(self, metric):
        hypercfg_embed, hypercfg_encodings = self.encode_metric(metric, return_no_proj=True)
        return l2norm(hypercfg_embed.float()), hypercfg_encodings.float()

    def embed_image(self, count):
        image_embed, _ = self.encode_event(count)
        return l2norm(image_embed.float()), None
    
    
    # need for checking condition for prior code
    @property
    def image_channels(self):
        return self.n_channels
    
    @property
    def dim_latent(self):
        return self.width
    


# Mean embedding clip
class CLIP_MEAN(nn.Module):
    def __init__(self,
                  width = 128, # dimension of latent vector
                  layers = 4, # the number of ResidualAttension block 
                  heads = 4, # the number of head of attension
                  length = 128, # the range of timesteps
                  m_in = 4, # input channel size of metric
                  c_in = 5, # input channel size of counter
                  use_attn = None # if None => no use attn mask
                 ):
        super(CLIP_MEAN, self).__init__()
        
        # setting
        self.context_length = 1 + m_in
        self.n_channels = c_in
        self.width = width
        
        # embed for metric using scala conv
        self.m_embedding = MeanTokenEmbed(d_embed=width)
        
            
        self.m_embedding = ScalarEmbedding(config_vocab_size=101, dim=128)
        self.ln_final = TokenLayerNorm(width)
        self.hypercfg_projection = nn.Parameter(torch.empty(width, width))
        # self.m_transformer = MetricTransformer(width=width, layers=layers, heads=heads, attn_mask=self.build_attention_mask())
        self.m_transformer = MetricTransformer(width=width, layers=layers, heads=heads) # maybe it does not use attn? => no time relationship
        
        # embed for event using 1d conv
        self.c_embedding = DataEmbedding(c_in, width)
        self.c_transformer = Transformer(width, layers, heads, use_attn) # event count transformer
        
        # logit_scale for contrastive learning
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
        # some weight init
        self.initialize_parameters()
        
        
    def initialize_parameters(self):
        # linear weight init
        proj_std = (self.m_transformer.width ** -0.5) * ((2 * self.m_transformer.layers) ** -0.5)
        attn_std = self.m_transformer.width ** -0.5
        fc_std = (2 * self.m_transformer.width) ** -0.5
        
        for block in self.m_transformer.resblocks:
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.c_transformer.resblocks:
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        
        # if isinstance(self.visual, ModifiedResNet):
        #     if self.visual.attnpool is not None:
        #         std = self.visual.attnpool.c_proj.in_features ** -0.5
        #         nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
        #         nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

        #     for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
        #         for name, param in resnet_block.named_parameters():
        #             if name.endswith("bn3.weight"):
        #                 nn.init.zeros_(param)

        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.hypercfg_projection is not None:
            nn.init.normal_(self.hypercfg_projection, std=self.m_transformer.width ** -0.5)
            
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
            
    
    def encode_metric(self, metric, sw_cls=False):        
        x = self.m_embedding(metric, sw_cls=sw_cls).type(torch.float)
        # x = self.m_embedding(metric).type(torch.float)
        
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.m_transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        
        # get CLS token
        projected = x[torch.arange(x.shape[0]), 0] @ self.hypercfg_projection    
        return projected
    
    
    def encode_event(self, count):
        embeded_count = self.c_embedding(count)
        encoded_count = self.c_transformer(embeded_count) # b, dim, timesteps
        
        encoded_count = einsum('ijk -> ikj', encoded_count) # b, timesteps, dim 
        
        return encoded_count[:,-1,:], encoded_count
            
            
    def forward(self, metric, event, sw_cls=False):
        metric_features = self.encode_metric(metric, sw_cls=sw_cls)
        event_fetures_one, event_fetures = self.encode_event(event)
        
        return metric_features, event_fetures_one, event_fetures
    

    def embed_text_with_l2norm(self, metrics, sw_cls=False):        
        return l2norm(self.encode_metric(metrics, sw_cls=sw_cls))


