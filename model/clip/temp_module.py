import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, einsum

from collections import OrderedDict
from typing import Tuple, Union

from einops import repeat



# Yeseong - added for DALLE-2
from collections import namedtuple
def l2norm(t):
    return F.normalize(t, dim = -1)


class ScalarEmbedding(nn.Module):
    def __init__(self, config_vocab_size, dim, eos):
        super().__init__()
        self.config_vocab_size = config_vocab_size
        self.dim = dim
        self.hypercfg_embedding = nn.Embedding(
                config_vocab_size + 1, dim)
        nn.init.normal_(self.hypercfg_embedding.weight, std=0.02)
        
        self.eos = eos
        if self.eos:
            self.cls_token = nn.Parameter(torch.randn(1,1, dim))

    def forward(self, x):
        b, _ = x.shape
        mask = torch.isnan(x)

        # Assign the token for each element by its index and for nan elements by token 0
        ind = torch.arange(x.shape[1], device=x.device).add_(1).unsqueeze(0).expand(x.shape[0], -1)
        token = ind.clone().detach()
        token[mask] = 0

        # Remove the nan elements
        x_ = x.clone().detach()
        x_[mask] = 0
        
        emb = self.hypercfg_embedding(token) * x_.unsqueeze(-1)
        
        if self.eos:
            cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
            emb = torch.cat([emb, cls_tokens], dim= 1)
            
        return emb


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(
            self, layers, output_dim, heads,
            image_channels=3, input_resolution=256, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(image_channels, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # print(self.attn_mask.shape) 4, 4
        # print(x.shape) 2 64 256
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, image_channels:int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.image_channels = image_channels
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
                in_channels=image_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CHIP(nn.Module):
    def __init__(self,
                 embed_dim: int, # image, text will return this dimensional embedding
                 # vision
                 image_channels: int,
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # hyperconfig
                 context_length: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length
        self.n_channels = image_channels

        if isinstance(vision_layers, (tuple, list)):
            print("Configured ResNet CHIP")
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                image_channels=image_channels,
                width=vision_width
            )
        else:
            print("Configured ViT CHIP")
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                image_channels=image_channels,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )


        # Original code for text encoding
        #self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        #self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))

        # For hyperconfigs,
        # we assume vocab_size (# of embedding entries) == context_length (size in attention),
        # as it will be highly limited in a few tens or hundreds at max
        # We also don't use the positional embedding by assuming that configs do not have meaning in appeared orders
        self.config_embed = ScalarEmbedding(self.context_length, transformer_width)
        self.ln_final = LayerNorm(transformer_width)

        self.hypercfg_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        #nn.init.normal_(self.token_embedding.weight, std=0.02)
        #nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.hypercfg_projection is not None:
            nn.init.normal_(self.hypercfg_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, hypercfg, return_no_proj=False):
        # Original code for text encoding
        #x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        #x = x + self.positional_embedding.type(self.dtype)

        x = self.config_embed(hypercfg).type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        
        projected = x[torch.arange(x.shape[0]), hypercfg.argmax(dim=-1)] @ self.hypercfg_projection

        if return_no_proj:
            return projected, x

        return projected

    def forward(self, image, hypercfg):
        image_features = self.encode_image(image)
        hypercfg_features = self.encode_text(hypercfg)

        return image_features, hypercfg_features 

    # Yeseong: added for dalle2
    @property
    def dim_latent(self):
        return self.ln_final.weight.shape[0]

    @property
    def image_size(self):
        return self.visual.input_resolution

    @property
    def image_channels(self):
        return self.n_channels

    @property
    def max_text_len(self):
        return self.context_length



def load_chip(model, path, device=None):
    # Find the latest chip model
    # filenames = list(glob(os.path.join(path, 'chip/*.pt')))
    # steps = [int(x.replace(".pt", "").split('_')[-1]) for x in filenames]
    # chippath = filenames[np.argmax(steps)]
    # print_log("CHIP Path: " + chippath)
    
    chippath = path
    # Load the weights
    checkpoint = torch.load(chippath, map_location=device)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    #model = model.to('cuda')
    model.eval()
    return model





# Mean embedding clip
class CLIP_MM(nn.Module):
    def __init__(self,
                  width = 128, # dimension of latent vector
                  layers = 4, # the number of ResidualAttension block 
                  heads = 4, # the number of head of attension
                  length = 128, # the range of timesteps
                  m_in = 4, # input channel size of metric
                  c_in = 5, # input channel size of counter
                  use_attn = None, # if None => no use attn mask
                  eos = False
                 ):
        super(CLIP_MM, self).__init__()
        
        # setting
        self.context_length = 19
        self.n_channels = c_in
        self.width = width
        

        # metric part
        self.m_embedding = ScalarEmbedding(config_vocab_size=101, dim=128, eos=eos)
        self.ln_final = LayerNorm(width)
        self.hypercfg_projection = nn.Parameter(torch.empty(width, width))
        self.m_transformer = Transformer(width=width, layers=layers, heads=heads, attn_mask=self.build_attention_mask()) # maybe it does not use attn? => no time relationship
        # self.m_transformer = Transformer(width=width, layers=layers, heads=heads) # maybe it does not use attn? => no time relationship
        
        # event part
        from model.clip.clip_embed import DataEmbedding
        from model.clip.clip_components import EVTransformer
        self.c_embedding = DataEmbedding(c_in, width)
        self.c_transformer = EVTransformer(width, layers, heads, use_attn) # event count transformer
        
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

    
    @property
    def dtype(self):
        return self.c_transformer.conv1.weight.dtype
            
    
    def encode_metric(self, hypercfg, return_no_proj=False):
        # Original code for text encoding
        #x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        #x = x + self.positional_embedding.type(self.dtype)

        # x = self.m_embedding(hypercfg).type(self.dtype)
        x = self.m_embedding(hypercfg)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.m_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_final(x).type(self.dtype)
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        
        # projected = x[torch.arange(x.shape[0]), hypercfg.argmax(dim=-1)] @ self.hypercfg_projection
        projected = x[torch.arange(x.shape[0]), -1] @ self.hypercfg_projection

        if return_no_proj:
            return projected, x

        return projected, x
    
    
    def encode_event(self, count):
        embeded_count = self.c_embedding(count)
        encoded_count = self.c_transformer(embeded_count) # b, dim, timesteps
        
        encoded_count = einsum('ijk -> ikj', encoded_count) # b, timesteps, dim 
        
        return encoded_count[:,-1,:], encoded_count
            
            
    def forward(self, metric, event, sw_cls=False):
        metric_fetures_one, metric_features = self.encode_metric(metric)
        event_fetures_one, event_fetures = self.encode_event(event)
        
        return metric_features, event_fetures, metric_fetures_one, event_fetures_one
    

    def embed_text_with_l2norm(self, metrics, sw_cls=False):    
        projected, x  = self.encode_metric(metrics)
        return l2norm(projected), l2norm(x)