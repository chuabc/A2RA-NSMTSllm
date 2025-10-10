import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from layers.ada_dec_IN import adaptive_decomp_Normalization, PredModule
from layers.TCN import TemporalConvNet
from sklearn.svm import SVC

class Generator(nn.Module):
    def __init__(self, configs, depth=3, forward_drop_rate=0.5, attn_drop_rate=0.5, period_list=None):
        super(Generator, self).__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model

        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 2
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        self.blocks = Gen_TransformerEncoder(
                                depth=self.depth,
                                emb_size = self.d_model,
                                drop_p = self.attn_drop_rate,
                                forward_drop_p=self.forward_drop_rate
                                )
        self.in_layer = nn.ModuleList()
        self.period_list = period_list
        self.LP = len(self.period_list)

        for i in range(self.LP):
            self.in_layer.append(nn.Linear(self.period_list[i], self.d_model))
        self.patch_num = math.ceil(self.seq_len / self.patch_size)
        self.in_layer.append(nn.Linear(self.patch_size, self.d_model))

    def forward(self, x):
        output = []
        input_emb=[]
        for i in range(self.LP + 1):
            emb = self.in_layer[i](x[i])
            input_emb.append(emb)
            out = self.blocks(emb)
            output.append(out)
        return output, input_emb
    

class Text_Generator(nn.Module):
    def __init__(self, configs, depth=3, forward_drop_rate=0.5, attn_drop_rate=0.5, period_list=None):
        super(Text_Generator, self).__init__()
        self.depth = depth
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        self.blocks = Gen_TransformerEncoder(
                                depth=self.depth,
                                emb_size = self.d_model,
                                drop_p = self.attn_drop_rate,
                                forward_drop_p=self.forward_drop_rate
                                )
        self.out_layer = nn.ModuleList()
        self.period_list = period_list
        self.LP = len(self.period_list)
        self.seq_patch_num = np.ceil(self.seq_len / self.period_list).astype(int)

        for i in range(self.LP):
            self.out_layer.append(nn.Linear(self.d_model, self.period_list[i]))
        self.patch_num = math.ceil(self.seq_len / self.patch_size)
        self.out_layer.append(nn.Linear(self.d_model, self.patch_size))        

    def forward(self, x):
        output = []
        for i in range(self.LP + 1):
            dec_emb = self.blocks(x[i])
            out = self.out_layer[i](dec_emb)
            output.append(out)
        return output


class Generator_decoder(nn.Module):
    def __init__(self, configs, device, period_list=None):
        super().__init__()
        self.device = device
        self.period_list = period_list
        self.d_model = configs.d_model
        self.patch_size = configs.patch_size
        self.tcn_dec_blocks = nn.ModuleList()
        self.LP = len(self.period_list)
        self.tcn_blocks = TemporalConvNet(self.d_model, [self.d_model * 2, self.d_model * 4, self.d_model * 2, self.d_model]).to(self.device)
        
    def forward(self, output):
        dec_gen = []
        trend_dec_emb = self.tcn_blocks(output[0])
        dec_gen.append(trend_dec_emb)
        for i in range(self.LP):
            dec_emb = self.tcn_blocks(output[i+1])
            dec_gen.append(dec_emb)
        return dec_gen
        

class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=8,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

        
class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)])       
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

        
        
class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size=768,
                 num_heads=8,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=768, n_classes=1):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out      


class Discriminator(nn.Module):
    def __init__(self, emb_size=768, n_classes=1, depth=3, **kwargs):
        super(Discriminator, self).__init__()
        self.attention = Dis_TransformerEncoder(depth, emb_size=emb_size//4, drop_p=0.5, forward_drop_p=0.5, **kwargs)
        self.fc1 = nn.Linear(emb_size, emb_size//2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(emb_size//2, emb_size//4)
        self.classification = ClassificationHead(emb_size//4, n_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.attention(out)
        out = self.classification(out)
        return out