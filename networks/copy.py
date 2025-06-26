from functools import partial
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from .network_utils import (
    Classifier, 
    ResBlock, 
    ConvNormAct,
    convert_to_rpm_matrix_v9,
    convert_to_rpm_matrix_v6,
    convert_to_rpm_matrix_mnr,
    LinearNormAct
)

from .HCVARR import HCVARR
from .SCAR import RelationNetworkSCAR
from .Pred import Pred
from .MM import MM
from .MRnet import MRNet

class SymbolEncoding(nn.Module):
    def __init__(self, num_contexts=4, d_model=32, f_len=24):
        super(SymbolEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, d_model, num_contexts, f_len))
        nn.init.xavier_uniform_(self.position_embeddings)

    def forward(self):
        return self.position_embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        return x + self.pe[:, :x.size(1), :]

class PredictionAttention(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1, token_len=24):
        super(PredictionAttention, self).__init__()
        self.kernel = nn.Sequential(nn.Linear(d_model*2, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.m = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.num_heads=nhead
        self.head_dim=d_model//nhead

        self.position_prompt = SymbolEncoding()
        self.rule_prompt = SymbolEncoding(4, d_model, 24)
        self.pre_prompt = SymbolEncoding(4, d_model, 24)
        self.p = nn.Sequential(ConvNormAct(64, 32, 3, 1), nn.Linear(24, 6))


    def forward(self, x, atten_flag):
        b, c, t, l = x.shape
        prompt = self.position_prompt()
        r = self.rule_prompt().expand(b, -1, -1, -1)
        pre_prompt = self.pre_prompt().expand(b, -1, -1, -1)
        prompt = prompt.expand(b, -1, -1, -1)
        if atten_flag == 1:
            tar = torch.cat([prompt[:,:,:,:18], x[:,:,:,18:]], dim=3)
            con = torch.cat([x[:,:,:,:18], prompt[:,:,:,18:]], dim=3)
        elif atten_flag == 2:
            tar = torch.cat([prompt[:,:,:,:12], x[:,:,:,12:18], prompt[:,:,:,18:]], dim=3)
            con = torch.cat([x[:,:,:,:12], prompt[:,:,:,12:18], x[:,:,:,18:]], dim=3)
        elif atten_flag == 3:
            tar = torch.cat([prompt[:,:,:,:6], x[:,:,:,6:12], prompt[:,:,:,12:]], dim=3)
            con = torch.cat([x[:,:,:,:6], prompt[:,:,:,6:12], x[:,:,:,12:]], dim=3)
        else:
            tar = torch.cat([x[:,:,:,:6], prompt[:,:,:,6:]], dim=3)
            con = torch.cat([prompt[:,:,:,:6], x[:,:,:,6:]], dim=3)

        
        con, tar, rul = con.permute(0,2,3,1), tar.permute(0,2,3,1), r.permute(0,2,3,1)
        tc_kernel = F.normalize(self.kernel(torch.cat([con,tar], dim=-1)).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)

        q = F.normalize(self.q(tar).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)
        k = F.normalize(self.k(con).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)
        v = F.normalize(self.v(rul).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)

        q, k = q*tc_kernel, k*tc_kernel

        atten = self.drop(q @ k.transpose(-2, -1))
        atten = F.softmax(atten / math.sqrt(self.head_dim), dim=-1)
        r = (atten @ v)

        r = self.m(r.permute(0,1,3,2,4).reshape(b,t,l,c)).permute(0,3,1,2)
        con = con.permute(0,3,1,2)

        if atten_flag == 1:
            con = torch.cat([con[:,:,:,:18], pre_prompt[:,:,:,18:]], dim=3)
        elif atten_flag == 2:
            con = torch.cat([con[:,:,:,:12], pre_prompt[:,:,:,12:18], con[:,:,:,18:]], dim=3)
        elif atten_flag == 3:
            con = torch.cat([con[:,:,:,:6], pre_prompt[:,:,:,6:12], con[:,:,:,12:]], dim=3)
        else:
            con = torch.cat([pre_prompt[:,:,:,:6], con[:,:,:,6:]], dim=3)

        p = self.p(torch.cat([con,r], dim=1).contiguous())

        if atten_flag == 1:
            x = torch.cat([x[:,:,:,:18], p], dim=-1)
        elif atten_flag == 2:
            x = torch.cat([x[:,:,:,:12], p, x[:,:,:,18:]], dim=-1)
        elif atten_flag == 3:
            x = torch.cat([x[:,:,:,:6], p, x[:,:,:,12:]], dim=-1)
        else:
            x = torch.cat([p, x[:,:,:,6:]], dim=-1)

        return x.permute(0,2,3,1)

class GatedPredictionAttentionBlock(nn.Module):

    def __init__(
        self, 
        in_planes, 
        dropout = 0.0, 
        num_heads = 8,
        token_len=25,
    ):
        super().__init__()
        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)
        self.lp1 = nn.Linear(in_planes, in_planes*2)
        self.lp2 = nn.Linear(in_planes, in_planes)
        self.m = nn.Linear(in_planes, in_planes)
        self.drop = nn.Dropout(dropout)
        
        self.pre_att = PredictionAttention(in_planes, num_heads, token_len=token_len)
        self.conv1 = nn.Sequential(ConvNormAct(4, 4*4, 3, 1, activate=True), ConvNormAct(4*4, 4, 3, 1, activate=True))
        self.conv2 = nn.Sequential(ConvNormAct(in_planes, in_planes*4, 3, 1, activate=True), ConvNormAct(in_planes*4, in_planes, 3, 1, activate=True))
    
    def forward(self, x, atten_flag):
        shortcut = self.downsample(x)
        x = F.normalize(x.permute(0,2,3,1), dim=-1)
        g, x = self.lp1(x).chunk(2, dim=-1)
        g = self.m(self.conv1(g))
        x = x.permute(0,3,1,2)
        x = self.pre_att(x, atten_flag)

        x = self.lp2(F.gelu(g)*x)
        x = self.conv2(x.permute(0,3,1,2).contiguous())
        x = self.drop(x) + shortcut
        return x

class PredictiveReasoningBlock(nn.Module):

    def __init__(
        self, 
        in_planes, 
        ou_planes,
        number_gp = 16, 
        dropout = 0.1, 
        num_heads = 8
    ):

        super().__init__()
        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)
        self.m = nn.Linear(25, 24)
        self.number_gp = number_gp
        for l in range(number_gp):
            setattr(
                self, "gp"+str(l), 
                GatedPredictionAttentionBlock(in_planes, num_heads=num_heads)
            )
        for l in range(4):
            setattr(
                self, "atten"+str(l),
                CroAttention(32)
            )
        self.fuse = ConvNormAct(in_planes*4, in_planes, 1, 0)

    def forward(self, x):
        b,_,_,_ = x.size()
        x = self.m(x)

        x1 = getattr(self, "gp"+str(0))(x, 1)
        x1 = getattr(self, "gp"+str(1))(x1, 2)
        x1 = getattr(self, "gp"+str(2))(x1, 3)
        x1 = getattr(self, "gp"+str(3))(x1, 4)
        e1 = F.relu(x - x1)
        x2 = getattr(self, "gp"+str(4))(x, 2)
        x2 = getattr(self, "gp"+str(5))(x2, 3)
        x2 = getattr(self, "gp"+str(6))(x2, 1)
        x2 = getattr(self, "gp"+str(7))(x2, 4)
        e2 = F.relu(x - x2)
        x3 = getattr(self, "gp"+str(8))(x, 3)
        x3 = getattr(self, "gp"+str(9))(x3, 2)
        x3 = getattr(self, "gp"+str(10))(x3, 4)
        x3 = getattr(self, "gp"+str(11))(x3, 1)
        e3 = F.relu(x - x3)
        x4 = getattr(self, "gp"+str(12))(x, 4)
        x4 = getattr(self, "gp"+str(13))(x4, 3)
        x4 = getattr(self, "gp"+str(14))(x4, 2)
        x4 = getattr(self, "gp"+str(15))(x4, 1)
        e4 = F.relu(x - x4)
        x1 = getattr(self, "atten"+str(0))(e1, x1)
        x2 = getattr(self, "atten"+str(1))(e2, x2)
        x3 = getattr(self, "atten"+str(2))(e3, x3)
        x4 = getattr(self, "atten"+str(3))(e4, x4)
        xp = self.fuse(torch.cat([x1,x2,x3,x4], dim=1))
        e = F.relu(x-xp)
        return e

class SelfAttention(nn.Module):
    def __init__(
        self,
        in_planes,
        dropout = 0.1,
        num_heads = 8
    ): 
        super().__init__()
        self.q = nn.Linear(in_planes, in_planes)
        self.kv = nn.Linear(in_planes, in_planes*2)
        self.num_heads=num_heads
        self.head_dim=in_planes//num_heads
        self.m = nn.Linear(in_planes, in_planes)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        b,t,l,c = x.shape
        shortcut = x
        q = F.normalize(self.q(x).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)
        k, v = self.kv(x).reshape(b, t, l, self.num_heads*2, self.head_dim).permute(0, 1, 3, 2, 4).chunk(2, dim=2)
        k, v = F.normalize(k, dim=-1), F.normalize(v, dim=-1)

        atten = self.drop(q @ k.transpose(-2, -1))
        atten = F.softmax(atten / math.sqrt(self.head_dim), dim=-1)
        x = (atten @ v)

        x = self.m(x.permute(0,1,3,2,4).reshape(b,t,l,c))
        return x

class CroAttention(nn.Module):
    def __init__(
        self,
        in_planes,
        dropout = 0.1,
        num_heads = 8
    ): 
        super().__init__()
        self.q = nn.Linear(in_planes, in_planes)
        self.kv = nn.Linear(in_planes, in_planes*2)
        self.num_heads=num_heads
        self.head_dim=in_planes//num_heads
        self.m = nn.Linear(in_planes, in_planes)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, e, x):
        shortcut = x
        x, e = x.permute(0,2,3,1), e.permute(0,2,3,1)
        b,t,l,c = x.shape
        q = F.normalize(self.q(e).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)
        k, v = self.kv(x).reshape(b, t, l, self.num_heads*2, self.head_dim).permute(0, 1, 3, 2, 4).chunk(2, dim=2)
        k, v = F.normalize(k, dim=-1), F.normalize(v, dim=-1)

        atten = self.drop(q @ k.transpose(-2, -1))
        atten = F.softmax(atten / math.sqrt(self.head_dim), dim=-1)
        x = (atten @ v)

        x = self.m(x.permute(0,1,3,2,4).reshape(b,t,l,c)).permute(0,3,1,2)+shortcut
        return x


class Alignment(nn.Module):

    def __init__(
        self,
        in_planes,
        ou_planes,
        dropout = 0.1,
        num_heads = 8,
        ffn=True
    ): 
        super().__init__()
        self.selfatten = SelfAttention(in_planes)
        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)
        self.m = nn.Sequential(nn.Linear(in_planes, ou_planes), nn.LayerNorm(ou_planes), nn.GELU())
        self.position1 = PositionalEncoding(in_planes)
        self.position2 = PositionalEncoding(in_planes)
        self.position3 = PositionalEncoding(in_planes)
        self.position4 = PositionalEncoding(in_planes)
        self.position5 = PositionalEncoding(in_planes)
        self.position6 = PositionalEncoding(in_planes)
        self.position7 = PositionalEncoding(in_planes)
        self.position8 = PositionalEncoding(in_planes)
        self.position9 = PositionalEncoding(in_planes)
        self.ffn = ffn
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        b,c,t,l = x.shape
        shortcut = self.downsample(x)
        x = x.permute(0,2,3,1)
        c1, c2, c3, c4 = self.position1(x[:,0]), self.position2(x[:,1]), self.position3(x[:,2]), self.position4(x[:,3])
        x = torch.stack([c1, c2, c3, c4], dim=1)
        x = self.selfatten(x).permute(0,3,1,2)
        out = self.drop(x)+shortcut
        if self.ffn:
            out = self.m(out.permute(0,2,3,1)).permute(0,3,1,2)
        return out
    

class PredRNet(nn.Module):

    def __init__(self, num_filters=48, block_drop=0.0, classifier_drop=0.0, 
                 classifier_hidreduce=1.0, in_channels=1, num_classes=8, 
                 num_extra_stages=1, reasoning_block=PredictiveReasoningBlock,
                 num_contexts=5):

        super().__init__()

        channels = [num_filters, num_filters*2, num_filters*3, num_filters*4]
        strides = [2, 2, 2, 2]

        # -------------------------------------------------------------------
        # frame encoder 

        self.in_planes = in_channels

        for l in range(len(strides)):
            setattr(
                self, "res"+str(l), 
                self._make_layer(
                    channels[l], stride=strides[l], 
                    block=ResBlock, dropout=block_drop,
                )
            )
        # -------------------------------------------------------------------

        

        # -------------------------------------------------------------------
        # predictive coding 
        self.num_contexts = num_contexts
        self.atten = Alignment(32,128)
        self.think_branches = 1
        self.channel_reducer = ConvNormAct(128, 32, 1, 0, activate=False)

        for l in range(self.think_branches):
            setattr(
                self, "MAutoRR"+str(l), 
                PredictiveReasoningBlock(32, 32, num_heads=8)
            )
        # -------------------------------------------------------------------

        self.featr_dims = 1024

        self.classifier = Classifier(
            self.featr_dims, 1, 
            norm_layer = nn.BatchNorm1d, 
            dropout = classifier_drop, 
            hidreduce = classifier_hidreduce
        )

        self.in_channels = in_channels
        self.ou_channels = num_classes


    def _make_layer(self, planes, stride, dropout, block, downsample=True):
        if downsample and block == ResBlock:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size = 2, stride = stride) if stride != 1 else nn.Identity(),
                ConvNormAct(self.in_planes, planes, 1, 0, activate = False, stride=1),
            )
        else:
            downsample = nn.Identity()

        if block == ResBlock:
            stage = block(self.in_planes, planes, downsample, stride = stride, dropout = dropout)

        self.in_planes = planes

        return stage

    def forward(self, x, train=False):
        if self.in_channels == 1:
            b, n, h, w = x.size()
            x = x.reshape(b*n, 1, h, w)
        elif self.in_channels == 3:
            b, n, _, h, w = x.size()
            x = x.reshape(b*n, 3, h, w)

        for l in range(4):
            x = getattr(self, "res"+str(l))(x)

        if self.num_contexts == 8:
            _, c, h, w = x.size()
            x = convert_to_rpm_matrix_v9(x, b, h, w)
        elif self.num_contexts == 3:
            _, c, h, w = x.size()
            x = convert_to_rpm_matrix_mnr(x, b, h, w)
        else:
            _, c, h, w = x.size()
            x = convert_to_rpm_matrix_v6(x, b, h, w)

        x = x.reshape(b * self.ou_channels, self.num_contexts + 1, -1, h*w)
        x = x.permute(0,2,1,3)
        x = self.channel_reducer(x)
        
        e = getattr(self, "MAutoRR"+str(0))(x)
        x = self.atten(e)

        x = x.reshape(b, self.ou_channels, -1)
        x = F.adaptive_avg_pool1d(x, 1024)

        x = x.reshape(b * self.ou_channels, self.featr_dims)

        out = self.classifier(x)

        return out.view(b, self.ou_channels)
    

def predrnet_raven(**kwargs):
    return PredRNet(**kwargs, num_contexts=8)

def predrnet_analogy(**kwargs):
    return PredRNet(**kwargs, num_contexts=5, num_classes=4)

def predrnet_mnr(**kwargs):
    return PredRNet(**kwargs, num_contexts=3)

def hcvarr(**kwargs):
    return HCVARR(**kwargs, num_contexts=5, num_classes=4)

def scar(**kwargs):
    return RelationNetworkSCAR(**kwargs, num_contexts=5, num_classes=4)

def pred(**kwargs):
    return Pred(**kwargs, num_contexts=5, num_classes=4)

def mm(**kwargs):
    return MM(**kwargs, num_contexts=5, num_classes=4)

def mrnet(**kwargs):
    return MRNet(**kwargs, num_contexts=5, num_classes=4)