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

        self.position_prompt = SymbolEncoding(9, d_model, 24)
        self.rule_prompt = SymbolEncoding(9, d_model, 24)
        self.pre_prompt = SymbolEncoding(9, d_model, 24)
        self.p = nn.Sequential(ConvNormAct(64, 32, 3, 1), nn.Linear(24, 6))
        self.slow_r = CroAttention(d_model)
        self.lp = nn.Linear(6,24)


    def forward(self, x, atten_flag):
        b, c, t, l = x.shape
        prompt = self.position_prompt()
        r = self.rule_prompt().expand(b, -1, -1, -1)
        pre_prompt = self.pre_prompt().expand(b, -1, -1, -1)
        prompt = prompt.expand(b, -1, -1, -1)
        if atten_flag == 1:
            tar = torch.cat([prompt[:,:,:,:18], x[:,:,:,18:]], dim=3)
            tp = x[:,:,:,18:]
            con = torch.cat([x[:,:,:,:18], prompt[:,:,:,18:]], dim=3)
        elif atten_flag == 2:
            tar = torch.cat([prompt[:,:,:,:12], x[:,:,:,12:18], prompt[:,:,:,18:]], dim=3)
            tp = x[:,:,:,12:18]
            con = torch.cat([x[:,:,:,:12], prompt[:,:,:,12:18], x[:,:,:,18:]], dim=3)
        elif atten_flag == 3:
            tar = torch.cat([prompt[:,:,:,:6], x[:,:,:,6:12], prompt[:,:,:,12:]], dim=3)
            tp = x[:,:,:,6:12]
            con = torch.cat([x[:,:,:,:6], prompt[:,:,:,6:12], x[:,:,:,12:]], dim=3)
        else:
            tar = torch.cat([x[:,:,:,:6], prompt[:,:,:,6:]], dim=3)
            tp = x[:,:,:,:6]
            con = torch.cat([prompt[:,:,:,:6], x[:,:,:,6:]], dim=3)

        
        con, tar, rul = con.permute(0,2,3,1), tar.permute(0,2,3,1), r.permute(0,2,3,1)
        # tc_kernel = F.normalize(self.kernel(torch.cat([con,tar], dim=-1)).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)

        q = F.normalize(self.q(tar).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)
        k = F.normalize(self.k(con).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)
        v = F.normalize(self.v(rul).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)

        # q, k = q*tc_kernel, k*tc_kernel

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
        e = F.relu(tp)-p
        e = self.lp(e)
        # x = self.slow_r(self.lp(e), x)
        
        # if atten_flag == 1:
        #     ex = torch.cat([x[:,:,:,:18], e], dim=-1)
        # elif atten_flag == 2:
        #     ex = torch.cat([x[:,:,:,:12], e, x[:,:,:,18:]], dim=-1)
        # elif atten_flag == 3:
        #     ex = torch.cat([x[:,:,:,:6], e, x[:,:,:,12:]], dim=-1)
        # else:
        #     ex = torch.cat([e, x[:,:,:,6:]], dim=-1)
        # x = self.slow_r(ex, x)

        return e


class SlowGatedPredictionAttentionBlock(nn.Module):

    def __init__(
        self, 
        in_planes, 
        dropout = 0.0, 
        num_heads = 8,
        token_len=25,
        num_contexts=3
    ):
        super().__init__()
        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)
        self.lp1 = nn.Linear(in_planes, in_planes*2)
        self.lp2 = nn.Linear(in_planes, in_planes)
        self.m = nn.Linear(in_planes, in_planes)
        self.drop = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(dropout)
        
        self.pre_att = PredictionAttention(in_planes, num_heads, token_len=token_len)
        self.conv1 = nn.Sequential(ConvNormAct((num_contexts+1), (num_contexts+1)*4, 3, 1, activate=True), ConvNormAct((num_contexts+1)*4, (num_contexts+1), 3, 1, activate=True))
        self.conv2 = nn.Sequential(ConvNormAct(in_planes, in_planes*4, 3, 1, activate=True), ConvNormAct(in_planes*4, in_planes, 3, 1, activate=True))
        self.slow_r = CroAttention(in_planes)
    
    def forward(self, x, atten_flag):
        shortcut = self.downsample(x)
        x = F.normalize(x.permute(0,2,3,1), dim=-1)
        g, x = self.lp1(x).chunk(2, dim=-1)
        g = self.m(self.conv1(g))
        x = x.permute(0,3,1,2)
        e = self.pre_att(x, atten_flag).permute(0,2,3,1).contiguous()
        e = self.lp2(F.gelu(g)*e).permute(0,3,1,2).contiguous()
        x = self.slow_r(e,x).contiguous()
        x = self.conv2(x)
        x = self.drop(x)+shortcut
        return x

class FastInduction(nn.Module):

    def __init__(
        self,
        in_planes,
        num_heads = 8,
        num_contexts=3
    ):
        super().__init__()
        self.m = nn.Linear(in_planes, in_planes*2)
    
    def forward(self, x):
        fx, g = self.m(x.permute(0, 2, 3, 1).contiguous()).chunk(2, dim=-1)
        fx, g = fx.contiguous(), g.contiguous()
        return fx, g
    
class SlowInduction(nn.Module):
    def __init__(
        self,
        in_planes,
        num_contexts=4,
        num_heads = 8):
        super().__init__()
        self.m = nn.Linear(in_planes, in_planes)
        self.slow_p = SlowGatedPredictionAttentionBlock(in_planes, num_heads=num_heads, num_contexts=num_contexts)
        self.g = nn.Sequential(ConvNormAct((num_contexts+1), (num_contexts+1)*4, 3, 1, activate=True), ConvNormAct((num_contexts+1)*4, (num_contexts+1), 3, 1, activate=True), nn.Linear(in_planes, in_planes))
    
    def forward(self, x, sx, fx, g, i):
        g = self.g(g)
        fx = self.m(F.gelu(g) * fx).permute(0, 3, 1, 2).contiguous()
        b,t,c,l = fx.shape
        sx = self.slow_p(sx, i+1)+fx
        return sx


class PredictiveReasoningBlock(nn.Module):

    def __init__(
        self, 
        in_planes, 
        ou_planes,
        steps = 4, 
        dropout = 0.1, 
        num_heads = 8,
        num_contexts=8
    ):

        super().__init__()
        self.m = nn.Linear(25, 24)
        self.fast_induction = FastInduction(in_planes, num_contexts=num_contexts)
        self.steps = steps

        for l in range(steps):
            setattr(
                self, "slow"+str(l),
                SlowInduction(in_planes,num_contexts=num_contexts)
            )

    def forward(self, x):
        b, _, _, _ = x.size()
        x = self.m(x)
        fx,g = self.fast_induction(x)
        sx = fx.permute(0, 3, 1, 2).contiguous()

        for i in range(self.steps):
            sx = getattr(self, "slow"+str(i))(x,sx,fx,g,i).contiguous()

        return sx

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
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, x):
        b,t,l,c = x.shape
        shortcut = x
        q = F.normalize(self.q(x).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)
        k, v = self.kv(x).reshape(b, t, l, self.num_heads*2, self.head_dim).permute(0, 1, 3, 2, 4).chunk(2, dim=2)
        k, v = F.normalize(k, dim=-1), F.normalize(v, dim=-1)

        atten = self.drop1(q @ k.transpose(-2, -1))
        atten = F.softmax(atten / math.sqrt(self.head_dim), dim=-1)
        x = (atten @ v)

        x = self.drop2(self.m(x.permute(0,1,3,2,4).reshape(b,t,l,c)))
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
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.gating_mlp = nn.Sequential(
            nn.Linear(in_planes, in_planes),
            nn.Tanh(),
            nn.Linear(in_planes, num_heads)
        )

        self.gating_mlp1 = nn.Sequential(
            nn.Linear(in_planes, in_planes),
            nn.Tanh(),
            nn.Linear(in_planes, num_heads)
        )


    def forward(self, e, x):
        shortcut = x
        x, e = x.permute(0,2,3,1), e.permute(0,2,3,1)
        b,t,l,c = x.shape

        pooled_e = e.mean(dim=2) 
        gate_logits = self.gating_mlp(pooled_e) 
        gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(-1).unsqueeze(-1)

        pooled_e1 = e.mean(dim=2) 
        gate_logits1 = self.gating_mlp(pooled_e1) 
        gate_weights1 = F.softmax(gate_logits1, dim=-1).unsqueeze(-1).unsqueeze(-1)


        q = self.q(e).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        q = F.normalize(q, dim=-1)
        k, v = self.kv(x).reshape(b, t, l, self.num_heads*2, self.head_dim).permute(0, 1, 3, 2, 4).chunk(2, dim=2)
        k, v = F.normalize(k, dim=-1), F.normalize(v, dim=-1)
        v = v*gate_weights

        atten = self.drop1(q @ k.transpose(-2, -1))
        atten = F.softmax(atten / math.sqrt(self.head_dim), dim=-1)
        x = (atten @ v)*gate_weights1

        x = self.drop2(self.m(x.permute(0,1,3,2,4).reshape(b,t,l,c)).permute(0,3,1,2))
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
    
    def forward(self, x, num_contexts):
        b,c,t,l = x.shape
        shortcut = self.downsample(x)
        x = x.permute(0,2,3,1)

        if num_contexts == 3:
            c1, c2, c3, c4 = self.position1(x[:,0]), self.position2(x[:,1]), self.position3(x[:,2]), self.position4(x[:,3])
            x = torch.stack([c1, c2, c3, c4], dim=1)
        elif num_contexts == 8:
            c1, c2, c3, c4, c5, c6, c7, c8, c9 = self.position1(x[:,0]), self.position2(x[:,1]), self.position3(x[:,2]), self.position4(x[:,3]), self.position1(x[:,4]), self.position2(x[:,5]), self.position3(x[:,6]), self.position4(x[:,7]), self.position4(x[:,8])
            x = torch.stack([c1, c2, c3, c4, c5, c6, c7, c8, c9], dim=1)
        x = self.selfatten(x).permute(0,3,1,2)
        out = self.drop(x)+shortcut
        if self.ffn:
            out = self.m(out.permute(0,2,3,1)).permute(0,3,1,2)
        return out
    
class GPRB(nn.Module):

    def __init__(
        self, 
        in_planes, 
        ou_planes,
        stride = 1, 
        dropout = 0.1, 
        num_contexts = 8
    ):

        super().__init__()

        self.stride = stride

        md_planes = ou_planes*4
        self.pconv1 = ConvNormAct(in_planes, in_planes, (2, 1))
        self.pconv2 = ConvNormAct(in_planes, in_planes, (2, 1))
        self.pconv3 = ConvNormAct(in_planes, in_planes, (2, 1))
        self.conv1 = ConvNormAct(in_planes, md_planes, 3, 1)
        self.conv2 = ConvNormAct(md_planes, ou_planes, 3, 1)
        self.conv = nn.Sequential(ConvNormAct((num_contexts+1), (num_contexts+1)*4, 3, 1), ConvNormAct((num_contexts+1)*4, (num_contexts+1), 3, 1))
        self.drop = nn.Dropout(dropout) if dropout > .0 else nn.Identity()
        self.lp = nn.Linear(in_planes, in_planes*2)
        self.m = nn.Linear(in_planes, in_planes)
        self.m1 = nn.Linear(in_planes, in_planes)

        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)

    def forward(self, x):
        
        b, c, t, l = x.size()
        identity = self.downsample(x)
        g, x = self.lp(x.permute(0,2,3,1)).chunk(2, dim=-1)
        g = self.m(self.conv(g.contiguous()))
        x = x.permute(0,3,1,2).contiguous()
        
        b, c, t, l = x.size()
        contexts1, choices1 = x[:,:,:2], x[:,:,2:3]
        predictions1 = self.pconv1(contexts1)
        prediction_errors1 = F.relu(choices1) - predictions1

        contexts2, choices2 = x[:,:,3:5], x[:,:,5:6]
        predictions2 = self.pconv2(contexts2)
        prediction_errors2 = F.relu(choices2) - predictions2

        contexts3, choices3 = x[:,:,6:8], x[:,:,8:]
        predictions3 = self.pconv3(contexts3)
        prediction_errors3 = F.relu(choices3) - predictions3
        
        out = torch.cat((contexts1, prediction_errors1, contexts2, prediction_errors2, contexts3, prediction_errors3), dim=2)
        out = self.m1(out.permute(0,2,3,1)*F.gelu(g)).permute(0,3,1,2).contiguous()
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.drop(out)
        out = out + identity
        
        return out


class PredRNet(nn.Module):

    def __init__(self, num_filters=48, block_drop=0.0, classifier_drop=0.0, 
                 classifier_hidreduce=1.0, in_channels=1, num_classes=8, 
                 num_extra_stages=1, reasoning_block=PredictiveReasoningBlock,
                 num_contexts=8):

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
                PredictiveReasoningBlock(32, 32, num_heads=8, num_contexts=self.num_contexts)
            )
        
        for l in range(3):
            setattr(
                self, "PRB"+str(l), 
                GPRB(32, 32, num_contexts=self.num_contexts)
            )
        # -------------------------------------------------------------------

        self.featr_dims = 1024

        self.classifier = Classifier(
            self.featr_dims, 1, 
            norm_layer = nn.BatchNorm1d, 
            dropout = classifier_drop, 
            hidreduce = classifier_hidreduce
        )
        self.classifier1 = Classifier(
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

        for i in range(1):
            x = getattr(self, "PRB"+str(i))(x)

        out1 = x.reshape(b, self.ou_channels, -1)
        out1 = F.adaptive_avg_pool1d(out1, 1024)
        out1 = out1.reshape(b * self.ou_channels, self.featr_dims)
        out1 = self.classifier1(out1)

        
        x = getattr(self, "MAutoRR"+str(0))(x)
        x = self.atten(x, self.num_contexts)

        x = x.reshape(b, self.ou_channels, -1)
        x = F.adaptive_avg_pool1d(x, 1024)

        x = x.reshape(b * self.ou_channels, self.featr_dims)

        out = self.classifier(x)

        return out.view(b, self.ou_channels), out1.view(b, self.ou_channels)
    

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