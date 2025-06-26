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
from .position_embedding import PositionalEncoding, LearnedAdditivePositionalEmbed
from .HCVARR import HCVARR
from .SCAR import RelationNetworkSCAR
from .Pred import Pred
from .MM import MM
from .MRnet import MRNet
from torch.nn import init

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 4, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 4, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, t, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, t, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, t, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('btid,btjd->btij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps

            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('btjd,btij->btid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, t, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots

# class ImplicitSlotAttention(nn.Module):
#     def __init__(self, dim, num_heads=4, num_slots=8, iters=3, eps=1e-8, gate_logit_normalizer=8.0):
#         super().__init__()
#         self.iters = iters
#         self.num_heads = num_heads
#         self.num_slots = num_slots
#         self.eps = eps
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.gate_logit_normalizer = gate_logit_normalizer

#         self.to_q = nn.Linear(dim, dim)
#         self.to_k = nn.Linear(dim, dim)
#         self.to_v = nn.Linear(dim, dim)
#         self.to_f = nn.Linear(dim, num_heads * num_slots)

#         self.gru = nn.GRUCell(self.head_dim, self.head_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(self.head_dim, self.head_dim),
#             nn.ReLU(),
#             nn.Linear(self.head_dim, self.head_dim)
#         )

#         self.norm_inputs = nn.LayerNorm(dim)
#         self.norm_heads = nn.LayerNorm(self.head_dim)
#         self.norm_pre_ff = nn.LayerNorm(self.head_dim)

#     def forward(self, x):
#         # x: [B, T, N, D]
#         B, T, N, D = x.shape
#         x = self.norm_inputs(x)

#         k = self.to_k(x)  # [B, T, N, D]
#         v = self.to_v(x)  # [B, T, N, D]
#         f = self.to_f(x).view(B, T, N, self.num_heads, self.num_slots)  # gate logits
#         f = F.logsigmoid(f) / self.gate_logit_normalizer
#         s = 1 - f.exp()  # [B, T, N, H, S]

#         q = self.to_q(x).view(B, T, N, self.num_heads, self.head_dim)  # use data itself as initial slots

#         for _ in range(self.iters):
#             q_norm = self.norm_heads(q)  # [B, T, N, H, Dh]
#             k_ = k.view(B, T, N, self.num_heads, self.head_dim)
#             v_ = v.view(B, T, N, self.num_heads, self.head_dim)

#             # [B, T, H, S, N]
#             attn_logits = torch.einsum('btnhd,btnhd->bthn', q_norm, k_) * self.scale
#             attn = F.softmax(attn_logits, dim=-1) + self.eps
#             attn = attn / attn.sum(dim=-1, keepdim=True)

#             gated_attn = attn * s.permute(0, 1, 3, 4, 2)  # [B, T, H, S, N]

#             updates = torch.einsum('bthn,btnhd->bthd', gated_attn, v_)  # [B, T, H, Dh]

#             q_reshaped = q.view(B * T * N * self.num_heads, self.head_dim)
#             updates_reshaped = updates.view(B * T * self.num_heads, self.head_dim)
#             q_updated = self.gru(updates_reshaped, q_reshaped[:updates_reshaped.size(0)])
#             q = q_updated.view(B, T, self.num_heads, self.head_dim)

#             q = q + self.mlp(self.norm_pre_ff(q))

#         return q  # [B, T, num_heads, D_h]

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


    def forward(self, q, kv):
        shortcut = q
        b,t,l,c = q.shape
        b_,t_,l_,c_ = b,t,l,c

        q = self.q(q).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        q = F.normalize(q, dim=-1)

        b,t,l,c = kv.shape
        k, v = self.kv(kv).reshape(b, t, l, self.num_heads*2, self.head_dim).permute(0, 1, 3, 2, 4).chunk(2, dim=2)
        k, v = F.normalize(k, dim=-1), F.normalize(v, dim=-1)

        atten = q @ k.transpose(-2, -1)
        atten = self.drop1(F.softmax(atten / math.sqrt(self.head_dim), dim=-1))
        x = (atten @ v)

        x = self.drop2(self.m(x.permute(0,1,3,2,4).reshape(b_,t_,l_,c_)))+shortcut
        return x


class FusionAttention(nn.Module):
    def __init__(
        self,
        in_planes,
        dropout = 0.1,
        num_heads = 8
    ): 
        super().__init__()
        self.q = nn.Linear(in_planes, in_planes)
        self.k = nn.Linear(in_planes, in_planes)
        self.v = nn.Linear(in_planes, in_planes)
        self.num_heads=num_heads
        self.head_dim=in_planes//num_heads
        self.m = nn.Linear(in_planes, in_planes)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)


    def forward(self, q, k, v):
        shortcut = q
        b,t,l,c = q.shape
        b_,t_,l_,c_ = b,t,l,c

        q = self.q(q).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        q = F.normalize(q, dim=-1)

        b,t,l,c = k.shape
        k = self.k(k).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = F.normalize(k, dim=-1)

        v = self.v(v).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = F.normalize(v, dim=-1)

        atten = q @ k.transpose(-2, -1)
        atten = self.drop1(F.softmax(atten / math.sqrt(self.head_dim), dim=-1))
        x = (atten @ v)

        x = self.drop2(self.m(x.permute(0,1,3,2,4).reshape(b_,t_,l_,c_)))+shortcut
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
        self.ffn = ffn
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, num_contexts):
        b,t,l,c = x.shape
        shortcut = x
        x = self.selfatten(x)
        out = self.drop(x)+shortcut
        if self.ffn:
            out = self.m(out)
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
        self.pconv1 = ConvNormAct(in_planes, in_planes, (3, 1))
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
        
        x = x.permute(0,3,1,2)
        b, c, t, l = x.size()
        identity = self.downsample(x)
        g, x = self.lp(x.permute(0,2,3,1)).chunk(2, dim=-1)
        g = self.m(self.conv(g.contiguous()))
        x = x.permute(0,3,1,2).contiguous()
        
        contexts1, choices1 = x[:,:,:3], x[:,:,3:]
        predictions1 = self.pconv1(contexts1)
        prediction_errors1 = F.relu(choices1) - predictions1

        # contexts2, choices2 = x[:,:,3:5], x[:,:,5:6]
        # predictions2 = self.pconv2(contexts2)
        # prediction_errors2 = F.relu(choices2) - predictions2

        # contexts3, choices3 = x[:,:,6:8], x[:,:,8:]
        # predictions3 = self.pconv3(contexts3)
        # prediction_errors3 = F.relu(choices3) - predictions3
        
        # out = torch.cat((contexts1, prediction_errors1, contexts2, prediction_errors2, contexts3, prediction_errors3), dim=2)
        out = torch.cat((contexts1, prediction_errors1), dim=2)
        out = self.m1(out.permute(0,2,3,1)*F.gelu(g)).permute(0,3,1,2).contiguous()
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.drop(out)
        out = out + identity
        
        return out.permute(0,2,3,1)

class GIPRB(nn.Module):

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
        for l in range(8):
            setattr(
                self, "pconv"+str(l), 
                nn.Linear(20, 5)
            )

        self.conv1 = ConvNormAct(in_planes, md_planes, 3, 1)
        self.conv2 = ConvNormAct(md_planes, ou_planes, 3, 1)
        self.conv = nn.Sequential(ConvNormAct((num_contexts+1), (num_contexts+1)*4, 3, 1), ConvNormAct((num_contexts+1)*4, (num_contexts+1), 3, 1))
        self.drop = nn.Dropout(dropout) if dropout > .0 else nn.Identity()
        self.lp = nn.Linear(in_planes, in_planes*2)
        self.m = nn.Linear(in_planes, in_planes)
        self.m1 = nn.Linear(in_planes, in_planes)

        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)

    def forward(self, x):
        
        b, t, s, c = x.size()
        identity = x
        g, x = self.lp(x).chunk(2, dim=-1)
        g = self.m(self.conv(g.contiguous()))
        x = x.permute(0, 1, 3, 2).contiguous()
        
        b, t, c, l = x.size()
        xs = torch.chunk(x, 5, dim=-1)
        all_errors = []
        for i, xi in enumerate(xs):
            context = torch.cat([xj for j, xj in enumerate(xs) if j != i], dim=-1)
            pred = getattr(self, "pconv"+str(i))(context)
            pe = F.relu(xi) - pred  # shape: [b, c, 1, t]
            all_errors.append(pe)

        out = torch.cat(all_errors, dim=-1)
        out = self.m1(out.permute(0,1,3,2)*F.gelu(g)).permute(0,3,2,1).contiguous()
        # out = self.conv1(out)
        # out = self.conv2(out)
        out = self.drop(out)
        out = out.permute(0,3,2,1).contiguous() + identity
        
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
        reduced_channel = 32
        self.position1 = LearnedAdditivePositionalEmbed(reduced_channel)
        self.position2 = LearnedAdditivePositionalEmbed(reduced_channel)
        self.position3 = LearnedAdditivePositionalEmbed(reduced_channel)
        self.position4 = LearnedAdditivePositionalEmbed(reduced_channel)
        self.position5 = LearnedAdditivePositionalEmbed(reduced_channel)
        self.position6 = LearnedAdditivePositionalEmbed(reduced_channel)
        self.position7 = LearnedAdditivePositionalEmbed(reduced_channel)
        self.position8 = LearnedAdditivePositionalEmbed(reduced_channel)
        self.position9 = LearnedAdditivePositionalEmbed(reduced_channel)

        # -------------------------------------------------------------------
        # predictive coding 
        self.num_contexts = num_contexts
        self.atten = Alignment(reduced_channel,128)
        self.slotA1 = SlotAttention(num_slots=8, dim=reduced_channel)
        self.slotA2 = SlotAttention(num_slots=8, dim=reduced_channel)
        self.slotA3 = SlotAttention(num_slots=8, dim=reduced_channel)

        self.channel_reducer = ConvNormAct(128, reduced_channel, 1, 0, activate=False)
        self.fuse1 = FusionAttention(reduced_channel)
        self.fuse2 = FusionAttention(reduced_channel)
        self.fuse3 = FusionAttention(reduced_channel)

        self.croslt1 = CroAttention(reduced_channel)
        self.croslt2 = CroAttention(reduced_channel)
        self.croslt3 = CroAttention(reduced_channel)
        self.cro1 = CroAttention(reduced_channel)
        self.cro2 = CroAttention(reduced_channel)
        self.cro3 = CroAttention(reduced_channel)

        for l in range(3):
            setattr(
                self, "PRB"+str(l), 
                GPRB(reduced_channel, reduced_channel, num_contexts=self.num_contexts)
            )

        for l in range(3):
            setattr(
                self, "GIPRB"+str(l), 
                GIPRB(reduced_channel, reduced_channel, num_contexts=self.num_contexts)
            )
        # -------------------------------------------------------------------

        self.featr_dims = 1024
        self.mlp = nn.Linear(1024, 1024)

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
        x = x.permute(0,2,3,1)
        if self.num_contexts == 3:
            c1, c2, c3, c4 = self.position1(x[:,0]), self.position2(x[:,1]), self.position3(x[:,2]), self.position4(x[:,3])
            x = torch.stack([c1, c2, c3, c4], dim=1)
        elif self.num_contexts == 8:
            c1, c2, c3, c4, c5, c6, c7, c8, c9 = self.position1(x[:,0]), self.position2(x[:,1]), self.position3(x[:,2]), self.position4(x[:,3]), self.position1(x[:,4]), self.position2(x[:,5]), self.position3(x[:,6]), self.position4(x[:,7]), self.position4(x[:,8])
            x = torch.stack([c1, c2, c3, c4, c5, c6, c7, c8, c9], dim=1)
            
        slt_x = self.slotA1(x)
        x = self.croslt1(x, slt_x)

        x_ = x
        xprev = x
        x = getattr(self, "PRB"+str(0))(x)
        x_ = getattr(self, "GIPRB"+str(0))(x_)
        xnow = x+x_
        xnow = self.fuse1(xnow, x, x_)
        x = self.cro1(xnow, xprev)

        slt_x = self.slotA2(x)
        x = self.croslt2(x, slt_x)
    
        x_ = x
        xprev = x
        x = getattr(self, "PRB"+str(1))(x)
        x_ = getattr(self, "GIPRB"+str(1))(x_)
        xnow = x+x_
        xnow = self.fuse2(xnow, x, x_)
        x = self.cro2(xnow, xprev)

        slt_x = self.slotA3(x)
        x = self.croslt3(x, slt_x)

        x_ = x
        xprev = x
        x = getattr(self, "PRB"+str(2))(x)
        x_ = getattr(self, "GIPRB"+str(2))(x_)
        xnow = x+x_
        xnow = self.fuse3(xnow, x, x_)
        x = self.cro3(xnow, xprev)

        x = self.atten(x, self.num_contexts)
        
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