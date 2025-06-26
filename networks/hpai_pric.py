from functools import partial
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from .swin_model import swinB as create_model

from .network_utils import (
    Classifier, 
    ResBlock, 
    ConvNormAct,
    convert_to_rpm_matrix_v9,
    convert_to_rpm_matrix_v6,
    convert_to_rpm_matrix_mnr,
    LinearNormAct
)


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
    def __init__(self, d_model, token_len, nhead=8, dropout=0.1):
        super(PredictionAttention, self).__init__()
        self.kernel = nn.Sequential(nn.Linear(d_model*2, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.m = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.num_heads=nhead
        self.head_dim=d_model//nhead

        # self.position_prompt = SymbolEncoding(9, d_model, token_len)         # 3x3
        # self.rule_prompt = SymbolEncoding(9, d_model, token_len)
        # self.pre_prompt = SymbolEncoding(9, d_model, token_len)
        
        self.position_prompt = SymbolEncoding(6, d_model, token_len)       # Unicode
        self.rule_prompt = SymbolEncoding(6, d_model, token_len)
        self.pre_prompt = SymbolEncoding(6, d_model, token_len)

        # self.position_prompt = SymbolEncoding(4, d_model, 24)
        # self.rule_prompt = SymbolEncoding(4, d_model, 24)
        # self.pre_prompt = SymbolEncoding(4, d_model, 24)

        self.p = nn.Sequential(ConvNormAct(64, 32, 3, 1), nn.Linear(token_len, 6))

        self.token_len = token_len

    def forward(self, x, atten_flag):
        b, c, t, l = x.shape
        prompt = self.position_prompt()
        r = self.rule_prompt().expand(b, -1, -1, -1)
        pre_prompt = self.pre_prompt().expand(b, -1, -1, -1)
        prompt = prompt.expand(b, -1, -1, -1)

        if self.token_len == 18:
            if atten_flag == 1:
                tar = torch.cat([prompt[:, :, :, :12], x[:, :, :, 12:]], dim=3)
                tp = x[:, :, :, 12:]
                con = torch.cat([x[:, :, :, :12], prompt[:, :, :, 12:]], dim=3)
            elif atten_flag == 2:
                tar = torch.cat([prompt[:, :, :, :6], x[:, :, :, 6:12], prompt[:, :, :, 12:]], dim=3)
                tp = x[:, :, :, 6:12]
                con = torch.cat([x[:, :, :, :6], prompt[:, :, :, 6:12], x[:, :, :, 12:]], dim=3)
            else:  # atten_flag == 3
                tar = torch.cat([x[:, :, :, :6], prompt[:, :, :, 6:]], dim=3)
                tp = x[:, :, :, :6]
                con = torch.cat([prompt[:, :, :, :6], x[:, :, :, 6:]], dim=3)
        elif self.token_len == 24:
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
        elif self.token_len == 30:
            if atten_flag == 1:
                tar = torch.cat([prompt[:, :, :, :24], x[:, :, :, 24:]], dim=3)
                tp = x[:, :, :, 24:]
                con = torch.cat([x[:, :, :, :24], prompt[:, :, :, 24:]], dim=3)
            elif atten_flag == 2:
                tar = torch.cat([prompt[:, :, :, :18], x[:, :, :, 18:24], prompt[:, :, :, 24:]], dim=3)
                tp = x[:, :, :, 18:24]
                con = torch.cat([x[:, :, :, :18], prompt[:, :, :, 18:24], x[:, :, :, 24:]], dim=3)
            elif atten_flag == 3:
                tar = torch.cat([prompt[:, :, :, :12], x[:, :, :, 12:18], prompt[:, :, :, 18:]], dim=3)
                tp = x[:, :, :, 12:18]
                con = torch.cat([x[:, :, :, :12], prompt[:, :, :, 12:18], x[:, :, :, 18:]], dim=3)
            elif atten_flag == 4:
                tar = torch.cat([prompt[:, :, :, :6], x[:, :, :, 6:12], prompt[:, :, :, 12:]], dim=3)
                tp = x[:, :, :, 6:12]
                con = torch.cat([x[:, :, :, :6], prompt[:, :, :, 6:12], x[:, :, :, 12:]], dim=3)
            else:  # atten_flag == 5
                tar = torch.cat([x[:, :, :, :6], prompt[:, :, :, 6:]], dim=3)
                tp = x[:, :, :, :6]
                con = torch.cat([prompt[:, :, :, :6], x[:, :, :, 6:]], dim=3)    
        elif self.token_len == 36:
            if atten_flag == 1:
                tar = torch.cat([prompt[:, :, :, :30], x[:, :, :, 30:]], dim=3)
                tp = x[:, :, :, 30:]
                con = torch.cat([x[:, :, :, :30], prompt[:, :, :, 30:]], dim=3)
            elif atten_flag == 2:
                tar = torch.cat([prompt[:, :, :, :24], x[:, :, :, 24:30], prompt[:, :, :, 30:]], dim=3)
                tp = x[:, :, :, 24:30]
                con = torch.cat([x[:, :, :, :24], prompt[:, :, :, 24:30], x[:, :, :, 30:]], dim=3)
            elif atten_flag == 3:
                tar = torch.cat([prompt[:, :, :, :18], x[:, :, :, 18:24], prompt[:, :, :, 24:]], dim=3)
                tp = x[:, :, :, 18:24]
                con = torch.cat([x[:, :, :, :18], prompt[:, :, :, 18:24], x[:, :, :, 24:]], dim=3)
            elif atten_flag == 4:
                tar = torch.cat([prompt[:, :, :, :12], x[:, :, :, 12:18], prompt[:, :, :, 18:]], dim=3)
                tp = x[:, :, :, 12:18]
                con = torch.cat([x[:, :, :, :12], prompt[:, :, :, 12:18], x[:, :, :, 18:]], dim=3)
            elif atten_flag == 5:
                tar = torch.cat([prompt[:, :, :, :6], x[:, :, :, 6:12], prompt[:, :, :, 12:]], dim=3)
                tp = x[:, :, :, 6:12]
                con = torch.cat([x[:, :, :, :6], prompt[:, :, :, 6:12], x[:, :, :, 12:]], dim=3)
            else:  # atten_flag == 6
                tar = torch.cat([x[:, :, :, :6], prompt[:, :, :, 6:]], dim=3)
                tp = x[:, :, :, :6]
                con = torch.cat([prompt[:, :, :, :6], x[:, :, :, 6:]], dim=3)

        
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

        if self.token_len == 18:
            if atten_flag == 1:
                con = torch.cat([con[:, :, :, :12], pre_prompt[:, :, :, 12:]], dim=3)
            elif atten_flag == 2:
                con = torch.cat([con[:, :, :, :6], pre_prompt[:, :, :, 6:12], con[:, :, :, 12:]], dim=3)
            else:  # atten_flag == 3
                con = torch.cat([pre_prompt[:, :, :, :6], con[:, :, :, 6:]], dim=3)
        elif self.token_len == 24:
            if atten_flag == 1:
                con = torch.cat([con[:,:,:,:18], pre_prompt[:,:,:,18:]], dim=3)
            elif atten_flag == 2:
                con = torch.cat([con[:,:,:,:12], pre_prompt[:,:,:,12:18], con[:,:,:,18:]], dim=3)
            elif atten_flag == 3:
                con = torch.cat([con[:,:,:,:6], pre_prompt[:,:,:,6:12], con[:,:,:,12:]], dim=3)
            else:
                con = torch.cat([pre_prompt[:,:,:,:6], con[:,:,:,6:]], dim=3)
        elif self.token_len == 30:
            if atten_flag == 1:
                con = torch.cat([con[:, :, :, :24], pre_prompt[:, :, :, 24:]], dim=3)
            elif atten_flag == 2:
                con = torch.cat([con[:, :, :, :18], pre_prompt[:, :, :, 18:24], con[:, :, :, 24:]], dim=3)
            elif atten_flag == 3:
                con = torch.cat([con[:, :, :, :12], pre_prompt[:, :, :, 12:18], con[:, :, :, 18:]], dim=3)
            elif atten_flag == 4:
                con = torch.cat([con[:, :, :, :6], pre_prompt[:, :, :, 6:12], con[:, :, :, 12:]], dim=3)
            else:  # atten_flag == 5
                con = torch.cat([pre_prompt[:, :, :, :6], con[:, :, :, 6:]], dim=3)
        elif self.token_len == 36:
            if atten_flag == 1:
                con = torch.cat([con[:, :, :, :30], pre_prompt[:, :, :, 30:]], dim=3)
            elif atten_flag == 2:
                con = torch.cat([con[:, :, :, :24], pre_prompt[:, :, :, 24:30], con[:, :, :, 30:]], dim=3)
            elif atten_flag == 3:
                con = torch.cat([con[:, :, :, :18], pre_prompt[:, :, :, 18:24], con[:, :, :, 24:]], dim=3)
            elif atten_flag == 4:
                con = torch.cat([con[:, :, :, :12], pre_prompt[:, :, :, 12:18], con[:, :, :, 18:]], dim=3)
            elif atten_flag == 5:
                con = torch.cat([con[:, :, :, :6], pre_prompt[:, :, :, 6:12], con[:, :, :, 12:]], dim=3)
            else:  # atten_flag == 6
                con = torch.cat([pre_prompt[:, :, :, :6], con[:, :, :, 6:]], dim=3)      

        p = self.p(torch.cat([con,r], dim=1).contiguous())

        if self.token_len == 18:
            if atten_flag == 1:
                x = torch.cat([x[:,:,:,:12], p], dim=-1)
            elif atten_flag == 2:
                x = torch.cat([x[:,:,:,:6], p, x[:,:,:,12:]], dim=-1)
            elif atten_flag == 3:
                x = torch.cat([p, x[:,:,:,6:]], dim=-1)
            else:
                x = torch.cat([x[:,:,:,:6], p, x[:,:,:,6:]], dim=-1)
        elif self.token_len == 24:
            if atten_flag == 1:
                x = torch.cat([x[:,:,:,:18], p], dim=-1)
            elif atten_flag == 2:
                x = torch.cat([x[:,:,:,:12], p, x[:,:,:,18:]], dim=-1)
            elif atten_flag == 3:
                x = torch.cat([x[:,:,:,:6], p, x[:,:,:,12:]], dim=-1)
            else:
                x = torch.cat([p, x[:,:,:,6:]], dim=-1)
        elif self.token_len == 30:
            if atten_flag == 1:
                x = torch.cat([x[:, :, :, :24], p], dim=-1)
            elif atten_flag == 2:
                x = torch.cat([x[:, :, :, :18], p, x[:, :, :, 24:]], dim=-1)
            elif atten_flag == 3:
                x = torch.cat([x[:, :, :, :12], p, x[:, :, :, 18:]], dim=-1)
            elif atten_flag == 4:
                x = torch.cat([x[:, :, :, :6], p, x[:, :, :, 12:]], dim=-1)
            else:
                x = torch.cat([p, x[:, :, :, 6:]], dim=-1)
        elif self.token_len == 36:
            if atten_flag == 1:
                x = torch.cat([x[:, :, :, :30], p], dim=-1)
            elif atten_flag == 2:
                x = torch.cat([x[:, :, :, :24], p, x[:, :, :, 30:]], dim=-1)
            elif atten_flag == 3:
                x = torch.cat([x[:, :, :, :18], p, x[:, :, :, 24:]], dim=-1)
            elif atten_flag == 4:
                x = torch.cat([x[:, :, :, :12], p, x[:, :, :, 18:]], dim=-1)
            elif atten_flag == 5:
                x = torch.cat([x[:, :, :, :6], p, x[:, :, :, 12:]], dim=-1)
            else:
                x = torch.cat([p, x[:, :, :, 6:]], dim=-1)

        return x

class FastGatedPredictionAttentionBlock(nn.Module):

    def __init__(
        self, 
        in_planes, 
        token_len,
        dropout = 0.0, 
        num_heads = 8,
        num_contexts=3,
    ):
        super().__init__()
        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)
        self.lp1 = nn.Linear(in_planes, in_planes*2)
        self.lp2 = nn.Linear(in_planes, in_planes)
        self.m = nn.Linear(in_planes, in_planes)
        self.drop = nn.Dropout(dropout)
        
        self.pre_att = PredictionAttention(in_planes, nhead=num_heads, token_len=token_len)
        self.conv1 = nn.Sequential(ConvNormAct((num_contexts+1), (num_contexts+1)*4, 3, 1, activate=True), ConvNormAct((num_contexts+1)*4, (num_contexts+1), 3, 1, activate=True))
        self.conv2 = nn.Sequential(ConvNormAct(in_planes, in_planes*4, 3, 1, activate=True), ConvNormAct(in_planes*4, in_planes, 3, 1, activate=True))
    
    def forward(self, x, atten_flag):
        shortcut = self.downsample(x)
        x = F.normalize(x.permute(0,2,3,1), dim=-1)
        g, x = self.lp1(x).chunk(2, dim=-1)
        g = self.m(self.conv1(g))
        x = x.permute(0,3,1,2)
        p = self.pre_att(x, atten_flag).permute(0,2,3,1)

        x = self.lp2(F.gelu(g)*p)
        x = self.conv2(x.permute(0,3,1,2).contiguous())
        x = self.drop(x) + shortcut
        return x

class SlowGatedPredictionAttentionBlock(nn.Module):

    def __init__(
        self, 
        in_planes, 
        token_len,
        dropout = 0.0, 
        num_heads = 8,
        num_contexts=3
    ):
        super().__init__()
        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)
        self.lp1 = nn.Linear(in_planes, in_planes*2)
        self.lp2 = nn.Linear(in_planes, in_planes)
        self.m = nn.Linear(in_planes, in_planes)
        self.drop = nn.Dropout(dropout)
        
        self.pre_att = PredictionAttention(in_planes, nhead=num_heads, token_len=token_len)
        self.conv1 = nn.Sequential(ConvNormAct((num_contexts+1), (num_contexts+1)*4, 3, 1, activate=True), ConvNormAct((num_contexts+1)*4, (num_contexts+1), 3, 1, activate=True))
        self.conv2 = nn.Sequential(ConvNormAct(in_planes, in_planes*4, 3, 1, activate=True), ConvNormAct(in_planes*4, in_planes, 3, 1, activate=True))
        self.slow_r = CroAttention(in_planes)
    
    def forward(self, x, atten_flag):
        shortcut = self.downsample(x)
        x = F.normalize(x.permute(0,2,3,1), dim=-1)
        g, x = self.lp1(x).chunk(2, dim=-1)
        g = self.m(self.conv1(g))
        x = x.permute(0,3,1,2)
        p = self.pre_att(x, atten_flag).permute(0,2,3,1)

        x = self.lp2(F.gelu(g)*p)
        x = self.conv2(x.permute(0,3,1,2).contiguous())
        x = self.drop(x) + shortcut
        return x

class FastInduction(nn.Module):

    def __init__(
        self,
        in_planes,
        token_len,
        num_heads = 8,
        num_contexts=3
    ):
        super().__init__()
        self.m = nn.Linear(in_planes, in_planes*2)
        self.fast_p = FastGatedPredictionAttentionBlock(in_planes, num_heads=num_heads, num_contexts=num_contexts, token_len=token_len)
    
    def forward(self, x):
        fx = self.fast_p(x, 1)
        # fx = x
        fx, g = self.m(fx.permute(0, 2, 3, 1).contiguous()).chunk(2, dim=-1)
        fx, g = fx.contiguous(), g.contiguous()
        return fx, g
    
class SlowInduction(nn.Module):
    def __init__(
        self,
        in_planes,
        token_len,
        num_contexts=4,
        num_heads = 8):
        super().__init__()
        self.m = nn.Linear(in_planes, in_planes)
        self.slow_p = SlowGatedPredictionAttentionBlock(in_planes, num_heads=num_heads, num_contexts=num_contexts, token_len=token_len)
        self.g = nn.Sequential(ConvNormAct((num_contexts+1), (num_contexts+1)*4, 3, 1, activate=True), ConvNormAct((num_contexts+1)*4, (num_contexts+1), 3, 1, activate=True), nn.Linear(in_planes, in_planes))
        self.w = SymbolEncoding(1, 1, 1)
        self.slow_r = CroAttention(in_planes)
    
    def forward(self, x, sx, fx, g, i):
        g = self.g(g)
        fx = self.m(F.gelu(g) * fx).permute(0, 3, 1, 2).contiguous()
        # g = torch.ones_like(g)  # Ablation: Set g to all ones
        # fx = self.m(fx).permute(0, 3, 1, 2).contiguous()

        b,t,c,l = fx.shape
        # wm = self.w()
        # wm = wm.expand(b,-1,-1,-1)
        p = self.slow_p(sx, i+2)+fx
        e = F.relu(x) - p
        sx = self.slow_r(e, p)
        return sx


class PredictiveReasoningBlock(nn.Module):

    def __init__(
        self, 
        in_planes, 
        ou_planes,
        steps = 4, 
        token_len = 24,
        dropout = 0.1, 
        num_heads = 8,
        num_contexts=8
    ):

        super().__init__()
        self.m = nn.Linear(25, token_len)
        self.fast_induction = FastInduction(in_planes, num_contexts=num_contexts, token_len=token_len)
        self.steps = steps

        for l in range(steps):
            setattr(
                self, "slow"+str(l),
                SlowInduction(in_planes,num_contexts=num_contexts, token_len=token_len)
            )

    def forward(self, x):
        b, _, _, _ = x.size()
        x = self.m(x)
        fx,g = self.fast_induction(x)
        sx = fx.permute(0, 3, 1, 2).contiguous()

        for i in range(self.steps-1):
            sx = getattr(self, "slow"+str(i))(x,sx,fx,g,i).contiguous()

        # e = F.relu(x - sx)

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

        x = self.drop2(self.m(x.permute(0,1,3,2,4).reshape(b,t,l,c))).permute(0,3,1,2)+shortcut
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
        elif num_contexts == 5:
            c1, c2, c3, c4, c5, c6 = self.position1(x[:,0]), self.position2(x[:,1]), self.position3(x[:,2]), self.position4(x[:,3]), self.position1(x[:,4]), self.position2(x[:,5])
            x = torch.stack([c1, c2, c3, c4, c5, c6], dim=1)    
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
        self.pconv = ConvNormAct(in_planes, in_planes, (num_contexts, 1))
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
        contexts, choices = x[:,:,:t-1], x[:,:,t-1:]
        predictions = self.pconv(contexts)
        prediction_errors = F.relu(choices) - predictions
        
        out = torch.cat((contexts, prediction_errors), dim=2)
        out = self.m1(out.permute(0,2,3,1)*F.gelu(g)).permute(0,3,1,2).contiguous()
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.drop(out)
        out = out + identity
        
        return out


class HPAI_PRIC(nn.Module):

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

        self.swin_model = create_model(in_chans=in_channels)
        self.dim_reducer1 = ConvNormAct(128, 96, 1, 0, activate=False)
        self.dim_reducer2 = ConvNormAct(256, 128, 1, 0, activate=False)        

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
        
        for l in range(5):
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


        swin_output = self.swin_model(x)
        swin_output_l = swin_output[0].view(-1, 64,20,20)
        swin_output_m = self.dim_reducer1(swin_output[1].view(-1, 128,10,10))
        swin_output_h = self.dim_reducer2(swin_output[2].view(-1, 256,5,5))

        hie_vis_out = []
        for l in range(4): 
            x = getattr(self, "res"+str(l))(x)
            ###################################
            hie_vis_out.append(x) # [torch.Size([16, 32, 40, 40]), 
                                  #  torch.Size([16, 64, 20, 20]), 
                                  #  torch.Size([16, 128, 10, 10]), 
                                  #  torch.Size([16, 256, 5, 5])]

        # print("hie_vis_out[0].shape", hie_vis_out[1].shape)
        # print("swin_output_l.shape", swin_output_l.shape)

        hie_vis_mix_out=[hie_vis_out[0], hie_vis_out[1]+swin_output_l, hie_vis_out[2]+swin_output_m, hie_vis_out[3]+swin_output_h]

        x = hie_vis_mix_out[3]

        if self.num_contexts == 8:
            _, c, h, w = x.size()
            x = convert_to_rpm_matrix_v9(x, b, h, w)
        elif self.num_contexts == 3:
            _, c, h, w = x.size()
            x = convert_to_rpm_matrix_mnr(x, b, h, w)
        else:
            _, c, h, w = x.size()
            x = convert_to_rpm_matrix_v6(x, b, h, w)

        # print("3. x.shape", x.shape) # torch.Size([1152, 32, 8, 5])

        x = x.reshape(b * self.ou_channels, self.num_contexts + 1, -1, h*w)
        x = x.permute(0,2,1,3)

        x1 = getattr(self, "PRB"+str(0))(x[:,:32])
        x2 = getattr(self, "PRB"+str(1))(x[:,32:64])
        x3 = getattr(self, "PRB"+str(2))(x[:,64:96])
        x4 = getattr(self, "PRB"+str(3))(x[:,96:])

        x = self.channel_reducer(torch.cat([x1,x2,x3,x4], dim=1))

        # x = self.channel_reducer(x)
        x = getattr(self, "PRB"+str(4))(x)
        
        e = getattr(self, "MAutoRR"+str(0))(x)
        x = self.atten(e, self.num_contexts)

        x = x.reshape(b, self.ou_channels, -1)
        x = F.adaptive_avg_pool1d(x, 1024)

        x = x.reshape(b * self.ou_channels, self.featr_dims)

        out = self.classifier(x)

        return out.view(b, self.ou_channels), out.view(b, self.ou_channels)

    
def hpai_pric_raven(**kwargs):
    return HPAI_PRIC(**kwargs, num_contexts=8)

def hpai_pric_analogy(**kwargs):
    return HPAI_PRIC(**kwargs, num_contexts=5, num_classes=4)