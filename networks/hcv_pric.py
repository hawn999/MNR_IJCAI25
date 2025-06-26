from functools import partial
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
from typing import Optional
import torch.utils.checkpoint as checkpoint
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
from .mrnet_pric import MRNet_PRIC


def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.downsample(x) + self.bn2(self.conv2(out)))

        return out


class ResBlock1x1(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock1x1, self).__init__()
        self.conv1 = conv1x1(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(x + self.bn2(self.conv2(out)))

        return out

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, L, C]
        # x = x.view(-1, 80, 80).unsqueeze(1)
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        out = [x]
        for layer in self.layers:
            x, H, W = layer(x, H, W)
            out.append(x)
        x = self.norm(x)  # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        x = torch.flatten(x, 1)
        x = self.head(x)
        return out


def swinB(in_chans, num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    model = SwinTransformer(in_chans=in_chans,
                            patch_size=4,
                            window_size=7,
                            embed_dim=32,
                            depths=(2, 2, 2),
                            num_heads=(4, 8, 16),
                            num_classes=num_classes,
                            drop_path_rate=0.5,
                            **kwargs)
    return model

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

        self.position_prompt = SymbolEncoding(9, d_model, token_len)         # 3x3
        self.rule_prompt = SymbolEncoding(9, d_model, token_len)
        self.pre_prompt = SymbolEncoding(9, d_model, token_len)

        # self.position_prompt = SymbolEncoding(6, d_model, token_len)       # Unicode
        # self.rule_prompt = SymbolEncoding(6, d_model, token_len)
        # self.pre_prompt = SymbolEncoding(6, d_model, token_len)
        
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


class HCV_PRIC(nn.Module):

    def __init__(self, num_filters=48, block_drop=0.0, classifier_drop=0.0, 
                 classifier_hidreduce=1.0, in_channels=1, num_classes=8, 
                 num_extra_stages=1, reasoning_block=PredictiveReasoningBlock,
                 num_contexts=8, do_contrast=False, levels='111'):
        super().__init__()
        # -------------------------------------------------------------------
        # frame encoder 

        self.in_planes = in_channels

        self.do_contrast = do_contrast
        self.levels = levels
        print(f'CONTRAST: {self.do_contrast}')
        print(f'LEVELS: {self.levels}')

        self.high_dim, self.high_dim0 = 32, 16
        self.mid_dim, self.mid_dim0 = 64, 32
        self.low_dim, self.low_dim0 = 128, 64

        self.perception_net_high = nn.Sequential(
            nn.Conv2d(in_channels, self.high_dim0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.high_dim0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.high_dim0, self.high_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.high_dim),
            nn.ReLU(inplace=True))

        self.perception_net_mid = nn.Sequential(
            nn.Conv2d(self.high_dim, self.mid_dim0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_dim0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.mid_dim0, self.mid_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_dim),
            nn.ReLU(inplace=True)
            )

        self.perception_net_low = nn.Sequential(
            nn.Conv2d(self.mid_dim, self.low_dim0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.low_dim0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(self.low_dim0, self.low_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.low_dim),
            nn.ReLU(inplace=True)
            )
        
        self.swinT = swinB(in_chans=in_channels)
        
        # -------------------------------------------------------------------

        

        # -------------------------------------------------------------------
        # predictive coding 
        self.num_contexts = num_contexts
        self.atten = Alignment(32,128)
        self.think_branches = 1
        self.channel_reducer = ConvNormAct(128, 32, 1, 0, activate=False)
        self.channel_reducer1 = ConvNormAct(256, 128, 1, 0, activate=False)

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

        # print("x.shape", x.shape) # torch.Size([1152, 128, 5, 5])

        ### Perception Branch
        input_features_high = self.perception_net_high(x)
        input_features_mid = self.perception_net_mid(input_features_high)
        input_features_low = self.perception_net_low(input_features_mid)
        # ((32*16),64,20,20), ((32*16),128,5,5), ((32*16),256,1,1)
        # print("input_features_high.shape", input_features_high.shape) # torch.Size([1152, 32, 20, 20])
        # print("input_features_mid.shape", input_features_mid.shape) # torch.Size([1152, 64, 10, 10])
        # print("input_features_low.shape", input_features_low.shape) # torch.Size([1152, 128, 5, 5])
        transformer_features = self.swinT(x)

        input_features_high = input_features_high + transformer_features[0].view(-1, 32,20,20) #(32*16,64,20,20)
        input_features_mid = input_features_mid + transformer_features[1].view(-1, 64,10,10) #(32*16,128,10,10)
        input_features_low = input_features_low + transformer_features[2].view(-1, 128,5,5) #(32*16,256,5,5)

        # x = self.channel_reducer1(input_features_low)
        x = input_features_low

        # print("x.shape", x.shape) # torch.Size([1152, 128, 5, 5])

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