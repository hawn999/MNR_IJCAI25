import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

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
        # print("2.x.shape: ", x.shape)
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
                            embed_dim=64,
                            depths=(2, 2, 2),
                            num_heads=(4, 8, 16),
                            num_classes=num_classes,
                            drop_path_rate=0.5,
                            **kwargs)
    return model

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

class element_wise_attention(nn.Module):
    def __init__(self,dim):
        super().__init__()
        B,C,H,W = dim
        q = torch.empty(1,C,H,W, requires_grad=True)
        nn.init.kaiming_normal_(q)
        self.q=nn.Parameter(q)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x1, x2):
        stream1, stream2 = x1, x2
        d1 = self.q * x1
        d1 = d1.unsqueeze(0)

        d2 = self.q * x2
        d2 = d2.unsqueeze(0)

        ds = torch.cat((d1, d2), 0)

        temp = torch.softmax(ds, 0)
        weight1 = temp[0,:]
        weight2 = temp[1,:]

        stream1 = weight1* stream1
        stream2 = weight2* stream2

        result = stream1 + stream2

        return result
        

class HCVARR_RPV(nn.Module):
    def __init__(self, num_filters=48, block_drop=0.0, classifier_drop=0.0, 
                 classifier_hidreduce=1.0, in_channels=1, num_classes=4, 
                 num_extra_stages=1, reasoning_block=None,
                 num_contexts=5, row_col=True, dropout=True, do_contrast=False, levels='111'):
        super(HCVARR_RPV, self).__init__()
        self.do_contrast = do_contrast
        self.levels = levels
        print(f'CONTRAST: {self.do_contrast}')
        print(f'LEVELS: {self.levels}')

        self.in_channels = in_channels

        self.high_dim, self.high_dim0 = 64, 32
        self.mid_dim, self.mid_dim0 = 128, 64
        self.low_dim, self.low_dim0 = 256, 128

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

        self.g_function_high = nn.Sequential(Reshape(shape=(-1, 3 * self.high_dim, 20, 20)),
                                             conv3x3(3 * self.high_dim, self.high_dim),
                                             ResBlock(self.high_dim, self.high_dim),
                                             ResBlock(self.high_dim, self.high_dim))
        self.g_function_mid = nn.Sequential(Reshape(shape=(-1, 3 * self.mid_dim, 10, 10)),
                                            conv3x3(3 * self.mid_dim, self.mid_dim),
                                            ResBlock(self.mid_dim, self.mid_dim),
                                            ResBlock(self.mid_dim, self.mid_dim))
        self.g_function_low = nn.Sequential(Reshape(shape=(-1, 3 * self.low_dim, 5, 5)),
                                            conv1x1(3 * self.low_dim, self.low_dim),
                                            ResBlock1x1(self.low_dim, self.low_dim),
                                            ResBlock1x1(self.low_dim, self.low_dim))

        self.conv_row_high = conv3x3(self.high_dim, self.high_dim)
        self.bn_row_high = nn.BatchNorm2d(self.high_dim)
        self.conv_col_high = conv3x3(self.high_dim, self.high_dim) if row_col else self.conv_row_high
        self.bn_col_high = nn.BatchNorm2d(self.high_dim, ) if row_col else self.bn_row_high

        self.conv_row_mid = conv3x3(self.mid_dim, self.mid_dim)
        self.bn_row_mid = nn.BatchNorm2d(self.mid_dim)
        self.conv_col_mid = conv3x3(self.mid_dim, self.mid_dim) if row_col else self.conv_row_mid
        self.bn_col_mid = nn.BatchNorm2d(self.mid_dim) if row_col else self.bn_row_mid

        self.conv_row_low = conv1x1(self.low_dim, self.low_dim)
        self.bn_row_low = nn.BatchNorm2d(self.low_dim)
        self.conv_col_low = conv1x1(self.low_dim, self.low_dim) if row_col else self.conv_row_low
        self.bn_col_low = nn.BatchNorm2d(self.low_dim) if row_col else self.bn_row_low

        self.bn_row_high.register_parameter('bias', None)
        self.bn_col_high.register_parameter('bias', None)
        self.bn_row_mid.register_parameter('bias', None)
        self.bn_col_mid.register_parameter('bias', None)
        self.bn_row_low.register_parameter('bias', None)
        self.bn_col_low.register_parameter('bias', None)

        self.mlp_dim_high = self.mlp_dim_mid = self.mlp_dim_low = 0
        if self.levels[0] == '1':
            self.mlp_dim_high = 128
            self.res1_high = ResBlock(self.high_dim, 2 * self.high_dim, stride=2,
                                      downsample=nn.Sequential(conv1x1(self.high_dim, 2 * self.high_dim, stride=2),
                                                               nn.BatchNorm2d(2 * self.high_dim)
                                                               )
                                      )

            self.res2_high = ResBlock(2 * self.high_dim, self.mlp_dim_high, stride=2,
                                      downsample=nn.Sequential(conv1x1(2 * self.high_dim, self.mlp_dim_high, stride=2),
                                                               nn.BatchNorm2d(self.mlp_dim_high)
                                                               )
                                      )

        if self.levels[1] == '1':
            self.mlp_dim_mid = 128
            self.res1_mid = ResBlock(self.mid_dim, 2 * self.mid_dim, stride=2,
                                     downsample=nn.Sequential(conv1x1(self.mid_dim, 2 * self.mid_dim, stride=2),
                                                              nn.BatchNorm2d(2 * self.mid_dim)
                                                              )
                                     )

            self.res2_mid = ResBlock(2 * self.mid_dim, self.mlp_dim_mid, stride=2,
                                     downsample=nn.Sequential(conv1x1(2 * self.mid_dim, self.mlp_dim_mid, stride=2),
                                                              nn.BatchNorm2d(self.mlp_dim_mid)
                                                              )
                                     )

        if self.levels[2] == '1':
            self.mlp_dim_low = 128
            self.res1_low = nn.Sequential(conv1x1(self.low_dim, self.mlp_dim_low),
                                          nn.BatchNorm2d(self.mlp_dim_low),
                                          nn.ReLU(inplace=True))
            self.res2_low = ResBlock1x1(self.mlp_dim_low, self.mlp_dim_low)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp_dim = self.mlp_dim_high + self.mlp_dim_mid + self.mlp_dim_low
        self.mlp = nn.Sequential(nn.Linear(self.mlp_dim, 256, bias=False),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.5),
                                 nn.Linear(256, 128, bias=False),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(128, 1, bias=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if m.affine:
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        self.F_func_high = nn.Sequential(nn.Conv2d(64*3,64,1), 
                                    nn.ReLU(inplace=True))
        self.F_func_mid = nn.Sequential(nn.Conv2d(128*3,128,1), 
                                    nn.ReLU(inplace=True))
        self.F_func_low = nn.Sequential(nn.Conv2d(256*3,256,1), 
                                    nn.ReLU(inplace=True))

        self.swinT = swinB(in_chans=in_channels)
        
    def triples(self, input_features):
        N, K, C, H, W = input_features.shape
        K0 = K - 8
        choices_features = input_features[:, 8:, :, :, :].unsqueeze(2)  # N, 8, 64, 20, 20 -> N, 8, 1, 64, 20, 20

        row1_features = input_features[:, 0:3, :, :, :]  # N, 3, 64, 20, 20
        row2_features = input_features[:, 3:6, :, :, :]  # N, 3, 64, 20, 20
        row3_pre = input_features[:, 6:8, :, :, :].unsqueeze(1).expand(N, K0, 2, C, H,
                                                                       W)  # N, 2, 64, 20, 20 -> N, 1, 2, 64, 20, 20 -> N, 8, 2, 64, 20, 20
        row3_features = torch.cat((row3_pre, choices_features), dim=2).view(N * K0, 3, C, H, W)

        col1_features = input_features[:, 0:8:3, :, :, :]  # N, 3, 64, 20, 20
        col2_features = input_features[:, 1:8:3, :, :, :]  # N, 3, 64, 20, 20
        col3_pre = input_features[:, 2:8:3, :, :, :].unsqueeze(1).expand(N, K0, 2, C, H,
                                                                         W)  # N, 2, 64, 20, 20 -> N, 1, 2, 64, 20, 20 -> N, 8, 2, 64, 20, 20
        col3_features = torch.cat((col3_pre, choices_features), dim=2).view(N * K0, 3, C, H, W)

        return row1_features, row2_features, row3_features, col1_features, col2_features, col3_features

    def apply_reduce(self, x1, x2, x3):

        # dist12, dist23, dist31 = (x1 - x2).pow(2), (x2 - x3).pow(2), (x3 - x1).pow(2)
        # x = 1 - (dist12 + dist23 + dist31)

        B, _, C, H, W = x1.shape
        
        fusion = element_wise_attention((B, C, H, W))
        # fusion = torch.nn.DataParallel(fusion)
        fusion = fusion.cuda()
        x_12 = fusion(x1.reshape(-1,C,H,W), x2.reshape(-1,C,H,W))
        x_23 = fusion(x2.reshape(-1,C,H,W), x3.reshape(-1,C,H,W))
        x_13 = fusion(x1.reshape(-1,C,H,W), x3.reshape(-1,C,H,W))

        
        x_cat = torch.cat((x_12,x_23,x_13),dim=1)

        B, C, H, W = x_cat.shape
        x_cat = x_cat.view(-1,C,H,W)
        if C==64*3:
            x_out = self.F_func_high(x_cat)
        if C==128*3:
            x_out = self.F_func_mid(x_cat)
        if C==256*3:
            x_out = self.F_func_low(x_cat)

        x = x_out.view(B//4, 4, C//3, H, W)

        return x

    def reduce(self, row_features, col_features, N, K0):
        _, C, H, W = row_features.shape
        row1 = row_features[:N, :, :, :].unsqueeze(1).expand(N, K0, C, H, W)
        row2 = row_features[N:2 * N, :, :, :].unsqueeze(1).expand(N, K0, C, H, W)
        row3 = row_features[2 * N:, :, :, :].view(N, K0, C, H, W)

        final_row_features = self.apply_reduce(row1, row2, row3)

        col1 = col_features[:N, :, :, :].unsqueeze(1).expand(N, K0, C, H, W)
        col2 = col_features[N:2 * N, :, :, :].unsqueeze(1).expand(N, K0, C, H, W)
        col3 = col_features[2 * N:, :, :, :].view(N, K0, C, H, W)

        final_col_features = self.apply_reduce(col1, col2, col3)

        input_features = final_row_features + final_col_features
        return input_features

    def forward(self, x, train=False):
        # padding1 = torch.ones_like(x[:,0:1]).cuda()
        # padding2 = torch.ones_like(x[:,0:1]).cuda()
        # padding3 = torch.ones_like(x[:,0:1]).cuda()
        # x = torch.cat([padding1, padding2, padding3,x],dim=1)

        if self.in_channels == 1:
            b, n, h, w = x.size()
            x = x.reshape(b*n, 1, h, w)
        elif self.in_channels == 3:
            b, n, _, h, w = x.size()
            x = x.reshape(b*n, 3, h, w)

        _, _, H, W = x.shape
        N = b
        K = n
        K0 = K - 8

        ### Perception Branch
        input_features_high = self.perception_net_high(x)
        input_features_mid = self.perception_net_mid(input_features_high)
        input_features_low = self.perception_net_low(input_features_mid)
        # ((32*16),64,20,20), ((32*16),128,5,5), ((32*16),256,1,1)

        # print("1.x.shape", x.shape)
        transformer_features = self.swinT(x)

        input_features_high = input_features_high + transformer_features[0].view(-1, 64,20,20) #(32*16,64,20,20)
        input_features_mid = input_features_mid + transformer_features[1].view(-1, 128,10,10) #(32*16,128,10,10)
        input_features_low = input_features_low + transformer_features[2].view(-1, 256,5,5) #(32*16,256,5,5)

        # input_features_high = s_x[0].view(-1, 64,20,20) #(32*16,64,20,20)
        # input_features_mid = s_x[1].view(-1, 128,10,10) #(32*16,128,10,10)
        # input_features_low = s_x[2].view(-1, 256,5,5) #(32*16,256,5,5)

        
        ### Relation Module
        # High res
        if self.levels[0] == '1':
            row1_cat_high, row2_cat_high, row3_cat_high, col1_cat_high, col2_cat_high, col3_cat_high = \
                self.triples(input_features_high.view(N, K, self.high_dim, 20, 20))

            row_feats_high = self.g_function_high(torch.cat((row1_cat_high, row2_cat_high, row3_cat_high), dim=0))
            row_feats_high = self.bn_row_high(self.conv_row_high(row_feats_high))
            col_feats_high = self.g_function_high(torch.cat((col1_cat_high, col2_cat_high, col3_cat_high), dim=0))
            col_feats_high = self.bn_col_high(self.conv_col_high(col_feats_high))

            reduced_feats_high = self.reduce(row_feats_high, col_feats_high, N, K0)  # N, 8, 64, 20, 20

        # Mid res
        if self.levels[1] == '1':
            row1_cat_mid, row2_cat_mid, row3_cat_mid, col1_cat_mid, col2_cat_mid, col3_cat_mid = \
                self.triples(input_features_mid.view(N, K, self.mid_dim, 10, 10))

            row_feats_mid = self.g_function_mid(torch.cat((row1_cat_mid, row2_cat_mid, row3_cat_mid), dim=0))
            row_feats_mid = self.bn_row_mid(self.conv_row_mid(row_feats_mid))
            col_feats_mid = self.g_function_mid(torch.cat((col1_cat_mid, col2_cat_mid, col3_cat_mid), dim=0))
            col_feats_mid = self.bn_col_mid(self.conv_col_mid(col_feats_mid))

            reduced_feats_mid = self.reduce(row_feats_mid, col_feats_mid, N, K0)  # N, 8, 256, 5, 5
        # Low res
        if self.levels[2] == '1':
            row1_cat_low, row2_cat_low, row3_cat_low, col1_cat_low, col2_cat_low, col3_cat_low = \
                self.triples(input_features_low.view(N, K, self.low_dim, 5, 5))

            row_feats_low = self.g_function_low(torch.cat((row1_cat_low, row2_cat_low, row3_cat_low), dim=0))
            row_feats_low = self.bn_row_low(self.conv_row_low(row_feats_low))
            col_feats_low = self.g_function_low(torch.cat((col1_cat_low, col2_cat_low, col3_cat_low), dim=0))
            col_feats_low = self.bn_col_low(self.conv_col_low(col_feats_low))

            reduced_feats_low = self.reduce(row_feats_low, col_feats_low, N, K0)  # N, 8, 256, 5, 5

        ### Combine
        self.final_high = self.final_mid = self.final_low = None
        final = []
        # High
        if self.levels[0] == '1':
            res1_in_high = reduced_feats_high
            if self.do_contrast:
                res1_in_high = res1_in_high - res1_in_high.mean(dim=1).unsqueeze(1)
            res1_out_high = self.res1_high(res1_in_high.view(N * K0, self.high_dim, 20, 20))
            res2_in_high = res1_out_high.view(N, K0, 2 * self.high_dim, 10, 10)
            if self.do_contrast:
                res2_in_high = res2_in_high - res2_in_high.mean(dim=1).unsqueeze(1)
            out_high = self.res2_high(res2_in_high.view(N * K0, 2 * self.high_dim, 10, 10))
            final_high = self.avgpool(out_high)
            final_high = final_high.view(-1, self.mlp_dim_high)
            final.append(final_high)
            self.final_high = final_high

        # Mid
        if self.levels[1] == '1':
            res1_in_mid = reduced_feats_mid
            if self.do_contrast:
                res1_in_mid = res1_in_mid - res1_in_mid.mean(dim=1).unsqueeze(1)
            res1_out_mid = self.res1_mid(res1_in_mid.view(N * K0, self.mid_dim, 10, 10))
            res2_in_mid = res1_out_mid.view(N, K0, 2 * self.mid_dim, 5, 5)
            if self.do_contrast:
                res2_in_mid = res2_in_mid - res2_in_mid.mean(dim=1).unsqueeze(1)
            out_mid = self.res2_mid(res2_in_mid.view(N * K0, 2 * self.mid_dim, 5, 5))
            final_mid = self.avgpool(out_mid)
            final_mid = final_mid.view(-1, self.mlp_dim_mid)
            final.append(final_mid)
            self.final_mid = final_mid

        # Low
        if self.levels[2] == '1':
            res1_in_low = reduced_feats_low
            if self.do_contrast:
                res1_in_low = res1_in_low - res1_in_low.mean(dim=1).unsqueeze(1)
            res1_out_low = self.res1_low(res1_in_low.view(N * K0, self.low_dim, 5, 5))
            res2_in_low = res1_out_low.view(N, K0, self.mlp_dim_low, 5, 5)
            if self.do_contrast:
                res2_in_low = res2_in_low - res2_in_low.mean(dim=1).unsqueeze(1)
            out_low = self.res2_low(res2_in_low.view(N * K0, self.mlp_dim_low, 5, 5))
            final_low = self.avgpool(out_low)
            final_low = final_low.view(-1, self.mlp_dim_low)
            final.append(final_low)
            self.final_low = final_low

        final = torch.cat(final, dim=1) # 32,8,128

        # MLP
        out = self.mlp(final)
        errors = torch.zeros([1,10]).cuda()
        return out.view(-1, K0), errors

def hcvarr_rpv(**kwargs):
    return HCVARR_RPV(**kwargs, num_contexts=5, num_classes=4)