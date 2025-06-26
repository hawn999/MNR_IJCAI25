from typing import Optional, Callable

import torch
# from einops.layers.torch import Rearrange # Not used yet
from torch import nn
import math
from functools import partial

import torch.nn.functional as F
import random
from torchvision.models.resnet import BasicBlock, Bottleneck
# HELPER CLASSES ---------------------------------------------------------------------------------------------------- #

class conv_module(nn.Module):
    def __init__(self):
        super(conv_module, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()
        # self.fc = nn.Linear(32*4*4, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNet(nn.Module):
    def __init__(self,
            block,
            repeats,
            inplanes=64,
            channels=[64, 128, 256, 512],
            input_dim=3,
            zero_init_residual=False,
            norm_layer=None,
            enable_maxpool=True):
        super(ResNet, self).__init__()

        assert repeats is not None
        nr_layers = len(repeats)
        assert len(channels) == nr_layers

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(input_dim, self.inplanes,
            kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = None
        if enable_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layers = []
        for i in range(nr_layers):
            stride = 2 if i > 0 else 1
            layers.append(self._make_layer(
                block, channels[i], repeats[i], stride=stride))
        self.layers = nn.Sequential(*layers)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)

        x = self.layers(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNetWrapper(nn.Module):
    def __init__(self, repeats=[1, 2, 2, 2], inplanes=8, channels=[1, 8, 16, 32],
            input_dim=1, image_size=(80,80)):
        super().__init__()
        self.resnet = ResNet(
            block=BasicBlock,
            repeats=repeats,
            inplanes=inplanes,
            channels=channels,
            input_dim=input_dim,
            enable_maxpool=False)
        h, w = image_size
        for i in range(len(repeats)):
            h = h // 2
            w = w // 2
        self.output_dim = channels[-1]
        self.output_image_size = (h, w)

    def forward(self, x):
        # input.shape: 1, h, w
        # after conv1: inplanes, h//2, w//2
        # after layer1: inplanes, h//2, w//2
        # after layer2: inplanes, h//4, w//4
        # after layer3: inplanes, h//8, w//8
        # after layer2: inplanes, h//16, w//16
        return self.resnet(x)

class ConvBnRelu(nn.Module):
    def __init__(self, num_input_channels: int, num_output_channels: int, **kwargs):
        super(ConvBnRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_input_channels, num_output_channels, **kwargs),
            nn.BatchNorm2d(num_output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)

def FeedForward(            # ! is this okay to just import like this?
    dim: int,
    expansion_factor: int = 4,
    dropout: float = 0.0,
    dense: Callable[..., nn.Module] = nn.Linear,
    activation: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
    output_dim: Optional[int] = None,
):
    output_dim = output_dim if output_dim else dim
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        activation(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, output_dim),
        nn.Dropout(dropout),
    )

def convert_to_rpm_matrix_v3(input, b, h, w):
    # b: batch
    # h: height
    # w: width
    output = input.reshape(b, 16, -1, h, w)
    output = torch.stack(
        [torch.cat((output[:,:8], output[:,i].unsqueeze(1)), dim=1) for i in range(8, 16)], 
        dim=1
    )

    return output

def convert_to_rpm_matrix_v6(input, b, h, w):
    # b: batch
    # h: height
    # w: width
    output = input.reshape(b, 9, -1, h, w)
    output = torch.stack(
        [torch.cat((output[:,:5], output[:,i].unsqueeze(1)), dim=1) for i in range(5, 9)], 
        dim=1
    )

    return output

class ResBlock(nn.Module):

    def __init__(self, inplanes, ouplanes, downsample, stride=1, dropout=0.0):
        super().__init__()

        mdplanes = ouplanes

        self.conv1 = ConvNormAct(inplanes, mdplanes, 3, 1, stride=stride)
        self.conv2 = ConvNormAct(mdplanes, mdplanes, 3, 1)
        self.conv3 = ConvNormAct(mdplanes, ouplanes, 3, 1)

        self.downsample = downsample
        self.drop = nn.Dropout(p=dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.drop(out)
        identity = self.downsample(x)
        # print("out.shape: ", out.shape)
        # print("identity.shape: ", identity.shape)
            # torch.Size([704, 32, 40, 40])
            # out.shape:  torch.Size([704, 32, 40, 40])
            # identity.shape:  torch.Size([704, 32, 40, 40])
            # out.shape:  torch.Size([704, 64, 20, 20])
            # identity.shape:  torch.Size([704, 64, 20, 20])
            # out.shape:  torch.Size([704, 96, 10, 10])
            # identity.shape:  torch.Size([704, 96, 10, 10])
            # out.shape:  torch.Size([704, 128, 5, 5])
            # identity.shape:  torch.Size([704, 128, 5, 5])
        out = out + identity
        return out

class Classifier(nn.Module):

    def __init__(self, inplanes, ouplanes, norm_layer=nn.BatchNorm2d, dropout=0.0, hidreduce=1.0):
        super().__init__()

        midplanes = inplanes // hidreduce

        self.mlp = nn.Sequential(
            nn.Linear(inplanes, midplanes, bias=False),
            norm_layer(midplanes),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(midplanes, ouplanes)
        )

    def forward(self, x):
        return self.mlp(x)

def ConvNormAct(
        inplanes, ouplanes, kernel_size=3, 
        padding=0, stride=1, activate=True
    ):

    block = [nn.Conv2d(inplanes, ouplanes, kernel_size, padding=padding, bias=False, stride=stride)]
    block += [nn.BatchNorm2d(ouplanes)]
    if activate:
        block += [nn.ReLU()]
    
    return nn.Sequential(*block)


class CalError(nn.Module):
    def __init__(self,in_planes):
        super().__init__()
        self.pl = ConvNormAct(in_planes, in_planes, (2, 1))
    
    def forward(self,c1,c2,c3):
        e = F.gelu(c3) - self.pl(torch.cat([c1, c2], dim=2))
        return e

class PredictiveReasoningBlock(nn.Module):

    def __init__(
        self, 
        in_planes=25,
        stride = 1, 
        dropout = 0.1, 
        num_contexts = 8
    ):

        super().__init__()
        mi_planes = in_planes*4
        self.CalE1 = CalError(in_planes)
        self.CalE2 = CalError(in_planes)
        self.CalE3 = CalError(in_planes)

        self.mlp = LinearBNReLU(25, 25)

        self.m1 = LinearBNReLU(25*3, 1)
        self.m2 = LinearBNReLU(25*3, 1)
        self.m3 = LinearBNReLU(25*3, 1)

        self.conv1 = ConvNormAct(in_planes, mi_planes, 3, 1)
        self.conv2 = ConvNormAct(mi_planes, in_planes, 3, 1)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(25)

        self.num_contexts = num_contexts

    def forward(self, x):
        shortcut = x
        b, c, t, l = x.size()
        x = self.norm(x)
        contexts, answer = x[:, :, :t-1], x[:, :, t-1:]

        if self.num_contexts == 8:
            c1,c2,c3 = x[:,:,0].unsqueeze(2), x[:,:,1].unsqueeze(2), x[:,:,2].unsqueeze(2)
            c4,c5,c6 = x[:,:,3].unsqueeze(2), x[:,:,4].unsqueeze(2), x[:,:,5].unsqueeze(2)
            c7,c8,c9 = x[:,:,6].unsqueeze(2), x[:,:,7].unsqueeze(2), x[:,:,8].unsqueeze(2)

            e1 = self.CalE1(c1,c2,c3)
            e2 = self.CalE1(c4,c5,c6)
            e3 = self.CalE1(c7,c8,c9)
            
            e1_ = self.CalE2(c1,c3,c2)
            e2_ = self.CalE2(c4,c6,c5)
            e3_ = self.CalE2(c7,c9,c8)

            _e1 = self.CalE3(c3,c2,c1)
            _e2 = self.CalE3(c6,c5,c4)
            _e3 = self.CalE3(c9,c8,c7)

            p1 = torch.cat((c1,c2,e1,c4,c5,e2,c7,c8,e3), dim=2)
            p2 = torch.cat((c1,e1_,c3,c4,e2_,c6,c7,e3_,c9), dim=2)
            p3 = torch.cat((_e1,c2,c3,e2_,c5,c6,e3_,c8,c9), dim=2)
        else:
            c1,c2,c3 = x[:,:,0].unsqueeze(2), x[:,:,1].unsqueeze(2), x[:,:,2].unsqueeze(2)
            c4,c5,c6 = x[:,:,3].unsqueeze(2), x[:,:,4].unsqueeze(2), x[:,:,5].unsqueeze(2)

            e1 = self.CalE1(c1,c2,c3)
            e2 = self.CalE1(c4,c5,c6)

            e1_ = self.CalE2(c1,c3,c2)
            e2_ = self.CalE2(c4,c6,c5)

            _e1 = self.CalE3(c3,c2,c1)
            _e2 = self.CalE3(c6,c5,c4)

            p1 = torch.cat((c1,c2,e1,c4,c5,e2), dim=2)
            p2 = torch.cat((c1,e1_,c3,c4,e2_,c6), dim=2)
            p3 = torch.cat((_e1,c2,c3,e2_,c5,c6), dim=2)


        p1_ = F.sigmoid(self.m1(torch.cat([p1, p2, p3], dim=-1)))*p1
        p2_ = F.sigmoid(self.m2(torch.cat([p1, p2, p3], dim=-1)))*p2
        p3_ = F.sigmoid(self.m3(torch.cat([p1, p2, p3], dim=-1)))*p3

        p = self.mlp(p1_+p2_+p3_)

        p = self.conv1(p)
        p = self.conv2(p)
        p = self.drop(p)+shortcut
        
        return p

class SimilarityReasoningBlock(nn.Module):

    def __init__(
        self, 
        in_planes=25,
        stride = 1, 
        dropout = 0.1, 
        num_contexts = 8
    ):

        super().__init__()
        self.norm = nn.LayerNorm(in_planes)
        self.lp = LinearBNReLU(in_planes, 48)
        in_planes=48
        mi_planes = in_planes*4

        self.f2 = nn.Sequential(LinearBNReLU(48, mi_planes), LinearBNReLU(mi_planes, 25))
        self.f4 = nn.Sequential(ConvNormAct(32, 32*4, 3, 1), ConvNormAct(32*4, 32, 3, 1))

        self.m1 = LinearBNReLU(in_planes*6, 1)
        self.m2 = LinearBNReLU(in_planes*6, 1)
        self.m3 = LinearBNReLU(in_planes*6, 1)
        self.m4 = LinearBNReLU(in_planes*6, 1)
        self.m5 = LinearBNReLU(in_planes*6, 1)
        self.m6 = LinearBNReLU(in_planes*6, 1)


        self.p1 = LinearBNReLU(in_planes//8*7, in_planes//8)
        self.p2 = LinearBNReLU(in_planes//4*3, in_planes//4)


        self.drop = nn.Dropout(dropout)
        self.in_planes = in_planes

    def forward(self, x):
        shortcut = x
        b, c, t, l = x.size()
        x = self.norm(x)
        x = self.lp(x)

        mask_window2 = self.in_planes//4

        start2 = int(random.random()*mask_window2*3)

        context1, answer1 = torch.cat([x[:,:,:,:start2], x[:,:,:,start2+mask_window2:]], dim=-1), x[:,:,:,start2:start2+mask_window2]
        c1, c2, c3 = context1[:,:,:,:mask_window2], context1[:,:,:,mask_window2:2*mask_window2], context1[:,:,:,2*mask_window2:]
        context2 = torch.cat([c1, c3, c2], dim=-1)
        context3 = torch.cat([c2, c1, c3], dim=-1)
        context4 = torch.cat([c2, c3, c1], dim=-1)
        context5 = torch.cat([c3, c1, c2], dim=-1)
        context6 = torch.cat([c3, c2, c1], dim=-1)

        e1 = F.gelu(answer1) - self.p2(context1)
        e2 = F.gelu(answer1) - self.p2(context2)
        e3 = F.gelu(answer1) - self.p2(context3)
        e4 = F.gelu(answer1) - self.p2(context4)
        e5 = F.gelu(answer1) - self.p2(context5)
        e6 = F.gelu(answer1) - self.p2(context6)


        p1 = torch.cat([context1[:,:,:,:start2], e1, context1[:,:,:,start2:]], dim=-1)
        p2 = torch.cat([context2[:,:,:,:start2], e2, context2[:,:,:,start2:]], dim=-1)
        p3 = torch.cat([context3[:,:,:,:start2], e3, context3[:,:,:,start2:]], dim=-1)
        p4 = torch.cat([context4[:,:,:,:start2], e4, context4[:,:,:,start2:]], dim=-1)
        p5 = torch.cat([context5[:,:,:,:start2], e5, context5[:,:,:,start2:]], dim=-1)
        p6 = torch.cat([context6[:,:,:,:start2], e6, context6[:,:,:,start2:]], dim=-1)

        p1_ = F.sigmoid(self.m1(torch.cat([p1, p2, p3, p4, p5, p6], dim=-1)))*p1
        p2_ = F.sigmoid(self.m2(torch.cat([p1, p2, p3, p4, p5, p6], dim=-1)))*p2
        p3_ = F.sigmoid(self.m3(torch.cat([p1, p2, p3, p4, p5, p6], dim=-1)))*p3
        p4_ = F.sigmoid(self.m4(torch.cat([p1, p2, p3, p4, p5, p6], dim=-1)))*p4
        p5_ = F.sigmoid(self.m5(torch.cat([p1, p2, p3, p4, p5, p6], dim=-1)))*p5
        p6_ = F.sigmoid(self.m6(torch.cat([p1, p2, p3, p4, p5, p6], dim=-1)))*p6
        p = self.f2(p1_+p2_+p3_+p4_+p5_+p6_)
        # p = self.f2(p1+p2+p3+p4+p5+p6)
        # p = self.f2(p1)

        p = self.f4(p)

        p = self.drop(p)+shortcut
        
        return p

class LinearBNReLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(LinearBNReLU, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        shape = x.shape
        x = x.flatten(0, -2)
        x = self.bn(x)
        x = x.view(shape)
        x = self.relu(x)
        return x

class perm(nn.Module):
    def __init__(self):
        super(perm, self).__init__()
    def forward(self, x):
        return x.permute(0, 2, 1)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dim):
        super(ConvBlock, self).__init__()
        self.conv  = getattr(nn, 'Conv{}d'.format(dim))(in_ch, out_ch, 7, stride=dim, padding=7//2)
        self.bnrm  = getattr(nn, 'BatchNorm{}d'.format(dim))(out_ch)
        self.drop  = nn.Sequential(perm(), nn.Dropout2d(.1), perm()) if dim==1 else nn.Dropout2d(.1)
        self.block = nn.Sequential(self.conv, nn.ELU(), self.bnrm, self.drop)
    def forward(self, x):
        return self.block(x)

class ResBlock1(nn.Module):
    def __init__(self, in_ch, hd_ch, out_ch, dim):
        super(ResBlock1, self).__init__()
        self.dim  = dim
        self.conv = nn.Sequential(ConvBlock(in_ch, hd_ch, dim), ConvBlock(hd_ch, out_ch, dim))
        self.down = nn.Sequential(nn.MaxPool2d(3, 2, 1), nn.MaxPool2d(3, 2, 1))
        self.skip = getattr(nn, 'Conv{}d'.format(dim))(in_ch, out_ch, 1, bias=False)
    def forward(self, x):
        return self.conv(x) + self.skip(x if self.dim==1 else self.down(x))


class DARR(nn.Module):

    def __init__(self, num_filters=32, block_drop=0.1, classifier_drop=0.1, 
                 classifier_hidreduce=4, in_channels=1, num_classes=8, 
                 num_extra_stages=3, reasoning_block=PredictiveReasoningBlock,
                 num_contexts=3):

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
        
        self.in_planes = in_channels
        for l in range(len(strides)):
            setattr(
                self, "res1"+str(l), 
                self._make_layer(
                    channels[l], stride=strides[l], 
                    block=ResBlock, dropout=block_drop,
                )
            )
        # -------------------------------------------------------------------

        

        # -------------------------------------------------------------------
        # predictive coding 
        self.num_extra_stages = num_extra_stages
        self.num_contexts = num_contexts
        self.in_planes = 32
        self.L=3
        self.channel_reducer = ConvNormAct(128, self.in_planes, 1, 0, activate=False)    
        self.channel_reducer1 = ConvNormAct(128, self.in_planes, 1, 0, activate=False)
        # self.obj_enc = nn.Sequential(ResBlock1(   1, 64, 64, 2), ResBlock1(64, 64, 16, 2))    

        for l in range(self.L):
            setattr(
                self, "prb"+str(l), 
                PredictiveReasoningBlock(in_planes=self.in_planes, num_contexts=self.num_contexts)
            )
        
        for l in range(num_extra_stages):
            setattr(
                self, "srb"+str(l), 
                SimilarityReasoningBlock(in_planes=25)
            )
        
        
        for l in range(1):
            setattr(
                self, "mlp1"+str(l), 
                LinearBNReLU(6400, 3200)
            )

        self.featr_dims = 1024

        self.classifier = Classifier(
            self.featr_dims, 1, 
            norm_layer = nn.BatchNorm1d, 
            dropout = classifier_drop, 
            hidreduce = classifier_hidreduce
        )


        
        self.in_channels = in_channels
        self.ou_channels = num_classes

        self.split_indices = [25, 25]


    def _make_layer(self, planes, stride, dropout, block, downsample=True):
        if downsample and block == ResBlock:
            downsample = nn.Sequential(
                # nn.AvgPool2d(kernel_size = 2, stride = stride) if stride != 1 else nn.Identity(),   # NOTE: For 80 x 80
                nn.AvgPool2d(kernel_size = 1, stride = stride) if stride != 1 else nn.Identity(),   # NOTE: For 120 x 120
                ConvNormAct(self.in_planes, planes, 1, 0, activate = False, stride=1),
            )
        elif downsample and (block == PredictiveReasoningBlock or type(block) == partial):
            downsample = ConvNormAct(self.in_planes, planes, 1, 0, activate = False)
        else:
            downsample = nn.Identity()

        if block == PredictiveReasoningBlock or type(block) == partial:
            stage = block(self.in_planes, planes, downsample, stride = stride, 
                          dropout = dropout, num_contexts = self.num_contexts)
        elif block == ResBlock:
            stage = block(self.in_planes, planes, downsample, stride = stride, dropout = dropout)

        self.in_planes = planes

        return stage

    def randomly_rotate_tensor(self, tensor, num_to_rotate):
        """
        Randomly rotate a specified number of elements in the batch dimension of the tensor.
        
        tensor (torch.Tensor): The input tensor of shape [b, n, h, w].
        num_to_rotate (int): The number of elements in the batch dimension to rotate.
        
        Returns:
        torch.Tensor: The tensor with randomly selected elements rotated by 90, 180, or 270 degrees.
        """
        b, n, h, w = tensor.shape
        rotated_tensor = tensor.clone()  # Make a copy of the tensor to avoid in-place modification

        # Randomly select indices in the batch dimension to rotate
        indices = random.sample(range(b), num_to_rotate)
        # Randomly select corresponding rotation angles
        angles = random.choices([90, 180, 270], k=num_to_rotate)

        for idx, angle in zip(indices, angles):
            if angle == 90:
                rotated_tensor[idx] = torch.rot90(tensor[idx], k=1, dims=[1, 2])
            elif angle == 180:
                rotated_tensor[idx] = torch.rot90(tensor[idx], k=2, dims=[1, 2])
            elif angle == 270:
                rotated_tensor[idx] = torch.rot90(tensor[idx], k=3, dims=[1, 2])

        return rotated_tensor

    def forward(self, x, train=False):

        if self.in_channels == 1:
            b, n, h, w = x.size()

            if train:
                # x1 = self.randomly_rotate_tensor(x,b//2)
                # x2 = self.randomly_rotate_tensor(x,b//2)
                # x1 = x1.reshape(b*n, 1, h, w)
                # x2 = x2.reshape(b*n, 1, h, w)
                x1 = x.reshape(b*n, 1, h, w)
                x2 = x1
            else:
                x1 = x.reshape(b*n, 1, h, w)
                x2 = x1
        elif self.in_channels == 3:
            b, n, _, h, w = x.size()
            x = x.reshape(b*n, 3, h, w)
            x1,x2 = x,x
        
        # x1,x2 = x,x
        

        for l in range(4):
            x1 = getattr(self, "res"+str(l))(x1)
        for l in range(4):
            # if l == 0:
            #     x2 = getattr(self, "res"+str(l))(x2)
            # else:
            x2= getattr(self, "res1"+str(l))(x2)

        # x1 = self.wre_en(x1)
        # x2 = self.wre_en(x2)

        x1 = self.channel_reducer(x1)
        x2 = self.channel_reducer1(x2)


        _, c, h, w = x1.size()

        # x1 = convert_to_rpm_matrix_v3(x1, b, h, w)
        # x2 = convert_to_rpm_matrix_v3(x2, b, h, w)

        if self.num_contexts == 5:
            x1 = convert_to_rpm_matrix_v6(x1, b, h, w)
            x2 = convert_to_rpm_matrix_v6(x2, b, h, w)
        else:
            x1 = convert_to_rpm_matrix_v3(x1, b, h, w)
            x2 = convert_to_rpm_matrix_v3(x2, b, h, w)

        # print("x1.shape: ", x1.shape)
        # print("x2.shape: ", x2.shape)
        
        x1 = x1.reshape(b * self.ou_channels, self.num_contexts + 1, -1, h * w)
        x2 = x2.reshape(b * self.ou_channels, self.num_contexts + 1, -1, h * w)
        # e.g. [b,9,c,l] -> [b,c,9,l] (l=h*w)
        x1 = x1.permute(0,2,1,3)
        x2 = x2.permute(0,2,1,3)

        for l in range(0, self.num_extra_stages): 
            x1 = getattr(self, "srb"+str(l))(x1)
        
        for l in range(0, self.L):
            x2 = getattr(self, "prb"+str(l))(x2)
        # x = getattr(self, "mlp1"+str(0))(torch.cat((x1,x2), dim=-1))
        # shortcut = x
        # x = getattr(self, "conv"+str(0))(x)+shortcut
        # x = x1
        x1 = x1.reshape(b, self.ou_channels, -1)
        x2 = x2.reshape(b, self.ou_channels, -1)
        x1 = F.adaptive_avg_pool1d(x1, 3200)
        x2 = F.adaptive_avg_pool1d(x2, 3200)
        x = getattr(self, "mlp1"+str(0))(torch.cat([x1, x2], dim=-1))
        x = F.adaptive_avg_pool1d(x, self.featr_dims)    
        x = x.reshape(b * self.ou_channels, self.featr_dims)

        out = self.classifier(x)
        errors = torch.zeros([1,10]).cuda()

        return out.view(b, self.ou_channels), errors
    

def darr_raven(**kwargs):
    return DARR(**kwargs, num_contexts=8)

def darr_analogy(**kwargs):
    return DARR(**kwargs, num_contexts=5, num_classes=4)
