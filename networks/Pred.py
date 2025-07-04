from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_to_rpm_matrix_v9(input, b, h, w):
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


def ConvNormAct(
        inplanes, ouplanes, kernel_size=3, 
        padding=0, stride=1, activate=True
    ):

    block = [nn.Conv2d(inplanes, ouplanes, kernel_size, padding=padding, bias=False, stride=stride)]
    block += [nn.BatchNorm2d(ouplanes)]
    if activate:
        block += [nn.ReLU()]
    
    return nn.Sequential(*block)


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

class PredictiveReasoningBlock(nn.Module):

    def __init__(
        self, 
        in_planes, 
        ou_planes, 
        downsample, 
        stride = 1, 
        dropout = 0.0, 
        num_contexts = 8
    ):

        super().__init__()

        self.stride = stride

        md_planes = ou_planes*4
        self.pconv = ConvNormAct(in_planes, in_planes, (num_contexts, 1))
        self.conv1 = ConvNormAct(in_planes, md_planes, 3, 1)
        self.conv2 = ConvNormAct(md_planes, ou_planes, 3, 1)
        self.drop = nn.Dropout(dropout) if dropout > .0 else nn.Identity()

        self.downsample = downsample

    def forward(self, x):
        
        b, c, t, l = x.size()
        contexts, choices = x[:,:,:t-1], x[:,:,t-1:]
        predictions = self.pconv(contexts)
        prediction_errors = F.relu(choices) - predictions
        
        out = torch.cat((contexts, prediction_errors), dim=2)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.drop(out)
        identity = self.downsample(x)
        out = out + identity
        
        return out

class Pred(nn.Module):

    def __init__(self, num_filters=32, block_drop=0.0, classifier_drop=0.0, 
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
        self.num_extra_stages = num_extra_stages
        self.num_contexts = num_contexts
        self.in_planes = 32
        self.channel_reducer = ConvNormAct(channels[-1], self.in_planes, 1, 0, activate=False)    

        for l in range(num_extra_stages):
            setattr(
                self, "prb"+str(l), 
                self._make_layer(
                    self.in_planes, stride = 1, 
                    block = reasoning_block, 
                    dropout = block_drop
                )
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

    def forward(self, x, train=False):

        if self.in_channels == 1:
            b, n, h, w = x.size()
            x = x.reshape(b*n, 1, h, w)
        elif self.in_channels == 3:
            b, n, _, h, w = x.size()
            x = x.reshape(b*n, 3, h, w)

        for l in range(4):
            x = getattr(self, "res"+str(l))(x)

        x = self.channel_reducer(x)

        _, c, h, w = x.size()

        if self.num_contexts == 8:
            x = convert_to_rpm_matrix_v9(x, b, h, w)
        else:
            x = convert_to_rpm_matrix_v6(x, b, h, w)
        
        x = x.reshape(b * self.ou_channels, self.num_contexts + 1, -1, h * w)
        # e.g. [b,9,c,l] -> [b,c,9,l] (l=h*w)
        x = x.permute(0,2,1,3)

        for l in range(0, self.num_extra_stages): 
            x = getattr(self, "prb"+str(l))(x)
  
        x = x.reshape(b, self.ou_channels, -1)
        x = F.adaptive_avg_pool1d(x, self.featr_dims)    
        x = x.reshape(b * self.ou_channels, self.featr_dims)

        out = self.classifier(x)
        errors = torch.zeros([1,10]).cuda()
        return out.view(b, self.ou_channels), errors
    