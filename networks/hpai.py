from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_model import swinB as create_model

from .network_utils import (
    Classifier, 
    ResBlock, 
    ConvNormAct, 
    convert_to_rpm_matrix_v9,
    convert_to_rpm_matrix_v6
)


class OriginalReasoningBlock(nn.Module):

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
        # self.pconv = ConvNormAct(in_planes, in_planes, (num_contexts, 1))
        self.pconv = ConvNormAct(in_planes, in_planes, (2, 1))
        self.conv1 = ConvNormAct(in_planes, md_planes, 3, 1)
        self.conv2 = ConvNormAct(md_planes, ou_planes, 3, 1)
        self.drop = nn.Dropout(dropout) if dropout > .0 else nn.Identity()

        self.downsample = downsample

        self.se_block1=nn.Sequential(
            nn.Linear(32, 32 // 16, False),
            nn.ReLU(),
            nn.Linear(32 // 16, 32, False),
            nn.Sigmoid()
        )
        self.se_block2=nn.Sequential(
            nn.Linear(32, 32 // 16, False),
            nn.ReLU(),
            nn.Linear(32 // 16, 32, False),
            nn.Sigmoid()
        )
        self.se_block3=nn.Sequential(
            nn.Linear(32, 32 // 16, False),
            nn.ReLU(),
            nn.Linear(32 // 16, 32, False),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        
        b, c, t, l = x.size()
        contexts, choices = x[:,:,:t-1], x[:,:,t-1:]
        first_row_q, first_row_ans = contexts[:,:,0:2], contexts[:,:,2,:].unsqueeze(2)
        second_row_q, second_row_ans = contexts[:,:,3:5], contexts[:,:,5,:].unsqueeze(2)
        third_row_q = contexts[:,:,6:8]
        

        first_row_pred = self.pconv(first_row_q)
        second_row_pred = self.pconv(second_row_q)
        third_row_pred = self.pconv(third_row_q)

        b_1, c_1, _, _ = first_row_pred.size()
        y_1 = self.avg_pool(first_row_pred).view(b_1,c_1)
        y_1 = self.se_block1(y_1).view(b_1,c_1,1,1)

        b_2, c_2, _, _ = second_row_pred.size()
        y_2 = self.avg_pool(second_row_pred).view(b_2,c_2)
        y_2 = self.se_block2(y_2).view(b_2,c_2,1,1)

        b_3, c_3, _, _ = third_row_pred.size()
        y_3 = self.avg_pool(third_row_pred).view(b_3,c_3)
        y_3 = self.se_block3(y_3).view(b_3,c_3,1,1)

        first_pred_err = F.relu(first_row_ans) - first_row_pred
        second_pred_err = F.relu(second_row_ans) - second_row_pred
        third_pred_err = F.relu(choices) - third_row_pred

        out = torch.cat((first_row_q + first_row_q * y_1, first_pred_err, second_row_q + second_row_q * y_2, second_pred_err, third_row_q + third_row_q * y_3, third_pred_err), dim=2)

        # predictions = self.pconv(contexts)
        # prediction_errors = F.relu(choices) - predictions
        
        # out = torch.cat((contexts, prediction_errors), dim=2)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.drop(out)
        identity = self.downsample(x)
        # out = out + identity
        
        return out

class Version1ReasoningBlock(nn.Module):

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
        # self.pconv = ConvNormAct(in_planes, in_planes, (num_contexts, 1))
        self.pconv = ConvNormAct(in_planes, in_planes, (2, 1))
        self.conv1 = ConvNormAct(in_planes, md_planes, 3, 1)
        self.conv2 = ConvNormAct(md_planes, ou_planes, 3, 1)
        self.drop = nn.Dropout(dropout) if dropout > .0 else nn.Identity()

        self.downsample = downsample

    def forward(self, x):
        
        b, c, t, l = x.size()
        contexts, choices = x[:,:,:t-1], x[:,:,t-1:]
        first_row_q, first_row_ans = contexts[:,:,0:2], contexts[:,:,2,:].unsqueeze(2)
        second_row_q, second_row_ans = contexts[:,:,3:5], contexts[:,:,5,:].unsqueeze(2)
        third_row_q = contexts[:,:,6:8]

        first_row_pred = self.pconv(first_row_q)
        second_row_pred = self.pconv(second_row_q)
        third_row_pred = self.pconv(third_row_q)

        first_pred_err = F.relu(first_row_ans) - first_row_pred
        second_pred_err = F.relu(second_row_ans) - second_row_pred
        third_pred_err = F.relu(choices) - third_row_pred

        out = torch.cat((first_row_q, first_pred_err, second_row_q, second_pred_err, third_row_q, third_pred_err), dim=2)

        # predictions = self.pconv(contexts)
        # prediction_errors = F.relu(choices) - predictions
        
        # out = torch.cat((contexts, prediction_errors), dim=2)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.drop(out)
        identity = self.downsample(x)
        out = out + identity
        
        return out

class HPAI(nn.Module):

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

        self.swin_model = create_model(in_chans=in_channels)
        self.dim_reducer1 = ConvNormAct(128, 96, 1, 0, activate=False)
        self.dim_reducer2 = ConvNormAct(256, 128, 1, 0, activate=False)

        # -------------------------------------------------------------------
        # predictive coding 
        self.num_extra_stages = num_extra_stages
        self.num_contexts = num_contexts
        self.in_planes = 32
        self.channel_reducer1 = ConvNormAct(channels[1], self.in_planes, 1, 0, activate=False)    
        self.channel_reducer2 = ConvNormAct(channels[2], self.in_planes, 1, 0, activate=False)    
        self.channel_reducer = ConvNormAct(channels[3], self.in_planes, 1, 0, activate=False)      

        for l in range(num_extra_stages*3):
            setattr(
                self, "prb"+str(l), 
                self._make_layer(
                    self.in_planes, stride = 1, 
                    block = reasoning_block, 
                    dropout = block_drop
                )
            )

        # setattr(self, "prb0", self._make_layer(64, stride = 1, block = reasoning_block, dropout = block_drop))
        # setattr(self, "prb1", self._make_layer(128, stride = 1, block = reasoning_block, dropout = block_drop))
        # setattr(self, "prb2", self._make_layer(256, stride = 1, block = reasoning_block, dropout = block_drop))
        # -------------------------------------------------------------------

        self.featr_dims = 1024

        # self.classifier = Classifier(
        #     self.featr_dims, 1, 
        #     norm_layer = nn.BatchNorm1d, 
        #     dropout = classifier_drop, 
        #     hidreduce = classifier_hidreduce
        # )

        self.classifier = Classifier(
            self.featr_dims*3, 1, 
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
            # downsample = ConvNormAct(planes, 128, 1, 0, activate = False)
        else:
            downsample = nn.Identity()

        if block == PredictiveReasoningBlock or type(block) == partial:
            stage = block(self.in_planes, planes, downsample, stride = stride, 
                          dropout = dropout, num_contexts = self.num_contexts)
            # stage = block(planes, 128, downsample, stride = stride, 
                        #   dropout = dropout, num_contexts = self.num_contexts)
        elif block == ResBlock:
            stage = block(self.in_planes, planes, downsample, stride = stride, dropout = dropout)

        self.in_planes = planes

        return stage

    def forward(self, x):
        # print("0. x.shape", x.shape)

        if self.in_channels == 1:
            b, n, h, w = x.size()
            x = x.reshape(b*n, 1, h, w)
        elif self.in_channels == 3:
            b, n, c, h, w = x.size()
            x = x.reshape(b*n, 3, h, w)
        
        # print("1. x.shape", x.shape)

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
        # hie_vis_mix_out=[hie_vis_out[0], torch.cat((hie_vis_out[1],swin_output_l),dim=1), torch.cat((hie_vis_out[2],swin_output_m),dim=1), torch.cat((hie_vis_out[3],swin_output_h),dim=1)]

        # x = self.channel_reducer(x) # c ==> 32
        x1 = self.channel_reducer1(hie_vis_mix_out[1])
        x2 = self.channel_reducer2(hie_vis_mix_out[2])
        x3 = self.channel_reducer(hie_vis_mix_out[3])
        
        xs = [x1,x2,x3]

        # _, c, h, w = x.size()

        # if self.num_contexts == 8:
        #     x = convert_to_rpm_matrix_v9(x, b, h, w)
        # else:
        #     x = convert_to_rpm_matrix_v6(x, b, h, w)

        hie_reform_out = []
        # for featr in hie_vis_out[1:]:
        for featr in xs:
            _, c, h, w = featr.size()
            hie_reform_out.append(convert_to_rpm_matrix_v9(featr, b, h, w))
                                  # [torch.Size([1, 8, 9, 64, 20, 20]), 
                                  #  torch.Size([1, 8, 9, 128, 10, 10]), 
                                  #  torch.Size([1, 8, 9, 256, 5, 5])]

        # x = x.reshape(b * self.ou_channels, self.num_contexts + 1, -1, h * w)
        # # e.g. [b,9,c,l] -> [b,c,9,l] (l=h*w)
        # x = x.permute(0,2,1,3)

        hie_reshape_out=[]
        for featr in hie_reform_out:
            _, _, _, c, h, w = featr.size()
            reshaped_featr = featr.reshape(b * self.ou_channels, self.num_contexts + 1, -1, h * w)
            reshaped_featr = reshaped_featr.permute(0,2,1,3)
            hie_reshape_out.append(reshaped_featr)
                                  # [torch.Size([8, 64, 9, 400]), 
                                  #  torch.Size([8, 128, 9, 100]), 
                                  #  torch.Size([8, 256, 9, 25])]

        # for l in range(0, self.num_extra_stages): 
        #     x = getattr(self, "prb"+str(l))(x)

        hie_prb_out=[]
        for l in range(len(hie_reshape_out[0:3])): 
            # hie_prb_out.append(getattr(self, "prb"+str(l))(hie_reshape_out[l]))
                                  # [torch.Size([8, 128, 9, 400]), 
                                  #  torch.Size([8, 128, 9, 100]), 
                                  #  torch.Size([8, 128, 9, 25])]
            x = hie_reshape_out[l]
            for m in range(l*self.num_extra_stages, l*self.num_extra_stages+self.num_extra_stages): 
                x = getattr(self, "prb"+str(m))(x)
            hie_prb_out.append(x)
        
        # x = x.reshape(b, self.ou_channels, -1)
        # x = F.adaptive_avg_pool1d(x, self.featr_dims)    
        # x = x.reshape(b * self.ou_channels, self.featr_dims)

        final_featr = []
        for featr in hie_prb_out:
            tmp_featr = featr.reshape(b, self.ou_channels, -1)
            tmp_featr = F.adaptive_avg_pool1d(tmp_featr, self.featr_dims)  
            tmp_featr = tmp_featr.reshape(b * self.ou_channels, self.featr_dims) 
            final_featr.append(tmp_featr)

        final = torch.cat(final_featr, dim=1)


        # out = self.classifier(x)
        out = self.classifier(final)

        # return out.view(b, self.ou_channels)
        return out.view(b, self.ou_channels), _
        # return out.view(b, self.ou_channels), err_total.tolist()
        # return final
    

def hpai_raven(**kwargs):
    return HPAI(**kwargs, num_contexts=8)


def hpai_analogy(**kwargs):
    return HPAI(**kwargs, num_contexts=5, num_classes=4)