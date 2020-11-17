import os
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

class _DCR_block(nn.Module):
    def __init__(self, channel_in):
        super(_DCR_block, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(num_features=int(channel_in/2.))
        self.relu1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in*3/2.), out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(num_features=int(channel_in/2.))
        self.relu2 = nn.ReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in*2, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(num_features=channel_in)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu1(self.bn_1(self.conv_1(x)))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.bn_2(self.conv_2(conc)))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.bn_3(self.conv_3(conc)))
        out = torch.add(out, residual)
        return out

class _DCR_block_Encod(nn.Module):
    def __init__(self, channel_in):
        super(_DCR_block_Encod, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(num_features=int(channel_in/2.))
        self.relu1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=int(channel_in*3/2.), out_channels=int(channel_in/2.), kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(num_features=int(channel_in/2.))
        self.relu2 = nn.ReLU()
        self.conv_3 = nn.Conv2d(in_channels=channel_in*2, out_channels=channel_in, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(num_features=channel_in)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu1(self.bn_1(self.conv_1(x)))
        conc = torch.cat([x, out], 1)
        out = self.relu2(self.bn_2(self.conv_2(conc)))
        conc = torch.cat([conc, out], 1)
        out = self.relu3(self.bn_3(self.conv_3(conc)))
        #channel수가 DHDN은 늘어나지않아서 _DCR_Encod residual add 부분을 concat으로 변경
        #out = torch.add(out, residual)
        out = torch.cat([out, residual],1)
        return out
    
    
class FusionNet(nn.Module):
    def __init__(self, num_filters=32, pretrained=False):
        super(FusionNet, self).__init__()
                
        self.in_dim = 3
        self.out_dim = 64
        self.final_out_dim = 1
        
       
 
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        #act_fn =  nn.ReLU()
        #act_fn_2 = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        print("\n------Initiating FusionNet------\n")

        # encoder

        #self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = maxpool()
        #self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool()
        #self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()
        #self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = maxpool()
        
        ##DHDN encoding PART
        self.conv_i = nn.Conv2d(in_channels=3, out_channels=self.out_dim//2, kernel_size=1, stride=1, padding=0)
        #self.relu1 = nn.PReLU()
        self.relu1 = nn.ReLU()
        self.down_1 = _DCR_block_Encod(self.out_dim//2)
        self.down_2 = _DCR_block_Encod(self.out_dim)
        self.down_3 = _DCR_block_Encod(self.out_dim*2)
        self.down_4 = _DCR_block_Encod(self.out_dim*4)
 
        # bridge

        #self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)
        self.bridge = self.make_layer(_DCR_block_Encod, self.out_dim*8)
        

        # decoder

        self.deconv_1 = conv_trans_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)
        #self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = conv_trans_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        #self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = conv_trans_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        #self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = conv_trans_block(self.out_dim * 2, self.out_dim, act_fn_2)
        #self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2)
        self.up_1 =  _DCR_block(self.out_dim*8)
        self.up_2 =  _DCR_block(self.out_dim*4)
        self.up_3 =  _DCR_block(self.out_dim*2)
        self.up_4 =  _DCR_block(self.out_dim)

        # output

        self.out = nn.Conv2d(self.out_dim,self.final_out_dim, kernel_size=3, stride=1, padding=1)
        #self.out = nn.Conv2d(self.out_dim,self.out_dim//2, kernel_size=3, stride=1, padding=1)
        
        self.out_2 = nn.Tanh()
        self.out_a = nn.ReLU()
        
        #self.out_f = nn.Conv2d(self.out_dim//2,self.final_out_dim, kernel_size=1, stride=1, padding=0)
        
        # initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)                
    
    def forward(self,input):
        
        
        out = self.conv_i(input)
        out = self.relu1(out)
        
        down_1 = self.down_1(out)
        #down_1 = self.down_1(input)
        
        pool_1 = self.pool_1(down_1)
        
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4)/2
        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3)/2
        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2)/2
        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1)/2
        up_4 = self.up_4(skip_4)

        out = self.out(up_4)
        out = self.out_a(out)
        #out = self.out_f(out)
        #out = self.out_2(out)
        #out = torch.clamp(out, min=-1, max=1)

        return out

    
    
def conv_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
        #nn.LeakyReLU(0.2, inplace=True),
    )
    return model


def conv_trans_block(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_3(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block(in_dim,out_dim,act_fn),
        conv_block(out_dim,out_dim,act_fn),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
    )
    return model

class Conv_residual_conv(nn.Module):

    def __init__(self,in_dim,out_dim,act_fn):
        super(Conv_residual_conv,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim,self.out_dim,act_fn)
        self.conv_2 = conv_block_3(self.out_dim,self.out_dim,act_fn)
        self.conv_3 = conv_block(self.out_dim,self.out_dim,act_fn)

    def forward(self,input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3



  
    
class XDXD_SpaceNet4_UNetVGG16(nn.Module):
    def __init__(self, num_filters=32, pretrained=False):
        super(XDXD_SpaceNet4_UNetVGG16, self).__init__()
        
        self.encoder_upper = FusionNet()
        self.encoder_lowr = FusionNet()
        
        
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)                
    
    def forward(self,x):
        upper_value = self.encoder_upper(x)
        lower_value = self.encoder_lowr(x)
        
        return_val = upper_value - lower_value
        
        return return_val