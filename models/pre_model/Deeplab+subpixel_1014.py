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
import torchvision.models as models

    
# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from .aspp import ASPP, ASPP_Bottleneck
#from .resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8

class _up_piexl(nn.Module):
    def __init__(self, channel_in):
        super(_up_piexl, self).__init__()
        self.relu = nn.ReLU()
        self.subpixel = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.subpixel(out)
        return out

class XDXD_SpaceNet4_UNetVGG16(nn.Module):
    def __init__(self, num_filters=16, pretrained=False):
        super(XDXD_SpaceNet4_UNetVGG16, self).__init__()

        self.num_classes = 1
        self.resnet = ResNet18_OS8()        
        self.aspp = ASPP(num_filters * 16 , num_filters * 8) 
        self.up3 = _up_piexl(num_filters * 16)
        self.up2 = _up_piexl(num_filters * 8)
        self.up1 = _up_piexl(num_filters * 4)

        self.conv_i1 = nn.Conv2d(in_channels=3, out_channels = num_filters , kernel_size=3, stride=1, padding=1)
        self.conv_i2 = nn.Conv2d(in_channels=num_filters , out_channels=num_filters , kernel_size=3, stride=1, padding=1)
        self.bn_i = nn.BatchNorm2d(num_features=num_filters)
        self.relu = nn.ReLU()

        self.bn_3 = nn.BatchNorm2d(num_features=num_filters*16)
        self.conv_31 = nn.Conv2d(in_channels=num_filters * 16, out_channels=num_filters * 16, kernel_size=3, stride=1, padding=1)
        self.conv_32 = nn.Conv2d(in_channels=num_filters * 16, out_channels=num_filters * 16, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(num_features=num_filters*8)
        self.conv_21 = nn.Conv2d(in_channels=num_filters * 8, out_channels=num_filters * 8, kernel_size=3, stride=1, padding=1)
        self.conv_22 = nn.Conv2d(in_channels=num_filters * 8, out_channels=num_filters * 8, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(num_features=num_filters*6)
        self.bn_12 = nn.BatchNorm2d(num_features=num_filters*4)
        self.conv_11 = nn.Conv2d(in_channels=num_filters * 6, out_channels=num_filters * 6, kernel_size=3, stride=1, padding=1)
        self.conv_12 = nn.Conv2d(in_channels=num_filters * 6, out_channels=num_filters * 4, kernel_size=3, stride=1, padding=1)
        self.bn_f = nn.BatchNorm2d(num_features=num_filters)
        self.conv_f1 = nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters, kernel_size=3, stride=1, padding=1)
        self.conv_f2 = nn.Conv2d(in_channels=num_filters, out_channels=self.num_classes, kernel_size=1, stride=1, padding=0)
        #self.relu_f = nn.ReLU()

    def forward(self, x):
         
        residual = x
        residual = self.conv_i1(residual)
        residual = self.bn_i(residual)
        residual = self.relu(residual)
        residual = self.conv_i2(residual)
        residual = self.bn_i(residual)
        residual = self.relu(residual)
        
        h = x.size()[2]
        w = x.size()[3]

        temp_ret = self.resnet(x) 
        enc1, enc2, enc3, enc4 = temp_ret[0], temp_ret[1], temp_ret[2], temp_ret[3]
        
    
        out = self.aspp(enc4) 
        out = torch.cat([enc3, out],1)
        out = self.conv_31(out)
        out = self.relu(self.bn_3(out))
        out = self.conv_32(out)
        out = self.relu(self.bn_3(out))        
        out = self.up3(out)
        out = torch.cat([enc2, out],1)
        out = self.conv_21(out)
        out = self.relu(self.bn_2(out))
        out = self.conv_22(out)
        out = self.relu(self.bn_2(out))
        out = self.up2(out)
        
        out = torch.cat([enc1, out],1)
        
        out = self.conv_11(out)
        
        out = self.relu(self.bn_1(out))
        
        out = self.conv_12(out)
        out = self.relu(self.bn_12(out))
        out = self.up1(out)
        out = torch.cat([residual, out],1)
        out = self.conv_f1 (out)
        out = self.relu(self.bn_f(out))
        out = self.conv_f2 (out)
        
        
        return out

  
    
#################################Resnet#######################################    
    
    
    
def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1) # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels

    layer = nn.Sequential(*blocks) # (*blocks: call with unpacked list entires as arguments)

    return layer

class BasicBlock_2(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock_2, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn2(self.conv2(out)) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(x) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = F.relu(out) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        return out

class Bottleneck_2(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck_2, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, channels, h, w))
        out = F.relu(self.bn2(self.conv2(out))) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn3(self.conv3(out)) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(x) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = F.relu(out) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        return out

class ResNet_Bottleneck_OS16(nn.Module):
    def __init__(self, num_layers):
        super(ResNet_Bottleneck_OS16, self).__init__()

        if num_layers == 50:
            resnet = models.resnet50()
            # load pretrained model:
            resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet50-19c8e357.pth"))
            # remove fully connected layer, avg pool and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            print ("pretrained resnet, 50")
        elif num_layers == 101:
            resnet = models.resnet101()
            # load pretrained model:
            resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet101-5d3b4d8f.pth"))
            # remove fully connected layer, avg pool and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            print ("pretrained resnet, 101")
        elif num_layers == 152:
            resnet = models.resnet152()
            # load pretrained model:
            #resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet152-b121ed2d.pth"))
            # remove fully connected layer, avg pool and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            print ("pretrained resnet, 152")
        else:
            raise Exception("num_layers must be in {50, 101, 152}!")

        self.layer5 = make_layer(Bottleneck_2, in_channels=4*256, channels=512, num_blocks=3, stride=1, dilation=2)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        c4 = self.resnet(x) # (shape: (batch_size, 4*256, h/16, w/16)) (it's called c4 since 16 == 2^4)

        output = self.layer5(c4) # (shape: (batch_size, 4*512, h/16, w/16))

        return output

class ResNet_BasicBlock_OS16(nn.Module):
    def __init__(self, num_layers):
        super(ResNet_BasicBlock_OS16, self).__init__()

        if num_layers == 18:
            resnet = models.resnet18()
            # load pretrained model:
            resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet18-5c106cde.pth"))
            # remove fully connected layer, avg pool and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            num_blocks = 2
            print ("pretrained resnet, 18")
        elif num_layers == 34:
            resnet = models.resnet34()
            # load pretrained model:
            resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet34-333f7ec4.pth"))
            # remove fully connected layer, avg pool and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            num_blocks = 3
            print ("pretrained resnet, 34")
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.layer5 = make_layer(BasicBlock_2, in_channels=256, channels=512, num_blocks=num_blocks, stride=1, dilation=2)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        c4 = self.resnet(x) # (shape: (batch_size, 256, h/16, w/16)) (it's called c4 since 16 == 2^4)

        output = self.layer5(c4) # (shape: (batch_size, 512, h/16, w/16))

        return output

class ResNet_BasicBlock_OS8(nn.Module):
    def __init__(self, num_layers):
        super(ResNet_BasicBlock_OS8, self).__init__()

        if num_layers == 18:
            resnet = models.resnet18()
            # load pretrained model:
            #resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet18-5c106cde.pth"))
            # remove fully connected layer, avg pool, layer4 and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-5])
            
            num_blocks_layer_3 = 2
            num_blocks_layer_4 = 2
            num_blocks_layer_5 = 2
            #print ("pretrained resnet, 18")
        elif num_layers == 34:
            resnet = models.resnet34()
            # load pretrained model:
            resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet34-333f7ec4.pth"))
            # remove fully connected layer, avg pool, layer4 and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])

            num_blocks_layer_4 = 6
            num_blocks_layer_5 = 3
            print ("pretrained resnet, 34")
        else:
            raise Exception("num_layers must be in {18, 34}!")
        
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)    
        self.layer3 = make_layer(BasicBlock_2, in_channels=64, channels=128, num_blocks=num_blocks_layer_3, stride=2, dilation=1)
        self.layer4 = make_layer(BasicBlock_2, in_channels=128, channels=256, num_blocks=num_blocks_layer_4, stride=1, dilation=2)
        self.layer5 = make_layer(BasicBlock_2, in_channels=256, channels=512, num_blocks=num_blocks_layer_5, stride=1, dilation=4)
        
    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        out = self.conv1(x)
        out = self.bn1(out)
        c1 = self.relu(out)  #out = layer1 output
        
        c2 = self.resnet(x)
        #c3 = self.resnet(x) # (shape: (batch_size, 128, h/8, w/8)) (it's called c3 since 8 == 2^3)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3) # (shape: (batch_size, 256, h/8, w/8))
        #c5 = self.layer5(c4) # (shape: (batch_size, 512, h/8, w/8))
        
        return [c1, c2, c3, c4]

def ResNet18_OS16():
    return ResNet_BasicBlock_OS16(num_layers=18)

def ResNet34_OS16():
    return ResNet_BasicBlock_OS16(num_layers=34)

def ResNet50_OS16():
    return ResNet_Bottleneck_OS16(num_layers=50)

def ResNet101_OS16():
    return ResNet_Bottleneck_OS16(num_layers=101)

def ResNet152_OS16():
    return ResNet_Bottleneck_OS16(num_layers=152)

def ResNet18_OS8():
    return ResNet_BasicBlock_OS8(num_layers=18)

def ResNet34_OS8():
    return ResNet_BasicBlock_OS8(num_layers=34)

