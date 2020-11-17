import os
import torch
from torch import nn
from torchvision.models import vgg16
import torch.nn.functional as F
import torch.nn as nn

# bear's add part
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# bear's work part ==================================================================
    
class _ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_ConvBNRelu, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out = self.block(x)
        return out

class _res_block(nn.Module):
    def __init__(self, in_channels):
        super(_res_block, self).__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.block_1(x)
        out = self.block_2(out)
        return out

class _down_sampling(nn.Module):
    def __init__(self, in_channels):
        super(_down_sampling, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=2*in_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(2*in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.bn(self.conv(x))
        out = self.relu(out)
        return out
    
class _up_sampling(nn.Module):
    def __init__(self):
        super(_up_sampling, self).__init__()
        self.subpixel = nn.PixelShuffle(2)
        
    def forward(self, x):
        out = self.subpixel(x)
        return out
    
class XDXD_SpaceNet4_UNetVGG16(nn.Module):
    def __init__(self, num_filters=16, pretrained=False):
        super(XDXD_SpaceNet4_UNetVGG16, self).__init__()

        # shape : 64, 512 x 512
        self.block_init = _ConvBNRelu(in_channels=3, out_channels=64)
        self.block_1_1 = _res_block(in_channels=64)
        self.block_1_2 = _res_block(in_channels=64)
        self.down_1 = _down_sampling(in_channels=64)
        
        # shape : 128, 256 x 256
        self.block_2_1 = _res_block(in_channels=128)
        self.block_2_2 = _res_block(in_channels=128)
        self.down_2 = _down_sampling(in_channels=128)

        # shape : 256, 128 x 128
        self.block_3_1 = _res_block(in_channels=256)
        self.block_3_2 = _res_block(in_channels=256)
        self.block_3_3 = _res_block(in_channels=256)
        self.block_3_4 = _res_block(in_channels=256)
        self.down_3 = _down_sampling(in_channels=256)
        
        # shape : 512, 64 x 64
        self.block_4_1 = _res_block(in_channels=512)
        self.block_4_2 = _res_block(in_channels=512)
        self.block_4_3 = _res_block(in_channels=512)
        self.block_4_4 = _res_block(in_channels=512)
        
        # shape : 512, 128 x 128
        self.decon_block_3 = _up_sampling()
        self.decon_block_3_1 = _res_block(in_channels=512)
        self.decon_block_3_2 = _res_block(in_channels=512)
        
        # shape : 256, 256 x 256
        self.decon_block_2 = _up_sampling()
        self.decon_block_2_1 = _res_block(in_channels=256)
        self.decon_block_2_2 = _res_block(in_channels=256)
        
        # shape : 128, 512 x 512
        self.decon_block_1 = _up_sampling()
        self.decon_block_1_1 = _res_block(in_channels=128)
        self.decon_block_1_2 = _res_block(in_channels=128)
        
        self.output_conv = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #residual = x
        
        # init block : 512 x 512
        out = self.block_init(x)
        # residual part - 1
        out = self.block_1_1(out)
        out1 = self.block_1_2(out)
        out = self.down_1(out1)
        
        # residual part - 2 : 256 x 256
        out = self.block_2_1(out)
        out2= self.block_2_2(out)
        out = self.down_2(out2)
        
        # residual part - 3 : 128 x 128
        out = self.block_3_1(out)
        out = self.block_3_2(out)
        out = self.block_3_3(out)
        out3= self.block_3_4(out)
        out4 = self.down_3(out3)
        
        # residual part - 4 : 64 x 64
        out = self.block_4_1(out4)
        out = self.block_4_2(out)
        out = self.block_4_3(out)
        out = self.block_4_4(out)
        conc = torch.cat([out4, out], 1)
        
        # deconv part - 3 : 128 x 128        
        out = self.decon_block_3(conc)
        conc = torch.cat([out3, out], 1)
        out = self.decon_block_3_1(conc)
        out = self.decon_block_3_2(out)
                
        # deconv part - 2: 256 x 256   
        out = self.decon_block_2(out)
        conc = torch.cat([out2, out], 1)
        out = self.decon_block_2_1(conc)
        out = self.decon_block_2_2(out)
                
        # deconv part - 1 : 512 x 512
        out = self.decon_block_1(out)
        conc = torch.cat([out1, out], 1)
        out = self.decon_block_1_1(conc)
        out = self.decon_block_1_2(out)
        
        out = self.output_conv(out)
        out = nn.BatchNorm2d(1)(out) # add this part
        return out