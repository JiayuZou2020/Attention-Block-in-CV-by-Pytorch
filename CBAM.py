import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.module import Module

class ChannelAttention(nn.Module):
    def __init__(self,channel,ratio = 16):
        super(ChannelAttention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.share_mlp = nn.Sequential(
            nn.Conv2d(channel,channel//ratio,1,bias = False),
            nn.ReLU(),
            nn.Conv2d(channel//ratio,channel,1,bias = False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        avg_out = self.avg_pool(x)
        avg_out = self.share_mlp(avg_out)
        max_out = self.max_pool(x)
        max_out = self.share_mlp(max_out)
        return self.sigmoid(avg_out+max_out)

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size = 7):
        super(SpatialAttention,self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2,1,kernel_size,padding = padding,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = torch.mean(x,dim = 1,keepdim=True)
        max_out,_ = torch.max(x,dim = 1,keepdim=True)
        out = torch.cat([avg_out,max_out],dim = 1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out
class CBAM(nn.Module):
    def __init__(self,channel):
        super(CBAM,self).__init__()
        self.channelAttn = ChannelAttention(channel)
        self.spatialAttn = SpatialAttention()
        
    def forward(self,x):
        temp = x
        print(temp.shape)
        temp = self.channelAttn(temp)
        print(temp.shape)
        temp = temp*x
        print(temp.shape)
        temp = self.spatialAttn(temp)
        print(temp.shape)
        temp = temp*x
        return temp

if __name__ == '__main__':
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1,64,128,128).float().cuda()
    model = CBAM(channel = 64)
    model = model.to(device)
    res = model(x)
    print(res.shape)
