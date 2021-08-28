import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import SELU

class SKNet(nn.Module):
    def __init__(self,channel):
        super(SKNet,self).__init__()
        self.conv_3 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size = 5,padding = 2),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.conv_7 = nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size = 7,padding = 3),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channel,channel//2)
        self.linear_back = nn.Linear(channel//2,channel)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self,x):
        n,c,h,w = x.size()
        u_1 = self.conv_3(x)
        u_2 = self.conv_5(x)
        u_3 = self.conv_7(x)
        temp_1 = u_1
        temp_2 = u_2
        temp_3 = u_3
        print(u_3.shape)
        u = u_1+u_2+u_3
        print(u.shape)
        u = self.avg_pool(u)
        print(u.shape)
        u = self.linear(u.mean(-1).mean(-1))        
        u_1 = self.linear_back(u).view(n,c,1,1)
        u_2 = self.linear_back(u).view(n,c,1,1)
        u_3 = self.linear_back(u).view(n,c,1,1)
        u_1 = u_1 * temp_1
        u_2 = u_2 * temp_2
        u_3 = u_3 * temp_3
        u_1 = self.softmax(u_1)
        u_2 = self.softmax(u_2)
        u_3 = self.softmax(u_3)
        u = u_1+u_2+u_3
        print(u.shape)
        return u

if __name__ == '__main__':
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1,64,128,128).cuda()
    model = SKNet(channel=64)
    model = model.to(device)
    res = model(x)
    # print(res.shape)
