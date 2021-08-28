import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNet(nn.Module):
    def __init__(self,channel,ratio = 16):
        super(GCNet,self).__init__()
        self.relu = nn.ReLU()
        self.ratio = ratio
        self.channel = channel
    def forward(self,x):
        n,c,h,w  = x.size()
        temp = x.view(n,c,-1)
        Wk = nn.Conv2d(self.channel,1,kernel_size = 1)(x)
        Wk = F.softmax(Wk,dim = 2)
        Wk = Wk.view(n,h*w,-1)
        Wk = torch.matmul(temp,Wk).view(n,c,1,1)
        Wv = nn.Conv2d(self.channel,self.channel//self.ratio,kernel_size=1,bias=False)(Wk)
        Wv = nn.LayerNorm([self.channel//self.ratio,1,1])(Wv)
        Wv = self.relu(Wv)
        Wv = nn.Conv2d(self.channel//self.ratio,self.channel,kernel_size=1,bias=False)(Wv)
        return Wv+x

if __name__ == '__main__':
    # device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1,64,32,32).float()
    model = GCNet(channel=64)
    # model = model.to(device)
    res = model(x)
    print(res.shape)
