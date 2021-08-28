import torch
import torch.nn as nn
import torch.nn.functional as F

class sSE_attention(nn.Module): 
    def __init__(self,channel):
        super(sSE_attention,self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(channel,1,kernel_size=1,bias = False)
    
    def forward(self,x):
        temp = x
        x = self.conv(x)
        x = self.sigmoid(x)
        x = x*temp
        return x
if __name__ == '__main__':
    x = torch.randn(1,64,128,128)
    model = sSE_attention(channel=64)
    res = model(x)
    print(res.shape)
