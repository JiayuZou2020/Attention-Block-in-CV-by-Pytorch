import torch
import torch.nn as nn
class cSE_attention(nn.Module): 
    def __init__(self,channel):
        super(cSE_attention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear_1 = nn.Sequential(
            nn.Linear(channel,channel//2),
            nn.ReLU()
        )

        self.linear_2 = nn.Sequential(
            nn.Linear(channel//2,channel),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        n,c,h,w = x.size()
        temp = x
        x = self.avg_pool(x)
        x = x.view(-1,c)
        x = self.linear_1(x)
        x = self.linear_2(x)
        # print(x.shape)
        x = x.view(n,c,1,1)
        x = x*temp
        return x

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

class scSE_attention(nn.Module):
    def __init__(self,channel):
        super(scSE_attention,self).__init__()

    def forward(self,x):
        channel = x.size()[1]
        attention_1 = sSE_attention(channel)(x)
        attention_2 = cSE_attention(channel)(x)
        return attention_1+attention_2


if __name__ == '__main__':
    x = torch.randn(1,64,128,128)
    model = scSE_attention(channel=64)
    res = model(x)
    print(res.shape)
