import torch
import torch.nn as nn

class channel_shuffle(nn.Module):
    def __init__(self,channels):
        super(channel_shuffle,self).__init__()
        self.channels = channels

    def forward(self,x):
        n,c,h,w = x.shape
        x = x.reshape(n,self.channels,-1,h,w)
        x = x.permute(0,2,1,3,4)
        x = x.reshape(n,c,h,w)
        return x 

class shuffle_attention(nn.Module):
    def __init__(self,channel,groups):
        super(shuffle_attention,self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.groups = groups
        self.linear = nn.Linear(channel//(2*self.groups),channel//(2*self.groups))
        self.groupnorm = nn.GroupNorm(channel//(2*self.groups),channel//(2*self.groups))

    def forward(self,x):
        n,c,h,w = x.shape
        print(x.shape)
        x = x.view(n*self.groups,-1,h,w)
        print(x.shape)
        x_channel,x_space = x.chunk(2,dim = 1)
        print(x_channel.shape)
        print(x_space.shape)
        temp_space = x_channel
        temp_channel = x_space
        x_channel = self.avg_pool(x_channel)
        x_channel = x_channel.view(-1,c//(2*self.groups))
        x_channel = self.linear(x_channel).view(-1,c//(2*self.groups),1,1)
        x_channel = self.sigmoid(x_channel)
        x_channel = temp_space*x_channel

        x_space = self.groupnorm(x_space)
        print(x_space.shape)
        x_space = x_space.view(-1,c//(2*self.groups))
        x_space = self.linear(x_space).view(-1,c//(2*self.groups),h,w)
        x_space = self.sigmoid(x_space)
        x_space = temp_channel*x_space
        x_final = torch.cat([x_channel,x_space],dim = 1)
        x_final = x_final.reshape(n,-1,h,w)
        print(x_final.shape)
        x_final = channel_shuffle(channels = 2)(x_final)
        return x_final

if __name__ == '__main__':
    x = torch.randn(1,64,128,128)
    model = shuffle_attention(channel = x.size()[1],groups = 4)
    res = model(x)
    print(res.shape)
