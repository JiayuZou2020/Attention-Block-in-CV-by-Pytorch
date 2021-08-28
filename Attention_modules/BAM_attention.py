import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self,in_planes,ratio = 16):
        super(ChannelAttention,self).__init__()
        # nn.AdaptiveAvgPool2d(1)，对输入tensor的NCHW的HW做改变，NC通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes,in_planes//ratio,1,bias = False),
            nn.ReLU(),
            nn.Conv2d(in_planes//ratio,in_planes,1,bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        x = avg_out+max_out
        x = self.sigmoid(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size):
        super(SpatialAttention,self).__init__()

        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2,1,kernel_size,padding = padding,bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        avgout = torch.mean(x,dim = 1,keepdim=True)
        maxout,_ = torch.max(x,dim = 1,keepdim= True)
        x = torch.cat([avgout,maxout],dim = 1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    
    def forward(self,x):
        return x.view(x.size(0),-1)

class ChannelGate(nn.Module):
    def __init__(self,gate_channel,ratio = 16,num_layers = 1):
        super(ChannelGate,self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten',Flatten())

        gate_channels = [gate_channel]
        gate_channels += [gate_channel//ratio]*num_layers
        gate_channels += [gate_channel]

        for i in range(len(gate_channels)-2):
            self.gate_c.add_module(
                'gate_c_fc_%d'%i,nn.Linear(gate_channels[i],gate_channels[i+1])
            )
            self.gate_c.add_module(
                'gate_c_bn_%d'%(i+1),nn.BatchNorm1d(gate_channels[i+1])
            )
            self.gate_c.add_module(
                'gate_c_relu_%d'%(i+1),nn.ReLU()
            )
        self.gate_c.add_module('gate_c_fc_final',
            nn.Linear(gate_channels[-2],gate_channels[-1])
        )

    def forward(self,x):
        avg_pool = F.avg_pool2d(x,x.size(2),stride = x.size(2))
        res = self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3)
        res = res.expand_as(x)
        return res

class SpatialGate(nn.Module):
    def __init__(self,gate_channel,ratio = 16,dilation_conv_num = 2,dialation_val = 4):
        super(SpatialGate,self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module(
            'gate_s_conv_reduce0',
            nn.Conv2d(gate_channel,gate_channel//ratio,kernel_size = 1)
        )
    
        self.gate_s.add_module(
            'gate_s_bn_reduce0',
            nn.BatchNorm2d(gate_channel//ratio)
        )

        self.gate_s.add_module(
            'gate_s_relu_reduce0',nn.ReLU()
        )
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                'gate_s_conv_di_%d'%i,
                nn.Conv2d(gate_channel//ratio,gate_channel//ratio,kernel_size=3,padding = dialation_val,dilation=dialation_val)
            )
            self.gate_s.add_module(
            'gate_s_bn_di_%d'%i,
            nn.BatchNorm2d(gate_channel//ratio)
        )

            self.gate_s.add_module(
                'gate_s_relu_di_%d'%i,nn.ReLU()
            )
        
        self.gate_s.add_module('gate_s_conv_final',
            nn.Conv2d(gate_channel//ratio,1,kernel_size=1)
        )

    def forward(self,x):
        return self.gate_s.expand_as(x)
        
class BAM(nn.Module):
    def __init__(self,gate_channel):
        super(BAM,self).__init__()
        self.channel_attention = ChannelGate(gate_channel)
        self.spatial_attention = SpatialGate(gate_channel)
    def forward(self,x):
        temp = self.channel_attention(x)*self.spatial_attention(x)
        print(temp.shape)
        temp = F.sigmoid(temp)+1
        print(temp.shape)
        return temp*x

if __name__ == '__main__':
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(12,64,32,32).float().cuda()
    model = BAM(gate_channel=64)
    model = model.to(device)
    res = model(x)
    print(res.shape)
