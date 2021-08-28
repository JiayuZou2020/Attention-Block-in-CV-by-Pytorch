import torch
import torch.nn as nn
import torch.nn.functional as F

class non_local(nn.Module):
    def __init__(self,channel):
        super(non_local,self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv2d(channel,channel//2,kernel_size = 1,bias = False),
            nn.BatchNorm2d(channel//2),
            nn.ReLU()
        )
        self.reverse_conv = nn.Sequential(
            nn.Conv2d(channel//2,channel,kernel_size = 1,bias = False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self,x):
        fai = self.conv(x)
        theta = self.conv(x)
        g_function = self.conv(x)
        fai = fai.view(fai.size(0),fai.size(1),-1)
        theta = fai.view(theta.size(0),theta.size(1),-1)
        g_function = fai.view(g_function.size(0),g_function.size(1),-1)
        theta = theta.permute(0,2,1)
        g_function = g_function.permute(0,2,1)
        theta_fai = torch.matmul(theta,fai)
        theta_fai = F.softmax(theta_fai,dim = -1)
        y = torch.matmul(theta_fai,g_function).permute(0,2,1)
        y = y.view(x.size(0),y.size(1),x.size(2),x.size(3))
        y = self.reverse_conv(y)
        return x+y

if __name__ == '__main__':
    x = torch.randn(1,64,32,32).cuda()
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = non_local(channel = 64)
    model = model.to(device)
    res = model(x)
    print(res.shape)
