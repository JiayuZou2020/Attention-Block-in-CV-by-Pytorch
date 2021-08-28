import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F

class SENet(nn.Module):
    def __init__(self,channel,ratio = 16):
        super(SENet,self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.fc_1 = nn.Linear(channel,channel//ratio)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(channel//ratio,channel)

    def forward(self,x):
        temp = x
        temp = self.global_pool(temp).view(x.size(0),x.size(1))
        temp = self.fc_1(temp)
        temp = self.relu(temp)
        temp = self.fc_2(temp)
        temp = self.sigmoid(temp).view(x.size(0),x.size(1),1,1)
        temp = temp.expand_as(x)
        return x+temp

if __name__ == '__main__':
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1,64,128,128).float().cuda()
    model = SENet(channel=64)
    model = model.to(device)
    res = model(x)
    print(res.shape)
