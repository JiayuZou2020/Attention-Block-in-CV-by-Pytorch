import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
class ARM(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(ARM,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=1)
    def forward(self,x):
        avg_pool = self.avg_pool(x)
        conv_out = self.conv(avg_pool)
        conv_out = self.batchnorm(conv_out)
        conv_out = self.sigmoid(conv_out)
        out = torch.mul(x,conv_out)
        return out

class FFM(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(FFM,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size = 3,padding = 1),
            nn.ReLU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.batchnorm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(out_channel,out_channel,kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x,y):
        input = torch.cat([x,y],dim = 1)
        out = self.conv(input)
        branch_1 = self.avg_pool(out)
        branch_1 = self.conv_1(branch_1)
        branch_1 = self.relu(branch_1)
        branch_1 = self.conv_1(branch_1)
        branch_1 = self.sigmoid(branch_1)
        res = torch.mul(out,branch_1)+out
        return res

class BiSeNet(nn.Module):
    def __init__(self,in_channel):
        super(BiSeNet,self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size = 3,padding = 1, stride = 2, bias = False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channel,in_channel,kernel_size = 3, stride = 4, bias = False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.down_conv_2 = nn.Sequential(
            nn.Conv2d(in_channel,in_channel//2,kernel_size = 3, stride = 2,padding = 1, bias = False),
            nn.BatchNorm2d(in_channel//2),
            nn.ReLU()
        )
        self.basic_conv_2 = nn.Sequential(
            nn.Conv2d(in_channel//2,in_channel//2,kernel_size = 3,padding = 1, stride = 2, bias = False),
            nn.BatchNorm2d(in_channel//2),
            nn.ReLU()
        )
    def forward(self,x):
        print(x.shape)
        spatial_path = self.basic_conv(x)
        spatial_path = self.basic_conv(spatial_path)
        spatial_path = self.down_conv_2(spatial_path)
        print(spatial_path.shape)
        context_path = self.down_conv(x)
        print(context_path.shape)
        context_path = self.basic_conv(context_path)
        print(context_path.shape)
        context_path = self.basic_conv(context_path)
        print(context_path.shape)
        context_path = self.down_conv_2(context_path)
        print(context_path.shape)
        context_path_1 = ARM(context_path.size(1),context_path.size(1))(context_path)
        context_path_2 = self.basic_conv_2(context_path)
        context_path_3 = ARM(context_path_2.size(1),context_path_2.size(1))(context_path_2)
        # print(spatial_path.shape)
        context_path_3 = context_path_3+context_path_2
        print('-'*30)
        context_path_3 = F.interpolate(context_path_3,size = context_path_1.size()[-2:],mode = 'bilinear')
        context_path_1 = context_path_1+context_path_3
        context_path_1 = F.interpolate(context_path_1,size = spatial_path.size()[-2:],mode = 'bilinear')
        print(context_path_1.shape)
        print(spatial_path.shape)
        out = FFM(spatial_path.size()[1]*2,spatial_path.size()[1]*2)(spatial_path,context_path_1)
        out = F.interpolate(out,size = x.size()[-2:],mode = 'bilinear')
        print(out.shape)
        return out

if __name__ == '__main__':
    # device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(4,64,128,128)
    model = BiSeNet(in_channel = 64)
    # model = model.to(device)
    res = model(x)
    

