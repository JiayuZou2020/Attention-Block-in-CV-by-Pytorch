import torch
import torch.nn as nn
import torch.nn.functional as F

class general_attn(nn.Module):
    def __init__(self,dim,heads):
        super(general_attn,self).__init__()
        self.heads = heads
        self.dim = dim//heads
        self.scale = heads**(-0.5)
        self.qkv = nn.Linear(dim,dim*3)
        self.softmax = nn.Softmax(-1)
        self.linear = nn.Linear(dim,dim)
    def forward(self,x):
        # input shape:B,N,C
        print(x.shape)
        # qkv_matrix shape:B,N,3,heads,C//heads
        qkv_matrix = self.qkv(x).reshape(x.size()[0],x.size()[1],3,self.heads,x.size()[2]//self.heads)
        print(qkv_matrix.shape)
        # qkv_matrix shape:3,B,heads,N,C//heads
        qkv_matrix = qkv_matrix.permute(2,0,3,1,4).contiguous()
        q,k,v = qkv_matrix.chunk(3,0)
        q = q.squeeze()
        k = k.squeeze()
        v = v.squeeze()
        # q,k,v shape:B,heads,C//heads,N
        print(q.shape)
        print('===========================ATTENTION===========================')
        # attention shape:B,heads,N,N
        attention = q @ (k.transpose(2,3)) * self.scale
        attention = self.softmax(attention)
        print(attention.shape)
        # output shape:B,heads,N,C//heads
        output = (attention @ v).transpose(1,2)
        print(output.shape)
        # output shape:B,N,C
        output = output.reshape(x.size()[0],x.size()[1],x.size()[2])
        print(output.shape)
        # output shape:B,N,C
        output = self.linear(output)
        return output

if __name__ == '__main__':
    # inp shape:B,N,C
    inp = torch.randn(3,768,1024)
    dim = 1024
    heads = 8
    model = general_attn(dim = dim,heads = heads)
    res = model(inp)
    print(res.shape)