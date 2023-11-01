import torch
import torch.nn as nn
import torch.nn.functional as F

class downsample_net(nn.Module):
    def __init__(self, in_dim):
        super(downsample_net,self).__init__()
        self.ln1 = nn.Linear(20*in_dim, 10*in_dim)
        self.ln2 = nn.Linear(10*in_dim, 5*in_dim)

        self.bn1 = nn.BatchNorm1d(10*in_dim)
        self.bn2 = nn.BatchNorm1d(5*in_dim)

        self.ln3 = nn.Linear(5*in_dim, 512)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.ln1(x)))
        x = F.relu(self.bn2(self.ln2(x))) 
        
        out = self.ln3(x)        
        return out


class double_conv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1):
        super(double_conv2d_bn,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,
                               kernel_size=kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,
                              kernel_size = kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x.float())))
        out = F.relu(self.bn2(self.conv2(out)))
        return out
    
    
class conv_net(nn.Module):
    def __init__(self):
        super(conv_net,self).__init__()
        self.layer1_conv = double_conv2d_bn(20,20)
        self.layer2_conv = double_conv2d_bn(20,20)
        self.layer3_conv = double_conv2d_bn(20,20)

        
    def forward(self,x):
        B_sz = x.size(0)
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1,4)
        
        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2,3)

        conv3 = self.layer3_conv(pool2)
        outp = F.max_pool2d(conv3,3)
        
        return outp.view(B_sz, 20, -1)