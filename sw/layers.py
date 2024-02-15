import torch 
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(x)
        return (down_out, skip_out)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.relu(self.conv2(x))
    
# NNConv taken from https://github.com/dwofk/fast-depth/blob/master/models.py (FastDepth 2019)  
class NNConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dw):
        super(NNConv, self).__init__()
        if dw:
            self.conv1 = nn.Sequential(
                depthwise(in_channels, kernel_size),
                pointwise(in_channels, out_channels))
        else:
            self.conv1 = conv(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        return F.interpolate(x, scale_factor=2, mode='nearest')        

# Helper methods for NNConv()
# May be able to omit BatchNorm2d (according to https://arxiv.org/pdf/2308.10569v1.pdf)
# but not sure if that should only be considered for backbone or also decoder
def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
        nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        )

def conv(in_channels, out_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=padding,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        )

class FusionConcat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionConcat, self).__init__()
        # Using conv for BatchNorm2d, unsure if this is the correct one to use
        self.conv = conv(in_channels, out_channels, kernel_size=3)
        
    def forward(self, down_input, skip_input):
        x = torch.cat([down_input, skip_input], dim=1)
        return self.conv(x)
        
class FusionElement(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionElement, self).__init__()
        # Using conv for BatchNorm2d, unsure if this is the correct one to use
        self.conv = conv(in_channels, out_channels, kernel_size=3)
    
    def forward(self, down_input, skip_input):
        x = down_input + skip_input
        return self.conv(x)        