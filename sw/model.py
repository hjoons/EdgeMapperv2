from layers import *

class MonoDepth(nn.Module):
    def __init__(self):
        super(MonoDepth, self).__init__()
        
        self.down_conv1 = DownBlock(3, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        
        self.up_conv1 = NNConv(512, 256, kernel_size=3, dw=True)
        self.fusion1 = FusionElement(256, 256)
        self.up_conv2 = NNConv(256, 128, kernel_size=3, dw=True)
        self.fusion2 = FusionElement(128, 128)
        self.up_conv3 = NNConv(128, 64, kernel_size=3, dw=True)
        self.fusion3 = FusionElement(64, 64)
        self.up_conv4 = pointwise(64, 1)

    def forward(self, x):
        print('l1', x.size())
        x, skip_out1 = self.down_conv1(x)
        print('l2', x.size())
        x, skip_out2 = self.down_conv2(x)
        print('l3', x.size())
        x, skip_out3 = self.down_conv3(x)
        print('l4', x.size())
        x, skip_out4 = self.down_conv4(x) # Fourth skip connection unused ??
        print('l5', x.size())
        x = self.fusion1(self.up_conv1(x), skip_out3)
        print('l6', x.size())
        x = self.fusion2(self.up_conv2(x), skip_out2)
        print('l7', x.size())
        x = self.fusion3(self.up_conv3(x), skip_out1)
        print('l8', x.size())
        return self.up_conv4(x)        