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
        self.fusion3 = FusionConcat(64, 64)
        self.up_conv4 = NNConv(64, 1, kernel_size=3, dw=True)

    # Literature shows that we should use skip1-3 but we use skip 2-4 and pointwise
    def forward(self, x):
        x, skip_out1 = self.down_conv1(x)                   # x = 240 x 320 || skip1 = 480 x 640
        # print('l1 x: ', x.size(), 'skip: ', skip_out1.size())
        
        x, skip_out2 = self.down_conv2(x)                   # x = 120 x 160 || skip2 = 240 x 320
        # print('l2 x: ', x.size(), 'skip: ', skip_out2.size())
        
        x, skip_out3 = self.down_conv3(x)                   # x = 60 x 80   || skip3 = 120 x 160 
        # print('l3 x: ', x.size(), 'skip: ', skip_out3.size())
        
        x, skip_out4 = self.down_conv4(x)                   # x = 30 x 40   || skip4 = 60 x 80
        # print('l4 x: ', x.size(), 'skip: ', skip_out4.size())
        
        
        # print('l5 x: ', x.size(), 'skip: ', skip_out4.size())
        x = self.fusion1(self.up_conv1(x), skip_out4)
        
        # print('l6 x: ', x.size(), 'skip: ', skip_out3.size())
        x = self.fusion2(self.up_conv2(x), skip_out3)
        
        # print('l7 x: ', x.size(), 'skip: ', skip_out2.size())
        x = self.fusion3(self.up_conv3(x), skip_out2)
        
        return self.up_conv4(x)        