import torch
import torch.nn as nn
import torch.nn.functional as F
from decoders import GCASCADE, GCASCADE_Cat
from pvtv2 import pvt_v2_b0

class PVT_GCASCADE(nn.Module):
    def __init__(self, n_class=1, img_size_h=640, img_size_w=480, k=11, padding=5, conv='mr', gcb_act='gelu',
                 activation='relu', skip_aggregation='additive'):
        super(PVT_GCASCADE, self).__init__()

        self.skip_aggregation = skip_aggregation
        self.n_class = n_class

        # backbone network initialization with pretrained weight
        self.backbone = pvt_v2_b0()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt/pvt_v2_b0.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # self.channels = [512, 320, 128, 64] # for pvt_v2_b2
        self.channels = [256, 160, 64, 32] # for pvt_v2_b0

        # decoder initialization
        if self.skip_aggregation == 'additive':
            self.decoder = GCASCADE(channels=self.channels, img_size_h=img_size_h, img_size_w=img_size_w, k=k,
                                    padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
        elif self.skip_aggregation == 'concatenation':
            self.decoder = GCASCADE_Cat(channels=self.channels, img_size_h=img_size_h, img_size_w=img_size_w, k=k,
                                        padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)
            self.channels = [self.channels[0], self.channels[1] * 2, self.channels[2] * 2, self.channels[3] * 2]
        else:
            print('No implementation found for the skip_aggregation ' + self.skip_aggregation + '. Continuing with the default additive aggregation.')
            self.decoder = GCASCADE(channels=self.channels, img_size_h=img_size_h, img_size_w=img_size_w, k=k,
                                    padding=padding, conv=conv, gcb_act=gcb_act, activation=activation)

        print('Model %s created, param count: %d' %
              ('GCASCADE decoder: ', sum([m.numel() for m in self.decoder.parameters()])))

        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)

    def forward(self, x):
        # transformer backbone as encoder
        x1, x2, x3, x4 = self.backbone(x)

        # decoder
        x1_o, x2_o, x3_o, x4_o = self.decoder(x4, [x3, x2, x1])

        # prediction heads
        p1 = self.out_head1(x1_o)
        p2 = self.out_head2(x2_o)
        p3 = self.out_head3(x3_o)
        p4 = self.out_head4(x4_o)

        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')
        return p1, p2, p3, p4

if __name__ == '__main__':
    model = PVT_GCASCADE(img_size_h=640, img_size_w=480).cuda()
    # summary(model, input_size=(1,3,640,480), device=torch.device('cuda'))
    # input_tensor = torch.randn(1,3,640,480).cuda()
    # flops, params = profile(model, inputs=(input_tensor,), verbose=True)
    # print(f"FLOPs: {flops}, Parameters: {params}")
    # FLOPs: 3,811,433,400.0, Parameters: 3,860,806.0
    # FLOPs: 3.811 GFLOPs, Parameters: 3.86M

    input_tensor = torch.randn(1, 3, 640, 480).cuda()
    p1, p2, p3, p4 = model(input_tensor)
    print(p1.size(), p2.size(), p3.size(), p4.size())