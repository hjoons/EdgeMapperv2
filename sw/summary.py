import torch

from torchsummary import summary
from thop import profile
from model import MonoDepth

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device
model = MonoDepth().to(device)
# input = torch.randn(1, 3, 480, 640)
# macs, params = profile(model, inputs=(input, ))

summary(model, input_size=(3, 480, 640))