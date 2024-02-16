import torch

from torchsummary import summary
from thop import profile
from model import MonoDepth

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MonoDepth().to(device)
input = torch.randn(1, 3, 480, 640).to(device=device)

summary(model, input_size=(3, 480, 640))

macs, params = profile(model, inputs=(input, ))

# Pretty print the results
print(f"MACs: {macs / 1e9} billion")
print(f"Parameters: {params / 1e6} million")
