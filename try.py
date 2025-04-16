from convnext import ConvNeXt
import torch
model_conv = ConvNeXt()
state_dict = torch.load('convnext.pth', map_location=torch.device('cpu'))
print(state_dict.keys())
