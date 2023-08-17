import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

print(torch.__version__)
# weight = ResNet34_Weights.DEFAULT

# model = resnet34(weights=weight)

# input = torch.rand(size=(4, 3, 256, 256), dtype=torch.float32)

# model_jit = torch.jit.trace(model, input)

# print(model_jit)
# print("__________________________________________________")

# print(model_jit.graph)
# print("__________________________________________________")

# print(model_jit.code)
# print("__________________________________________________")

# print(model(input))
# print("__________________________________________________")

# print(model_jit(input))