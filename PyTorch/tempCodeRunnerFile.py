import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

weight = ResNet34_Weights.DEFAULT

model = resnet34(weight=weight)

