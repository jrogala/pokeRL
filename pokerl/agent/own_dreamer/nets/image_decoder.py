import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageDecoderResnet(nn.Module):
    def __init__(self, input_shape = (3,64,64), depth=48, blocks=2, minres=4, sigmoid=False, **kw):
        self._shape = input_shape
        self._depth = depth
        self._blocks = blocks
        self._minres = minres
        self._sigmoid = sigmoid
        self._kw = kw

    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        x = F.relu(self.fc1(x))
        x = torch.reshape(x, (4,4,C))
        x = self.inverted_encoder(x)
        return x
        
