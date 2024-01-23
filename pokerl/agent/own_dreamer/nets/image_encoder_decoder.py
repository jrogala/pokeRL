import torch

import numpy as np
import torch.nn as nn


class ImageEncoderResnet(nn.Module):
    def __init__(self, input_shape = (3,64,64), depth=48, blocks=2, min_resolution=4, **kw):
        super().__init__()
        self._depth = depth  # Number of channels in the first layer
        self._blocks = blocks  # Number of residual blocks
        self._min_resolution = min_resolution  # Minimum resolution of the image
        self._input_shape = input_shape

        # Compute the number of layers that are needed to reduce image width to min_resolution
        stages = int(np.log2(self._input_shape[-2]) - np.log2(self._min_resolution))
        layers = []
        for i in range(stages):
            in_channels = self._input_shape[-3] if i == 0 else self._depth * (2 ** (i - 1))  # Compute the number of channels for each convolution
            out_channels = self._depth * (2**i)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            for _ in range(self._blocks):
                layers.append(ResidualBlock(self._depth * (2**i)))
        self._net = nn.Sequential(*layers)

    def forward(self, x):
        x = x - 0.5  # Normalize the image around 0
        x = self._net(x)
        x = x.reshape(x.shape[0], -1)
        return x

class ImageDecoderResnet(nn.Module):
    def __init__(self, feature_length=12288, output_shape = (3,128,128), depth=48, blocks=2, min_resolution=4, sigmoid=False, **kw):
        super().__init__()
        self._feature_length = feature_length
        self._output_shape = output_shape
        self._depth = depth
        self._blocks = blocks
        self._min_resolution = min_resolution
        self._sigmoid = sigmoid
                
        layers = []
        
        stages = int(np.log2(self._output_shape[-2]) - np.log2(self._min_resolution))
        in_depth = depth * 2 ** (stages - 1)
        self._lin = nn.Linear(self._feature_length, in_depth * self._min_resolution ** 2)
        for i in range(stages):
            for _ in range(self._blocks):
                layers.append(ResidualBlock(in_depth // (2**i), transpose=True))
            in_channels = int(in_depth // (2 ** (i + 1)))
            out_channels = int(in_depth // (2 ** (i + 2))) if i < stages - 1 else self._output_shape[-3]
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        
        self._net = nn.Sequential(*layers)
        
    def forward(self, z):
        x = self._lin(z)
        x = x.reshape(x.shape[0]//(2 ** self._min_resolution), self._min_resolution, self._min_resolution)
        x = self._net(x)
        if self._sigmoid:
            x = torch.sigmoid(x)
        else:
            x += 0.5
        return x


class ResidualBlock(nn.Module):
    def __init__(self, depth, transpose=False, **kw):
        super().__init__()
        self._depth = depth
        self._act1 = nn.SiLU()
        self._act2 = nn.SiLU()
        if transpose:
            self._conv1 = nn.ConvTranspose2d(self._depth, self._depth, kernel_size=3, stride=1, padding=1)
            self._conv2 = nn.ConvTranspose2d(self._depth, self._depth, kernel_size=3, stride=1, padding=1)
        else :
            self._conv1 = nn.Conv2d(self._depth, self._depth, kernel_size=3, stride=1, padding=1)
            self._conv2 = nn.Conv2d(self._depth, self._depth, kernel_size=3, stride=1, padding=1)
        

    def forward(self, x):
        identity = x
        print(x.shape)
        x = self._conv1(x)
        x = self._act1(x)
        print(x.shape)
        x = self._conv2(x)
        x += identity
        return self._act2(x)
