import numpy as np
import torch.nn as nn


class ImageEncoderResnet(nn.Module):
    def __init__(self, depth=48, blocks=2, input_shape = (3,64,64), min_resolution=4, **kw):
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
    

class ResidualBlock(nn.Module):
    def __init__(self, depth, **kw):
        super().__init__()
        self._depth = depth
        self._conv1 = nn.Conv2d(self._depth, self._depth, kernel_size=3, stride=1, padding=1)
        self._act1 = nn.SiLU()
        self._conv2 = nn.Conv2d(self._depth, self._depth, kernel_size=3, stride=1, padding=1)
        self._act2 = nn.SiLU()

    def forward(self, x):
        identity = x
        x = self._conv1(x)
        x = self._act1(x)
        x = self._conv2(x)
        x += identity
        return self._act2(x)
