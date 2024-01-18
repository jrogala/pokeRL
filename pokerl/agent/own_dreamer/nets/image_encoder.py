import numpy as np
import torch.nn as nn


class ImageEncoderResnet(nn.Module):
    def __init__(self, depth, blocks, input_shape, min_resolution, **kw):
        super().__init__()
        self._depth = depth  # Number of channels in the first layer
        self._blocks = blocks  # Number of residual blocks
        self._min_resolution = min_resolution  # Minimum resolution of the image
        self._input_shape = input_shape

        # Compute the number of layers that are needed to reduce image width to min_resolution
        stages = int(np.log2(self._input_shape[-2]) - np.log2(self._min_resolution))
        self._convs = nn.ModuleList()
        for i in range(stages):
            in_channels = self._input_shape[-3] if i == 0 else self._depth * (2 ** (i - 1))  # Compute
            out_channels = self._depth * (2**i)
            self._convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2))
            # Conv2d (depth, kernel, stride)

    def forward(self, x):
        x = x - 0.5  # Normalize the image around 0
        for i in range(len(self.convs)):
            current_depth = self._depth * (2**i)  # Compute the current depth for Convolutions
            x = self.convs[i](x)
            for _j in range(self._blocks):
                identity = x
                x = nn.Conv2d(current_depth, current_depth, kernel_size=3, stride=1)(x)
                x = nn.Conv2d(current_depth, current_depth, kernel_size=3, stride=1)(x)
                x += identity
        x = x.reshape(x.shape[0], -1)
        return x
