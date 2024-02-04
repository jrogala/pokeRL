import numpy as np
import torch
import torch.nn as nn

from . import mlp


class MultiEncoder(nn.Module):
    """A multi encoder.
    Args:
        input_shape (tuple): The shape of the input image.
        depth (int): The number of channels in the first layer.
        blocks (int): The number of residual blocks.
        min_resolution (int): The minimum resolution of the image.
        feature_length (int): The length of the feature vector.
        nb_layers (int): The number of layers in the MLP.
        units (int): The number of units in each layer of the MLP.
    Attributes:
        _input_shape (tuple): The shape of the input image.
        _depth (int): The number of channels in the first layer.
        _blocks (int): The number of residual blocks.
        _min_resolution (int): The minimum resolution of the image.
        _feature_length (int): The length of the feature vector.
        _nb_layers (int): The number of layers in the MLP.
        _units (int): The number of units in each layer of the MLP.
        _net (torch.nn.Sequential): The network.
    """

    def __init__(
        self,
        input_shape=(3, 64, 64),
        depth=48,
        blocks=2,
        min_resolution=4,
        feature_length=12288,
        mlp_layers=3,
        units=1024,
        **kw,
    ):
        super().__init__()
        self._input_shape = input_shape
        self._depth = depth
        self._blocks = blocks
        self._min_resolution = min_resolution
        self._feature_length = feature_length
        self._mlp_layers = mlp_layers
        self._units = units

        self._image_encoder = ImageEncoderResnet(
            input_shape=self._input_shape, depth=self._depth, blocks=self._blocks, min_resolution=self._min_resolution
        )
        self._mlp = mlp.MLP(input_shape=self._feature_length, nb_layers=self._mlp_layers, units=self._units)

    def forward(self, x):
        x = self._image_encoder(x)
        x = self._mlp(x)
        return x


class MultiDecoder(nn.Module):
    """A multi decoder.
    Args:
        feature_length (int): The length of the feature vector.
        output_shape (tuple): The shape of the output image.
        depth (int): The number of channels in the first layer.
        blocks (int): The number of residual blocks.
        min_resolution (int): The minimum resolution of the image.
        sigmoid (bool): Whether to apply sigmoid to the output.
        nb_layers (int): The number of layers in the MLP.
        units (int): The number of units in each layer of the MLP.
    Attributes:
        _feature_length (int): The length of the feature vector.
        _output_shape (tuple): The shape of the output image.
        _depth (int): The number of channels in the first layer.
        _blocks (int): The number of residual blocks.
        _min_resolution (int): The minimum resolution of the image.
        _sigmoid (bool): Whether to apply sigmoid to the output.
        _nb_layers (int): The number of layers in the MLP.
        _units (int): The number of units in each layer of the MLP.
        _net (torch.nn.Sequential): The network.
    """

    def __init__(
        self,
        feature_length=12288,
        output_shape=(1, 128, 128),
        depth=48,
        blocks=2,
        min_resolution=4,
        sigmoid=False,
        mlp_layers=3,
        units=1024,
    ):
        super().__init__()
        self._feature_length = feature_length
        self._output_shape = output_shape
        self._depth = depth
        self._blocks = blocks
        self._min_resolution = min_resolution
        self._sigmoid = sigmoid
        self._mlp_layers = mlp_layers
        self._units = units

        self._mlp = mlp.MLP(input_shape=self._feature_length, nb_layers=self._mlp_layers, units=self._units)
        self._image_decoder = ImageDecoderResnet(
            feature_length=self._units,
            output_shape=self._output_shape,
            depth=self._depth,
            blocks=self._blocks,
            min_resolution=self._min_resolution,
            sigmoid=self._sigmoid,
        )

    def forward(self, z):
        z = self._image_decoder(z)
        z = self._mlp(z)
        return z


class ImageEncoderResnet(nn.Module):
    """A ResNet-based image encoder.
    Args:
        input_shape (tuple): The shape of the input image.
        depth (int): The number of channels in the first layer.
        blocks (int): The number of residual blocks.
        min_resolution (int): The minimum resolution of the image.

    Attributes:
        _depth (int): The number of channels in the first layer.
        _blocks (int): The number of residual blocks.
        _min_resolution (int): The minimum resolution of the image.
        _input_shape (tuple): The shape of the input image.
        _net (torch.nn.Sequential): The network.

    """

    def __init__(self, input_shape=(1, 128, 128), depth=48, blocks=2, min_resolution=4, **kw):
        super().__init__()
        self._depth = depth  # Number of channels in the first layer
        self._blocks = blocks  # Number of residual blocks
        self._min_resolution = min_resolution  # Minimum resolution of the image
        self._input_shape = input_shape

        # Compute the number of layers that are needed to reduce image width to min_resolution
        stages = int(np.log2(self._input_shape[-2]) - np.log2(self._min_resolution))
        layers = []
        for i in range(stages):
            in_channels = (
                self._input_shape[-3] if i == 0 else self._depth * (2 ** (i - 1))
            )  # Compute the number of channels for each convolution
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
    """A ResNet-based image decoder.
    Args:
        feature_length (int): The length of the feature vector.
        output_shape (tuple): The shape of the output image.
        depth (int): The number of channels in the first layer.
        blocks (int): The number of residual blocks.
        min_resolution (int): The minimum resolution of the image.
        sigmoid (bool): Whether to apply sigmoid to the output.
    Attributes:
        _feature_length (int): The length of the feature vector.
        _output_shape (tuple): The shape of the output image.
        _depth (int): The number of channels in the first layer.
        _blocks (int): The number of residual blocks.
        _min_resolution (int): The minimum resolution of the image.
        _sigmoid (bool): Whether to apply sigmoid to the output.
        _net (torch.nn.Sequential): The network.
    """

    def __init__(
        self, feature_length=12288, output_shape=(1, 128, 128), depth=48, blocks=2, min_resolution=4, sigmoid=False
    ):
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
        self._lin = nn.Linear(self._feature_length, in_depth * self._min_resolution**2)
        for i in range(stages):
            for _ in range(self._blocks):
                layers.append(ResidualBlock(in_depth // (2 ** (i)), transpose=True))
            in_channels = int(in_depth // (2 ** (i)))
            out_channels = int(in_depth // (2 ** (i + 1))) if i < stages - 1 else self._output_shape[-3]
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))

        self._net = nn.Sequential(*layers)

    def forward(self, z):
        x = self._lin(z)
        x = x.reshape(x.shape[0] // (2**self._min_resolution), self._min_resolution, self._min_resolution)
        x = self._net(x)
        if self._sigmoid:
            x = torch.sigmoid(x)
        else:
            x += 0.5
        return x


class ResidualBlock(nn.Module):
    """A residual block.
    Args:
        depth (int): The number of channels in the first layer.
        transpose (bool): Whether to use a convolution transpose.
    Attributes:
        _depth (int): The number of channels in the first layer.
        _act1 (torch.nn.SiLU): The activation function.
        _act2 (torch.nn.SiLU): The activation function.
        _conv1 (torch.nn.Conv2d): The first convolution.
        _conv2 (torch.nn.Conv2d): The second convolution.
    """

    def __init__(self, depth, transpose=False, **kw):
        super().__init__()
        self._depth = depth
        self._act1 = nn.SiLU()
        self._act2 = nn.SiLU()
        if transpose:
            self._conv1 = nn.ConvTranspose2d(self._depth, self._depth, kernel_size=3, stride=1, padding=1)
            self._conv2 = nn.ConvTranspose2d(self._depth, self._depth, kernel_size=3, stride=1, padding=1)
        else:
            self._conv1 = nn.Conv2d(self._depth, self._depth, kernel_size=3, stride=1, padding=1)
            self._conv2 = nn.Conv2d(self._depth, self._depth, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        x = self._conv1(x)
        x = self._act1(x)
        x = self._conv2(x)
        x += identity
        return self._act2(x)
