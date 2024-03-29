from torch import nn


class MLP(nn.Module):
    def __init__(self, input_shape, nb_layers, units):
        super().__init__()
        assert isinstance(input_shape, int)
        self._input_shape = input_shape
        self._hidden_units = units

        layers = []

        for i in range(nb_layers):
            if i == 0:
                layers.append(nn.Linear(self._input_shape, self._hidden_units))
            else:
                layers.append(nn.Linear(self._hidden_units, self._hidden_units))
            layers.append(nn.SiLU())

        self._net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self._net(x)
        return x
