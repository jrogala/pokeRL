import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, discrete_representation_z, recurent_state_h, output_dim, C):
        super(Decoder, self).__init__()
        self.discrete_representation_z = discrete_representation_z
        self.recurent_state_h = recurent_state_h
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.discrete_representation_z + self.recurent_state_h, 256)
        self.inverted_encoder = nn.Sequential()

    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        x = F.relu(self.fc1(x))
        x = torch.reshape(x, (4,4,C))
        x = self.inverted_encoder(x)
        return x
        
