import os

import torch

# if os is windows:
if os.name == "nt":
    import torch_directml


def get_device(prefered_device=None):
    if prefered_device is not None:
        return torch.device(prefered_device)
    if os.name == "nt" and torch_directml.is_available():
        device = torch_directml.device()
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device
