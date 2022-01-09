import torch

from .Module import Module

class TorchModule(Module):

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self):
        return self._device
