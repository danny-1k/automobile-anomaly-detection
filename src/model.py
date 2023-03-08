import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, config: list, act: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx, layer in enumerate(config):
            self.layers.append(nn.Linear(layer[0], layer[1]))
            if idx != len(config) - 1:
                self.layers.append(act())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
