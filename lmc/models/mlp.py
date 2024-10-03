import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_lin import SparseLinear

class ImageMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dim, out_dim, num_layers)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.mlp(x)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, activation=nn.GELU(), sparse=False):
        super().__init__()
        self.lins = nn.ModuleList()
        self.activation = activation
        linear_builder = SparseLinear if sparse else nn.Linear
        if num_layers == 1:
            self.lins.append(nn.Linear(in_dim, out_dim)) # last layer always dense
        else:
            self.lins.append(linear_builder(in_dim, hidden_dim))
            for _ in range(num_layers-2):
                self.lins.append(linear_builder(hidden_dim, hidden_dim))
            self.lins.append(nn.Linear(hidden_dim, out_dim)) # last layer always dense

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = self.activation(x)
        x = self.lins[-1](x)
        return x