import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        assert out_dim < 2**in_dim, 'out dim cannot be much higher than in dim'
        # register the buffer too
        self.mask = self.make_mask(in_dim, out_dim)
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))
        hook = self.weight.register_hook(lambda grad: self.mask*grad) # zeros out gradients for masked parts
        if bias:
            self.bias = Parameter(torch.empty(out_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return F.linear(input, self.mask * self.weight, self.bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = self.weight.data * self.mask
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

def make_mask(in_dim, out_dim, mask_type):
    # out_dim x in_dim matrix
    # where each row is unique
    assert out_dim < 2**(in_dim)
    assert in_dim > 0 and out_dim > 0

    if mask_type == 'densest':
        mask = torch.ones(out_dim, in_dim)
        mask[0, :] = 1 # first row is dense
        row_idx = 1
        if out_dim == 1:
            return mask

        for nnz in range(1, in_dim):
            for zeros_in_row in itertools.combinations(range(in_dim), nnz):
                mask[row_idx, zeros_in_row] = 0
                row_idx += 1
                if row_idx >= out_dim:
                    return mask
    else:
        raise ValueError('Invalid mask type')



