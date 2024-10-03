import math
import itertools
import copy
import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
import random
import numpy as np
class AsymSwiGLU(nn.Module):
     def __init__(self, dim, scale=1.0, mask_num=0):
         super().__init__()
         g = torch.Generator()
         g.manual_seed(abs(hash(str(mask_num)+ str(0))))
         C = torch.randn(dim, dim, generator=g)
         self.register_buffer("C", C)
     def forward(self, x):
         gate = F.sigmoid(F.linear(x, self.C))
         return gate * x


class SigmaMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, norm = None, asym_act=True):
        super().__init__()
        self.lins = nn.ModuleList()
        self.activations = nn.ModuleList()
        if asym_act:
            for i in range(num_layers - 1):
                self.activations.append(AsymSwiGLU(hidden_dim, mask_num=i))
        else:
            for i in range(num_layers - 1):
                self.activations.append(nn.GELU())
        if not norm:
          self.norm = None
        else:
          self.norms = nn.ModuleList()
          if norm == 'layer':
              self.norm = nn.LayerNorm
          elif norm== 'batch':
            self.norm = nn.BatchNorm1d
          else:
            raise ValueError("Bad norm type. Should be 'layer' or 'batch'")

        if num_layers == 1:
            self.lins.append(nn.Linear(in_dim, out_dim))

        else:
            if self.norm:
              for _ in range(num_layers - 1):
                self.norms.append(self.norm(hidden_dim))

            self.lins.append(nn.Linear(in_dim, hidden_dim))

            for _ in range(num_layers-2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.lins.append(nn.Linear(hidden_dim, out_dim))
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.flatten(x)

        for idx, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.norm:
              x = self.norms[idx](x)
            x = self.activations[idx](x)
        x = self.lins[-1](x)
        return x

    def count_unused_params(self):
        return 0

class WMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, mask_params, norm = None):
        super().__init__()
        self.lins = nn.ModuleList()
        #Handle norm first
        if not norm:
          self.norm = None
        else:
          self.norms = nn.ModuleList()
          if norm == 'layer':
              self.norm = nn.LayerNorm
          elif norm== 'batch':
            self.norm = nn.BatchNorm1d
          else:
            raise ValueError("Bad norm type. Should be 'layer' or 'batch'")

        #setup Lins
        if num_layers == 1:
            self.lins.append(SparseLinear(in_dim, out_dim, **mask_params[0], mask_num = 0))

        else:
            if self.norm:
              for _ in range(num_layers - 1):
                self.norms.append(self.norm(hidden_dim))

            self.lins.append(SparseLinear(in_dim, hidden_dim, **mask_params[0], mask_num = 0))
            for i in range(num_layers-2):
                self.lins.append(SparseLinear(hidden_dim, hidden_dim, **mask_params[i+1], mask_num = i+1))
            self.lins.append(SparseLinear(hidden_dim, out_dim, **mask_params[num_layers - 1], mask_num = num_layers - 1))
        self.activation = nn.GELU()
            
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)

        for idx, lin in enumerate(self.lins[:-1]):
            prev=x
            x = lin(x)
    
            if self.norm:
                x = self.norms[idx](x)
            x = self.activation(x)
            
        x = self.lins[-1](x)
        return x

    def count_unused_params(self):
        return sum(lin.count_unused_params() for lin in self.lins if type(lin) != nn.Linear)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, norm = None):
        super().__init__()
        self.lins = nn.ModuleList()
        self.activation = nn.GELU()
        if not norm:
          self.norm = None
        else:
          self.norms = nn.ModuleList()
          if norm == 'layer':
              self.norm = nn.LayerNorm
          elif norm== 'batch':
            self.norm = nn.BatchNorm1d
          else:
            raise ValueError("Bad norm type. Should be 'layer' or 'batch'")

        if num_layers == 1:
            self.lins.append(nn.Linear(in_dim, out_dim))

        else:
            if self.norm:
              for _ in range(num_layers - 1):
                self.norms.append(self.norm(hidden_dim))

            self.lins.append(nn.Linear(in_dim, hidden_dim))

            for _ in range(num_layers-2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.lins.append(nn.Linear(hidden_dim, out_dim))
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.flatten(x)

        for idx, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.norm:
              x = self.norms[idx](x)
            x = self.activation(x)
        x = self.lins[-1](x)
        return x

    def count_unused_params(self):
        return 0
class SparseLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, mask_type='densest', mask_constant = 1, mask_num = 0, num_fixed = 6, do_normal_mask = True):
        super().__init__()
        assert out_dim < 2**in_dim, 'out dim cannot be much higher than in dim'
        mask = make_mask(in_dim, out_dim, mask_type=mask_type, num_fixed = num_fixed, mask_num = mask_num)

        self.register_buffer('mask', mask, persistent=True)
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))

        if do_normal_mask:
            self.register_buffer('normal_mask', normal_mask(out_dim, in_dim, mask_num), persistent=True)
        else:
            self.register_buffer('normal_mask', torch.ones(size = (out_dim, in_dim)), persistent=True) #torch.ones -> does nothing

        hook = self.weight.register_hook(lambda grad: self.mask*grad) # zeros out gradients for masked parts

        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter('bias', None)

        self.mask_constant = mask_constant
        self.mask_num = mask_num
        self.num_fixed = num_fixed
        self.reset_parameters()
    def forward(self, x):
        self.weight.data = (self.weight.data* self.mask + (1-self.mask)*self.mask_constant*self.normal_mask) 
        
        return F.linear(x, self.weight, self.bias)
        #return F.linear(x, self.mask * self.weight, self.bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = (self.weight.data* self.mask + (1-self.mask)*self.mask_constant*self.normal_mask) #set entries where mask is zero to the normal mask at that point

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def count_unused_params(self):
        return (1-self.mask.int()).sum().item()

def get_subset(num_cols, row_idx, num_sample, mask_num):
    g = torch.Generator()
    g.manual_seed(row_idx + abs(hash(str(mask_num))))
    indices = torch.arange(num_cols)
    return (indices[torch.randperm(num_cols, generator = g)[:num_sample]])

def normal_mask(out_dim,in_dim, mask_num):

    g = torch.Generator()
    g.manual_seed(abs(hash(str(mask_num))))
    return torch.randn(size=(out_dim,in_dim), generator = g)

def make_mask(in_dim, out_dim, mask_num = 0, num_fixed = 6, mask_type='densest'):
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

        for nz in range(1, in_dim):
            for zeros_in_row in itertools.combinations(range(in_dim), nz):
                mask[row_idx, zeros_in_row] = 0
                row_idx += 1
                if row_idx >= out_dim:
                    return mask
    elif mask_type == 'bound_zeros':
        # other type of mask based on lower bounding sparsity to break symmetries more
        mask = torch.ones(out_dim, in_dim)
        least_zeros = num_fixed
        row_idx = 0
        for nz in range(least_zeros, in_dim):
            for zeros_in_row in itertools.combinations(range(in_dim), nz):
                mask[row_idx, zeros_in_row] = 0
                row_idx += 1
                if row_idx >= out_dim:
                    return mask

        raise ValueError('Error in making mask, possibly because out_dim is too large for these settings')

    elif mask_type == 'random_subsets':
            # other type of mask based on lower bounding sparsity to break symmetries more
            mask = torch.ones(out_dim, in_dim)
            row_idx = 0
            least_zeros = num_fixed
            for nz in range(least_zeros, in_dim):
                while True:

                    zeros_in_row = get_subset(in_dim, row_idx, least_zeros, mask_num)
                    mask[row_idx, zeros_in_row] = 0
                    row_idx += 1
                    if row_idx >= out_dim:
                        return mask

            raise ValueError('Error in making mask, possibly because out_dim is too large for these settings')
    else:
        raise ValueError('Invalid mask type')
