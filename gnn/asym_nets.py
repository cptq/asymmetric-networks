import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 1

class AsymNonlin(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.register_buffer('C', C)
        self.nonlin = nn.GELU()
        
    def forward(self, x):
        x = self.nonlin(x)
        x = torch.matmul(x, self.C)
        x = self.nonlin(x)
        return x

class AsymSwiGLU(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.register_buffer('C', C)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        gate = self.sigmoid(torch.matmul(x, self.C))
        x = gate * x
        return x


class SparseMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, mask_params):
        super().__init__()
        self.num_layers = num_layers
        self.lins = nn.ModuleList()
        self.lns = nn.ModuleList()
        assert num_layers >= 2
        self.lins.append(SparseLinear(in_dim, hidden_dim, **mask_params))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 2):
            self.lins.append(SparseLinear(hidden_dim, hidden_dim, **mask_params))
            self.lns.append(nn.LayerNorm(hidden_dim))
        self.lins.append(SparseLinear(hidden_dim, out_dim, **mask_params))

    def forward(self, x):
        for i in range(self.num_layers-1):
            x = self.lins[i](x)
            x = self.lns[i](x)
            x = F.relu(x)
        x = self.lins[-1](x)
        return x

class SparseLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, mask_type='densest', mask_constant=1, mask_num=0, num_fixed=6, do_normal_mask=True, mask_path=None):
        super().__init__()
        assert out_dim < 2**in_dim, 'out dim cannot be much higher than in dim'
        
        if mask_path is not None:
            mask, _ = torch.load(mask_path)
        else:
            mask = make_mask(in_dim, out_dim, mask_type=mask_type, num_fixed = num_fixed, mask_num = mask_num)

        self.register_buffer('mask', mask, persistent=True)
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))

        if mask_path is not None:
            _, n_mask = torch.load(mask_path)
            self.register_buffer('normal_mask', n_mask, persistent=True)
        else:
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
        return F.linear(x, (self.weight), self.bias)

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
    g.manual_seed(row_idx + abs(hash(str(mask_num) + str(seed))))
    indices = torch.arange(num_cols)
    return (indices[torch.randperm(num_cols, generator = g)[:num_sample]])

def normal_mask(out_dim, in_dim, mask_num):
    g = torch.Generator()
    g.manual_seed(abs(hash(str(mask_num)+ str(seed))))
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
        least_zeros = 2
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
