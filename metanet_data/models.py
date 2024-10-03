import math
import itertools
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from torch.nn import init

seed = 1

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, mask_num, mask_type='densest', do_normal_mask = True, num_fixed = 6, mask_constant=0, kernel_size = 3, stride = 1, padding = 1, bias=False, mask_path=None):
        super().__init__()
        assert 2**(in_channels * kernel_size **2) >= out_channels, "out dimension to big for asymmetry"

        if mask_path is not None:
            mask, _ = torch.load(mask_path)
        else:
            mask = make_conv_mask(in_channels, out_channels, kernel_size, mask_type=mask_type, num_fixed = num_fixed, mask_num = mask_num)

        self.register_buffer('mask', mask, persistent=True)

        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)))
        hook = self.weight.register_hook(lambda grad: self.mask*grad) # zeros out gradients for masked parts

        if mask_path is not None:
            _, n_mask = torch.load(mask_path)
            self.register_buffer('normal_mask', n_mask, persistent=True)
        else:
            if do_normal_mask:
                self.register_buffer('normal_mask', conv_normal_mask(out_channels, in_channels, kernel_size, mask_num), persistent=True)
            else:
                self.register_buffer('normal_mask', torch.ones(size = (out_channels, in_channels, kernel_size, kernel_size)), persistent=True) #torch.ones -> do nothing

        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter('bias', None)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.mask_constant = mask_constant
        self.mask_num = mask_num
        self.stride = stride
        self.padding = padding
        self.reset_parameters()

    def forward(self, x):
        self.weight.data = (self.weight.data*self.mask + (1-self.mask)*self.mask_constant*self.normal_mask)
        out = F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        return out

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = (self.weight.data* self.mask + (1-self.mask)*self.mask_constant*self.normal_mask)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    def __repr__(self):
        return f"SparseConv2d(in_channels = {self.in_channels}, out_channels = {self.out_channels}, kernel_size = {self.kernel_size}, stride = {self.stride})"
    def count_unused_params(self):
        return (1-self.mask.int()).sum().item()


class SparseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, mask_num, mask_params, stride=1, option='B'):
        super(SparseBasicBlock, self).__init__()

        self.conv1 = SparseConv2d(in_planes, planes,  mask_num, stride = stride, **mask_params['conv1'])
        self.ln1 = nn.GroupNorm(1, planes)
        
        self.conv2 = SparseConv2d(planes, planes,  mask_num + 1,stride = 1, **mask_params['conv2'])
        self.ln2 = nn.GroupNorm(1, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
               self.shortcut = nn.Sequential(
                     SparseConv2d(in_planes, self.expansion * planes, mask_num + 2,  **mask_params['skip'], kernel_size=1, stride=stride, padding = 0, bias=False,),
                     nn.GroupNorm(1, self.expansion * planes)
                )
                
    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))

        out = self.ln2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SparseResNet(nn.Module):
    def __init__(self, block, num_blocks, mask_params, w=1, num_classes=10):
        super(SparseResNet, self).__init__()
        self.in_planes = round(16*w)
        mask_num = 0
        self.conv1 = SparseConv2d(3, round(16*w), mask_num, kernel_size=3, stride=1, padding=1, bias=False, **mask_params['conv_f'])
        self.ln1 = nn.GroupNorm(1, round(16*w))

        mask_num += 1

        self.layer1 = self._make_layer(block, mask_num, round(16*w), num_blocks[0], stride=1, mask_params = mask_params['conv_1'])
        mask_num += num_blocks[0]*3

        self.layer2 = self._make_layer(block, mask_num, round(32*w), num_blocks[1], stride=2, mask_params = mask_params['conv_2'])
        mask_num += num_blocks[1]*3

        self.layer3 = self._make_layer(block, mask_num, round(64*w), num_blocks[2], stride=2, mask_params = mask_params['conv_3'])
        mask_num += num_blocks[2]*3

        self.linear = SparseLinear(round(64*w), num_classes, mask_num = mask_num, **mask_params['linear'])


    def _make_layer(self, block, mask_num, planes, num_blocks, stride, mask_params):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, mask_num, mask_params, stride))
            mask_num += 3
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    def count_unused_params(self):
        count = 0
        for node in self.modules():
            if hasattr(node, 'mask'):
                count += (1-node.mask).sum()
        return count.item()
    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def sparse_resnet8(mask_params, w=1):
    return SparseResNet(SparseBasicBlock, [1,1,1], mask_params, w=w)
def sparse_resnet14(mask_params, w=1):
    return SparseResNet(SparseBasicBlock, [2,2,2], mask_params, w=w)
def sparse_resnet20(mask_params, w=1):
    return SparseResNet(SparseBasicBlock, [3,3,3], mask_params, w=w)
def sparse_resnet32(mask_params):
    return SparseResNet(SparseBasicBlock, [5,5,5], mask_params)

    
def make_conv_mask(in_channels, out_channels, kernel_size, mask_num, mask_type='random_subsets', num_fixed=6):

    if mask_type == 'densest':
        mask = torch.ones(size = (out_channels, in_channels, kernel_size, kernel_size))
        weights_per_out_channel = in_channels * kernel_size**2
        flattened_to_3d_index = lambda ind : (ind // kernel_size**2,(ind//kernel_size)%kernel_size,ind%kernel_size)
        out_channel_idx = 1
        if out_channels == 1:
            return mask

        for nz in range(1, weights_per_out_channel):
            for zeros_in_out_channel in itertools.combinations(range(weights_per_out_channel), nz):

                for zero_ind in map(flattened_to_3d_index, zeros_in_out_channel):
                    mask[out_channel_idx][zero_ind] = 0
                out_channel_idx += 1
                if out_channel_idx >= out_channels:
                    return mask

    elif mask_type == 'bound_zeros':
        mask = torch.ones(size = (out_channels, in_channels, kernel_size, kernel_size))
        weights_per_out_channel = in_channels * kernel_size**2
        flattened_to_3d_index = lambda ind : (ind // kernel_size**2,(ind//kernel_size)%kernel_size,ind%kernel_size)
        out_channel_idx = 0
        least_zeros = num_fixed
        for nz in range(least_zeros, weights_per_out_channel):
            for zeros_in_out_channel in itertools.combinations(range(weights_per_out_channel), nz):

                for zero_ind in map(flattened_to_3d_index, zeros_in_out_channel):
                    mask[out_channel_idx][zero_ind] = 0
                out_channel_idx += 1
                if out_channel_idx >= out_channels:
                    return mask

    elif mask_type == 'random_subsets':
        mask = torch.ones(size = (out_channels, in_channels, kernel_size, kernel_size))
        weights_per_out_channel = in_channels * kernel_size**2
        least_zeros = num_fixed

        flattened_to_3d_index = lambda ind : (ind // kernel_size**2,(ind//kernel_size)%kernel_size,ind%kernel_size)
        for out_channel_idx in range(out_channels):
            zeros_in_out_channel = get_subset(weights_per_out_channel, out_channel_idx, least_zeros, mask_num)
            for zero_ind in map(flattened_to_3d_index, zeros_in_out_channel):
                mask[out_channel_idx][zero_ind] = 0
        return mask
    elif mask_type == 'filter_random_subsets':
        mask = torch.ones(size = (out_channels, in_channels, kernel_size, kernel_size))
        least_zeros = num_fixed
        
        for out_channel_idx in range(out_channels):
            zeros_in_out_channel = get_subset(in_channels, out_channel_idx, least_zeros, mask_num)
            for zero_ind in zeros_in_out_channel:
                mask[out_channel_idx, zero_ind, :, :] = 0
        return mask
    elif mask_type == 'none':
        return torch.ones(size = (out_channels, in_channels, kernel_size, kernel_size))

def conv_normal_mask(out_channels, in_channels, kernel_size, mask_num):
        g = torch.Generator()
        g.manual_seed(abs(hash(str(mask_num)+ str(seed))))
        return torch.randn(size=(out_channels, in_channels, kernel_size, kernel_size), generator = g)

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B', **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ln1 = nn.GroupNorm(1, planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.ln2 = nn.GroupNorm(1, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.GroupNorm(1, self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))

        out = self.ln2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class AsymConv2dSwiGLU(nn.Module):
    def __init__(self, planes, scale=1.0, mask_num=0, mask_params=None):
        super().__init__()
        if mask_params is None or "mask_path" not in mask_params:
            g = torch.Generator()
            g.manual_seed(abs(hash(str(mask_num)+ str(seed))))
            F_mat = scale * torch.randn(planes, planes, 1, 1, generator=g) / math.sqrt(planes)
        else:
            F_mat = torch.load(mask_params["mask_path"])
        self.register_buffer("F_mat", F_mat)
    def forward(self, x):
        gate = F.sigmoid(F.conv2d(x, self.F_mat))
        return gate * x

class SigmaBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B', mask_params={}, mask_num=0):
        super(SigmaBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ln1 = nn.GroupNorm(1, planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.ln2 = nn.GroupNorm(1, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.GroupNorm(1, self.expansion * planes)
                )
        self.nonlin1 = AsymConv2dSwiGLU(planes, mask_num=mask_num+1, mask_params=mask_params.get("conv1", None))
        self.nonlin2 = AsymConv2dSwiGLU(planes, mask_num=mask_num+2, mask_params=mask_params.get("conv2", None))

    def forward(self, x):
        out = self.ln1(self.conv1(x))
        out = self.nonlin1(out)

        out = self.ln2(self.conv2(out))
        out += self.shortcut(x)
        out = self.nonlin2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, w=1, num_classes=10, nonlin='relu', mask_params={}):
        super(ResNet, self).__init__()
        self.in_planes = round(16*w)

        self.conv1 = nn.Conv2d(3, round(16*w), kernel_size=3, stride=1, padding=1, bias=False)
        self.ln1 = nn.GroupNorm(1, round(16*w))

        self.layer1 = self._make_layer(block, round(16*w), num_blocks[0], stride=1, mask_params=mask_params.get("conv_1", {}), mask_num=1)
        self.layer2 = self._make_layer(block, round(32*w), num_blocks[1], stride=2, mask_params=mask_params.get("conv_2", {}), mask_num=3)
        self.layer3 = self._make_layer(block, round(64*w), num_blocks[2], stride=2, mask_params=mask_params.get("conv_3", {}), mask_num=5)
        self.linear = nn.Linear(round(64*w), num_classes)

        if nonlin == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlin == 'asym':
            self.nonlin = AsymConv2dSwiGLU(round(16*w), mask_num=0, mask_params=mask_params.get("conv_f", None))
        else:
            raise ValueError('Invalid nonlinearity')

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, mask_params={}, mask_num=0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, mask_params=mask_params, mask_num=mask_num))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.nonlin(self.ln1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def count_unused_params(self):
        return 0


def resnet8(w=1):
    return ResNet(BasicBlock, [1, 1, 1], w=w)
def resnet14(w=1):
    return ResNet(BasicBlock, [2, 2, 2], w=w)
def resnet20(w=1):
    return ResNet(BasicBlock, [3, 3, 3], w=w)
def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def sigma_asym_resnet8(w=1, mask_params={}):
    return ResNet(SigmaBasicBlock, [1, 1, 1], w=w, nonlin='asym', mask_params=mask_params)


if __name__ == '__main__':
    linear_mask_params = {'mask_constant' : 3, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': 15}
    conv_mask_params_f = {'mask_constant' : 15, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': 12}
    
    conv_mask_params_1_conv = {'mask_constant' :3, 'mask_type' : 'filter_random_subsets', 'do_normal_mask' : True, 'num_fixed': 4}
    conv_mask_params_1_skip = {'mask_constant' : 3, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': 4}
    conv_mask_params_1 = {'conv' : conv_mask_params_1_conv, 'skip':conv_mask_params_1_skip}
    
    conv_mask_params_2_conv = {'mask_constant' : 3, 'mask_type' : 'filter_random_subsets', 'do_normal_mask' : True, 'num_fixed':6}
    conv_mask_params_2_skip = {'mask_constant' : 3, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': 6}
    conv_mask_params_2 = {'conv' : conv_mask_params_2_conv, 'skip':conv_mask_params_2_skip}
    
    conv_mask_params_3_conv = {'mask_constant' : 3, 'mask_type' : 'filter_random_subsets', 'do_normal_mask' : True, 'num_fixed': 8}
    conv_mask_params_3_skip = {'mask_constant' : 3, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': 8}
    conv_mask_params_3 = {'conv' : conv_mask_params_3_conv, 'skip':conv_mask_params_3_skip}
    mask_params = {
           'linear' : linear_mask_params, 
           'conv_f' : conv_mask_params_f, 
           'conv_1': conv_mask_params_1,
           'conv_2': conv_mask_params_2,
           'conv_3': conv_mask_params_3
          }
    
    MAKE_NEW_MASKS = False
    if MAKE_NEW_MASKS:
        # save fixed masks 
        in_dim, out_dim = 64, 10
        num_fixed = 8
        mask_num = 0
        mask_constant = 2.0
        mask_type = 'filter_random_subsets'
        path = 'new_masks'

        # linear
        mask = make_mask(in_dim, out_dim, mask_type='random_subsets', num_fixed=num_fixed, mask_num=mask_num)
        n_mask = normal_mask(out_dim, in_dim, mask_num=mask_num)
        torch.save((mask, n_mask), f'{path}/sparse_resnet8_linear.pt')
        mask_num += 1

        # conv_f
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        mask = make_conv_mask(in_channels, out_channels, kernel_size, mask_type=mask_type, num_fixed=num_fixed, mask_num=mask_num)
        n_mask = conv_normal_mask(out_channels, in_channels, kernel_size, mask_num)
        torch.save((mask, n_mask), f'{path}/sparse_resnet8_conv_f.pt')
        mask_num += 1
        
        # conv_1
        in_channels = 16
        out_channels = 16
        kernel_size = 3
        mask = make_conv_mask(in_channels, out_channels, kernel_size, mask_type=mask_type, num_fixed=num_fixed, mask_num=mask_num)
        n_mask = conv_normal_mask(out_channels, in_channels, kernel_size, mask_num)
        torch.save((mask, n_mask), f'{path}/sparse_resnet8_block1_conv1.pt')
        mask_num += 1

        mask = make_conv_mask(out_channels, out_channels, kernel_size, mask_type=mask_type, num_fixed=num_fixed, mask_num=mask_num)
        n_mask = conv_normal_mask(out_channels, out_channels, kernel_size, mask_num)
        torch.save((mask, n_mask), f'{path}/sparse_resnet8_block1_conv2.pt')
        mask_num += 1

        mask = make_conv_mask(in_channels, out_channels, 1, mask_type=mask_type, num_fixed=num_fixed, mask_num=mask_num)
        n_mask = conv_normal_mask(out_channels, in_channels, 1, mask_num)
        torch.save((mask, n_mask), f'{path}/sparse_resnet8_block1_skip.pt')
        mask_num += 1


        # conv_2
        in_channels = 16
        out_channels = 32
        kernel_size = 3
        mask = make_conv_mask(in_channels, out_channels, kernel_size, mask_type=mask_type, num_fixed=num_fixed, mask_num=mask_num)
        n_mask = conv_normal_mask(out_channels, in_channels, kernel_size, mask_num)
        torch.save((mask, n_mask), f'{path}/sparse_resnet8_block2_conv1.pt')
        mask_num += 1

        mask = make_conv_mask(out_channels, out_channels, kernel_size, mask_type=mask_type, num_fixed=num_fixed, mask_num=mask_num)
        n_mask = conv_normal_mask(out_channels, out_channels, kernel_size, mask_num)
        torch.save((mask, n_mask), f'{path}/sparse_resnet8_block2_conv2.pt')
        mask_num += 1


        mask = make_conv_mask(in_channels, out_channels, 1, mask_type=mask_type, num_fixed=num_fixed, mask_num=mask_num)
        n_mask = conv_normal_mask(out_channels, in_channels, 1, mask_num)
        torch.save((mask, n_mask), f'{path}/sparse_resnet8_block2_skip.pt')
        mask_num += 1

        # conv_3
        in_channels = 32
        out_channels = 64
        kernel_size = 3
        mask = make_conv_mask(in_channels, out_channels, kernel_size, mask_type=mask_type, num_fixed=num_fixed, mask_num=mask_num)
        n_mask = conv_normal_mask(out_channels, in_channels, kernel_size, mask_num)
        torch.save((mask, n_mask), f'{path}/sparse_resnet8_block3_conv1.pt')
        mask_num += 1

        mask = make_conv_mask(out_channels, out_channels, kernel_size, mask_type=mask_type, num_fixed=num_fixed, mask_num=mask_num)
        n_mask = conv_normal_mask(out_channels, out_channels, kernel_size, mask_num)
        torch.save((mask, n_mask), f'{path}/sparse_resnet8_block3_conv2.pt')
        mask_num += 1

        mask = make_conv_mask(in_channels, out_channels, 1, mask_type=mask_type, num_fixed=num_fixed, mask_num=mask_num)
        n_mask = conv_normal_mask(out_channels, in_channels, 1, mask_num)
        torch.save((mask, n_mask), f'{path}/sparse_resnet8_block3_skip.pt')
        mask_num += 1

    mask_params_saved = {
           'linear' : {"mask_path": f"{path}/sparse_resnet8_linear.pt", "mask_constant": mask_constant}, 
           'conv_f' : {"mask_path": f"{path}/sparse_resnet8_conv_f.pt", "mask_constant": mask_constant},
           'conv_1': {"conv1": {"mask_path": f"{path}/sparse_resnet8_block1_conv1.pt", "mask_constant": mask_constant}, "conv2": {"mask_path": f"{path}/sparse_resnet8_block1_conv2.pt", "mask_constant": mask_constant}, "skip": {"mask_path": f"{path}/sparse_resnet8_block1_skip.pt", "mask_constant": mask_constant}},
           'conv_2': {"conv1": {"mask_path": f"{path}sparse_resnet8_block2_conv1.pt", "mask_constant": mask_constant}, "conv2": {"mask_path": f"{path}/sparse_resnet8_block2_conv2.pt", "mask_constant": mask_constant}, "skip": {"mask_path": f"{path}/sparse_resnet8_block2_skip.pt", "mask_constant": mask_constant}},
           'conv_3': {"conv1": {"mask_path": f"{path}sparse_resnet8_block3_conv1.pt", "mask_constant": mask_constant}, "conv2": {"mask_path": f"{path}/sparse_resnet8_block3_conv2.pt", "mask_constant": mask_constant}, "skip": {"mask_path": f"{path}/sparse_resnet8_block3_skip.pt", "mask_constant": mask_constant}},
          }
    model = sparse_resnet8(mask_params_saved)
    model(torch.randn(2, 3, 32, 32))

    mask_params = {"conv_f": {"mask_path": f"{path}/sigma_asym_resnet8_conv_f.pt"},
                   "conv_1": {"conv1": {"mask_path": f"{path}/sigma_asym_resnet8_block1_conv1.pt"},
                              "conv2": {"mask_path": f"{path}/sigma_asym_resnet8_block1_conv2.pt"},
                             },
                   "conv_2": {"conv1": {"mask_path": f"{path}/sigma_asym_resnet8_block2_conv1.pt"},
                              "conv2": {"mask_path": f"{path}/sigma_asym_resnet8_block2_conv2.pt"},
                             },
                   "conv_3": {"conv1": {"mask_path": f"{path}/sigma_asym_resnet8_block3_conv1.pt"},
                              "conv2": {"mask_path": f"{path}/sigma_asym_resnet8_block3_conv2.pt"},
                             }
                  }
    model = sigma_asym_resnet8(mask_params=mask_params)

