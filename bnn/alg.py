import math
import types

import torch
from torch import nn
from torch.nn import functional as F
from sparse import *
def init_diag(self,
              config,
              fixed_variance=False,
              param_prior=False):
    if not fixed_variance:
        def register(n, x):
            return self.register_parameter(n, nn.Parameter(x))
    else:
        register = self.register_buffer

    init_val = config.q_init_logvar
    init_log_w = init_val * torch.ones_like(self.weight)
    register('_diag_w', init_log_w)

    if param_prior:
        init_pw = self.weight.std() * torch.randn_like(self.weight)
        register('_mean_pw', init_pw)

    if self.bias is not None:
        init_log_b = init_val * torch.ones_like(self.bias)
        register('_diag_b', init_log_b)

        if param_prior:
            init_pb = self.bias.std() * torch.randn_like(self.bias)
            register('_mean_pb', init_pb)

def set_prior_logvar(module, config):
    out_dim = module.weight.shape[0] \
        if isinstance(module, (nn.Linear, SparseLinear)) \
        else module.weight.shape[1]

    if not config.scale_prior:
        out_dim = 1

    module.p_logvar = -math.log(config.prior_precision * out_dim)


def get_variational_step(opt, num_points, num_variational_samples=1):
    def step(self, x, y, loss_func):
        opt.zero_grad()
        l = 0
        with torch.enable_grad():
            for _ in range(num_variational_samples):
                l += loss_func(self(x), y)
            l /= num_variational_samples
            kl = sum(m.kl()
                     for m in self.modules()
                     if hasattr(m, "kl"))
            neg_elbo = l + 1 / num_points * kl
        neg_elbo.backward()
        opt.step()

        return {'elbo': neg_elbo.neg().item(),
                'kl/n': kl.item() / num_points,
                'neg_data_term': l.item()}

    return step


def lrt(self, input_, mean):
    (x,) = input_

    var_w = self._diag_w.exp()
    var_b = self._diag_b.exp() \
        if self.bias is not None \
        else None

    if isinstance(self, nn.Linear):
        mf_var = F.linear(x.pow(2),
                          var_w,
                          var_b)
    elif isinstance(self, SparseLinear):
        mf_var = F.linear(x.pow(2),
                          var_w * self.mask,
                          var_b)
    elif isinstance(self, nn.Conv2d):
        mf_var = F.conv2d(x.pow(2),
                          var_w,
                          var_b,
                          self.stride,
                          self.padding,
                          self.dilation,
                          self.groups)
        mf_var = mf_var.clamp(min=1e-16)
    elif isinstance(self, SparseConv2d):
        mf_var = F.conv2d(x.pow(2),
                          var_w*self.mask,
                          var_b,
                          self.stride,
                          self.padding,
                          self.dilation,
                          self.groups)
        mf_var = mf_var.clamp(min=1e-16)
    else:
        raise NotImplementedError()

    mf_noise = mf_var.sqrt() * torch.randn_like(mf_var,
                                                requires_grad=False)

    return mean + mf_noise


def rank_kl(mean,
            v,
            diag,
            prior_var,
            alpha):
    rank, d = v.shape
    m = torch.eye(rank, device=v.device) + alpha * (v / diag) @ v.t()

    sum_diag = diag.sum() \
        if not isinstance(diag, float) \
        else d * diag
    sum_log_diag = diag.log().sum() \
        if not isinstance(diag, float) \
        else d * math.log(diag)

    terms = [
        1 / prior_var * sum_diag - sum_log_diag,
        alpha / prior_var * v.pow(2).sum(),
        -torch.logdet(m),
        1 / prior_var * mean.pow(2).sum(),
        d * (math.log(prior_var) - 1),
    ]

    return 0.5 * sum(terms)


def elrg(model,
         config,
         num_points):
    def modify(module):
        if isinstance(module, (nn.Linear, nn.Conv2d, SparseLinear, SparseConv2d)):
            
            set_prior_logvar(module, config)
            module.rank = config.rank

            module.learn_diag = config.learn_diag
            module.q_init_logvar = config.q_init_logvar
            module.alpha = (1 / config.rank) if config.rank > 0 else 0.

            w = config.q_init_logvar * torch.ones_like(module.weight)

            if module.learn_diag:
                module.register_parameter('_diag_w', nn.Parameter(w))
                if module.bias is not None:
                    b = config.q_init_logvar * torch.ones_like(module.bias)
                    module.register_parameter('_diag_b', nn.Parameter(b))

            else:
                module.register_buffer('_diag_w', w)
                if module.bias is not None:
                    b = config.q_init_logvar * torch.ones_like(module.bias)
                    module.register_buffer('_diag_b', b)

            if config.rank > 0:

                if isinstance(module, (nn.Linear,SparseLinear)):

                    init_w_std = module.weight.std().item() / math.sqrt(config.rank)
                    init_v_w = 0.5 * init_w_std * torch.randn(config.rank,
                                                              module.weight.shape[1],
                                                              module.weight.shape[0],
                                                              device=module.weight.device)
                    module.register_parameter('_v_w',
                                              nn.Parameter(init_v_w))

                    if module.bias is not None:
                        init_v_b = 0.01 * torch.randn(config.rank,
                                                      1,
                                                      module.bias.shape[0],
                                                      device=module.bias.device)
                        module.register_parameter('_v_b',
                                                  nn.Parameter(init_v_b))
                else:
                    layer = nn.Conv2d(config.rank * module.in_channels,
                                      config.rank * module.out_channels,
                                      module.kernel_size,
                                      stride=module.stride,
                                      padding=module.padding,
                                      dilation=module.dilation,
                                      groups=config.rank,
                                      bias=module.bias is not None,
                                      padding_mode=module.padding_mode,
                                      device=module.weight.device
                                      )
                    layer.weight.data *= 0
                    if layer.bias is not None:
                        layer.bias.data *= 0
                    module.v = nn.ModuleList([layer])

            def hook(self, input_, mean):
                # Shapes: i - input, o - output, b - batch, r - rank

                (x,) = input_
                bs = x.size(0)

                mf_noise = lrt(self, input_, mean)

                if config.rank == 0:
                    return mf_noise

                if isinstance(self, nn.Linear):
                    out = torch.bmm(x[None].repeat(self.rank, 1, 1), self._v_w) + self._v_b  # r, b, o
                    nout = out * torch.randn(self.rank,
                                             bs,
                                             1,
                                             device=x.device,
                                             dtype=x.dtype,
                                             requires_grad=False)
                    lr_noise = math.sqrt(module.alpha) * nout.sum(0)  # b, o
                elif isinstance(self, SparseLinear):
                    for i in range(len(self._v_w)):
                        self._v_w.data[i] = (self._v_w.data[i]* self.mask.T ) 
                    out = torch.bmm(x[None].repeat(self.rank, 1, 1), self._v_w) + self._v_b  # r, b, o
                    
                    nout = out * torch.randn(self.rank,
                                             bs,
                                             1,
                                             device=x.device,
                                             dtype=x.dtype,
                                             requires_grad=False)
                    lr_noise = math.sqrt(module.alpha) * nout.sum(0)  # b, o`
                elif isinstance(self, nn.Conv2d):
                    out = self.v[0](x.repeat(1, config.rank, *[1] * (x.ndim - 2)))  # b, c_out*r, h, w

                    _, cmulg, h, w = out.shape
                    nout = out.view(bs, cmulg // config.rank, config.rank, h, w) * torch.randn(bs,
                                                                                               1,
                                                                                               config.rank, 1, 1,
                                                                                               device=x.device,
                                                                                               dtype=x.dtype,
                                                                                               requires_grad=False)
                    lr_noise = math.sqrt(module.alpha) * nout.sum(2)
                elif isinstance(self, SparseConv2d):
                    mask = self.mask
                    for i in range(0, self.v[0].out_channels, self.out_channels):
                        self.v[0].weight.data[i:i+self.out_channels] = (self.v[0].weight.data[i:i+self.out_channels]*self.mask)
                    out = self.v[0](x.repeat(1, config.rank, *[1] * (x.ndim - 2)))  # b, c_out*r, h, w

                    _, cmulg, h, w = out.shape
                    nout = out.view(bs, cmulg // config.rank, config.rank, h, w) * torch.randn(bs,
                                                                                               1,
                                                                                               config.rank, 1, 1,
                                                                                               device=x.device,
                                                                                               dtype=x.dtype,
                                                                                               requires_grad=False)
                    lr_noise = math.sqrt(module.alpha) * nout.sum(2)
                
                x = lr_noise + mf_noise
                return x
            def kl(self):
                prior_var = math.exp(self.p_logvar)

                if self.rank == 0:
                    kl = diag_gauss_kl(self.weight,
                                       self._diag_w.exp(),
                                       0,
                                       prior_var)
                    if self.bias is not None:
                        kl += diag_gauss_kl(self.bias,
                                            self._diag_b.exp(),
                                            0,
                                            prior_var)

                    return kl
                else:
                    mean_w = (self.weight).view(1, -1)
                    if isinstance(self, (nn.Linear, SparseLinear)):
                        v_w = self._v_w.view(self.rank, -1)
                    else:
                        v_w = self.v[0].weight.view(self.rank, -1)

                    if self.bias is not None:
                        mean_b = self.bias.view(1, -1)
                        if isinstance(self, (nn.Linear, SparseLinear)):
                            v_b = self._v_b.view(self.rank, -1)
                        else:
                            v_b = self.v[0].bias.view(self.rank, -1)
                        mean = torch.cat([mean_w, mean_b], -1)
                        v = torch.cat([v_w, v_b], -1)
                    else:
                        mean = mean_w
                        v = v_w

                    if self.learn_diag:
                        diag_w = self._diag_w.exp().view(1, -1)
                        if self.bias is None:
                            diag = diag_w
                        else:
                            diag_b = self._diag_b.exp().view(1, -1)
                            diag = torch.cat([diag_w, diag_b], -1)
                    else:
                        diag = math.exp(self.q_init_logvar)

                    return rank_kl(mean,
                                   v,
                                   diag,
                                   prior_var,
                                   alpha=module.alpha)

            module.register_forward_hook(hook)
            module.kl = types.MethodType(kl, module)

    model.inference_type = 'VI'
    model.num_test_samples = config.num_test_samples
    model.apply(modify)
    opt = torch.optim.Adam(model.parameters(),
                           lr=config.lr)
    model.step = types.MethodType(get_variational_step(opt, num_points),
                                  model)

    return model
