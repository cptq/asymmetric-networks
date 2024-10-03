import math
import itertools
import copy
import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
import random
import numpy as np


from itertools import permutations

def interpolate_test(model1, model2, test_fn):
    model_interp = copy.deepcopy(model2)
    lambdas = torch.linspace(0, 1, steps=25)
    sd1 = copy.deepcopy(model1.state_dict())
    sd2 = copy.deepcopy(model2.state_dict())
    dist = dist_sd(sd1, sd2)
    num_params = get_num_params(model1)
    print('Dist / params:', dist / num_params)
    results = []
    for lam in lambdas:
        sd3 = lerp_sd(lam, sd1, sd2)
        model_interp.load_state_dict(sd3)
        results.append(test_fn(model_interp))
    return results


def lerp_sd(lam, sd1, sd2):
    sd3 = copy.deepcopy(sd1)
    for name in sd1:
        sd3[name] = (1-lam)*sd1[name] + lam*sd2[name]
    return sd3

def dist_sd(sd1, sd2, device=torch.device('cuda')):
    sqdist = torch.tensor([0.], dtype=torch.float, device=device)
    for name in sd1:
        sqdist += (sd1[name] - sd2[name]).float().square().sum()
        dist = sqdist.sqrt()
    return dist

def get_num_params(model):
    num_params = sum([p.numel() for p in model.parameters()])
    if hasattr(model, 'count_unused_params'):
        num_params = num_params - model.count_unused_params()
    return num_params