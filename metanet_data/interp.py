import copy
import numpy as np
import torch

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


def interpolate_test(model1, model2, test_fn):
    """
        test_fn maps model to test accuracy
    """
    model_interp = copy.deepcopy(model2)
    lambdas = torch.linspace(0, 1, steps=25)
    sd1 = copy.deepcopy(model1.state_dict())
    sd2 = copy.deepcopy(model2.state_dict())
    dist = dist_sd(sd1, sd2)
    num_params = get_num_params(model1)
    print(f'Dist / params: {(dist / num_params).item():.7f}')
    results = []
    for lam in lambdas:
        sd3 = lerp_sd(lam, sd1, sd2)
        model_interp.load_state_dict(sd3)
        results.append(test_fn(model_interp))
    return results

def mli_stats(results):
    """
        results is a 1d array of some train/test metric across interpolation values
    """
    results = np.array(results)
    diffs = results[1:] - results[:-1]
    max_diff = np.max(diffs)
    
    # convexity measures
    h = 1 / (results.shape[0]-1)
    second_derivatives = []
    for i in range(1, results.shape[0]-1):
        finite_diff = results[i+1] - 2*results[i] + results[i-1]
        finite_diff = finite_diff / h**2
        second_derivatives.append(finite_diff)
    second_derivatives = np.array(second_derivatives)
    convexity1 = np.min(second_derivatives)
    convexity2 = np.mean(second_derivatives >= 0)
    # proportion of points lying below the line segment between 0 and 1
    xs = np.linspace(0, 1, results.shape[0])
    line = results[0] + (results[1] - results[0]) * xs
    below_line = results <= line
    convexity3 = np.mean(below_line[1:-1])
    
    return max_diff, convexity1, convexity2, convexity3
