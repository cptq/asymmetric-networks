import copy
import torch
import torch.nn as nn

def reset_bn_stats(model, data):
    """ from github.com/KellerJordan/REPAIR """
    for m in model.modules():
        if type(m) == nn.BatchNorm1d:
            m.momentum = None # use simple average
            m.reset_running_stats()
    model.train()
    with torch.no_grad():
        for _ in range(50):
            _ = model(data.x, data.adj_t)

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


def interpolate_test(model1, model2, test_fn, steps=25, rewarm=False, data=None):
    """
        test_fn maps model to test accuracy
    """
    model_interp = copy.deepcopy(model2)
    lambdas = torch.linspace(0, 1, steps=steps)
    sd1 = copy.deepcopy(model1.state_dict())
    sd2 = copy.deepcopy(model2.state_dict())
    device = next(iter(sd1.values())).device
    dist = dist_sd(sd1, sd2, device=device)
    num_params = get_num_params(model1)
    print(f'Dist / params: {(dist / num_params).item():.7f}')

    results = []
    for lam in lambdas:
        sd3 = lerp_sd(lam, sd1, sd2)
        model_interp.load_state_dict(sd3)
        if rewarm:
            print("Resetting batch norm stats")
            reset_bn_stats(model_interp, data)
        results.append(test_fn(model_interp))
    return results
