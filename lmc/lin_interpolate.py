''' 
Built off of: https://github.com/themrzmaster/git-re-basin-pytorch/tree/main 
MIT License
'''
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

def main():
  model1 = -1
  model2 = -1
  model_interp = -1
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
    test_loss, test_acc = test(model_interp)
    results.append((test_loss, test_acc))

def lerp_sd(lam, sd1, sd2):
  sd3 = copy.deepcopy(sd1)
  for name in sd1:
    sd3[name] = (1-lam)*sd1[name] + lam*sd2[name]
  return sd3

def dist_sd(sd1, sd2):
  sqdist = torch.tensor([0.], dtype=torch.float)
  for name in sd1:
    sqdist += (sd1[name] - sd2[name]).float().square().sum()
  dist = sqdist.sqrt()
  return dist

def get_num_params(model):
  num_params = sum([p.numel() for p in model.parameters()])
  return num_params
  
  
if __name__ == '__main__':
  pass