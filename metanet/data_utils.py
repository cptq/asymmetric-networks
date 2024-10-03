import re
import numpy as np
import pandas as pd
import random
import os
import csv
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class NetDataset(torch.utils.data.Dataset):
    " built off of Graph Metanetworks codebase "
    def __init__(self, root='/net_data/', model_type="sparse_resnet8", target='test_acc', return_hparams=False):
        self.model_type = model_type
        self.root = root
        root = root + f'{model_type}/'
        self.paths = [root + name for name in os.listdir(root)]
        all_hparams = []
        with open('metanet_data/grid_runs/hparams.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                if i == 0:
                    assert row[0] == 'lr'
                    assert row[1] == 'weight_decay'
                    assert row[2] == 'label_smoothing'
                    assert row[3] == 'epochs'
                else:
                    lr = float(row[0])
                    wd = float(row[1])
                    label_smoothing = float(row[2])
                    epochs = int(row[3])
                    all_hparams.append([lr, wd, label_smoothing, epochs])

        self._prune_nans()

        self.hparams = []
        for idx in range(len(self.paths)):
            orig_idx = self._idx_to_orig(idx)
            self.hparams.append(all_hparams[orig_idx])

        self.target = target
        self.return_hparams = return_hparams
        del all_hparams

    def _prune_nans(self):
        # if path exists, load from there
        print("Keeping non-nan paths")
        nan_path = self.root + f'{self.model_type}_no_nan'
        new_paths = []
        if os.path.exists(nan_path):
            with open(nan_path, "rb") as f:
                new_paths = pickle.load(f)
        else:
            # if not, determine paths
            print("Computing non-nans")
            for idx, path in enumerate(self.paths):
                if idx % 500 == 0: print(idx)
                sd, _ = torch.load(path)
                isnan = any(v.isnan().any() for v in sd.values())
                if not isnan:
                    new_paths.append(path)
            with open(nan_path, "wb") as f:
                pickle.dump(new_paths, f)
        self.paths = new_paths
                

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sd, test_acc = torch.load(self.paths[idx])
        test_acc = test_acc / 100
        hparams = self.hparams[idx]
        if self.target == 'test_acc':
            y = test_acc
        elif self.target == 'lr':
            y = hparams[0]
        elif self.target == 'weight_decay':
            y = hparams[1]
        elif self.target == 'label_smoothing':
            y = hparams[2]
        else:
            raise ValueError('Invalid target')
        if not self.return_hparams:
            return sd, y
        else:
            return sd, y, hparams

    def _idx_to_orig(self, idx):
        orig_idx = int(re.search(r'\d+$', self.paths[idx].rstrip('.pt')).group())
        return orig_idx


if __name__ == '__main__':
    def sd_to_input(sd, bs, device, use_fixed=False):
        """
            use_fixed determines whether untrained entries of sparse networks are used
        """
        sparse = 'linear.mask' in sd
        if not sparse:
            weights = torch.cat([v.reshape(bs, -1) for v in sd.values()], 1)
            return weights
        else:
            weights = []
            for (k,v) in sd.items():
                v = v.to(device)
                base_name = k[:k.rfind('.')] # e.g. for layer3.0.ln1.weight, this would be layer3.0.ln1
                is_weight = k[k.rfind('.')+1:] == 'weight'
                if base_name + '.mask' in sd and is_weight:
                    mask = sd[base_name+'.mask'].to(device)
                    n_mask = sd[base_name+'.normal_mask'].to(device)
                    fixed_idx = torch.where(mask == 0.0)
                    v[fixed_idx] = n_mask[fixed_idx]

                if 'mask' not in k:
                    if use_fixed:
                        weights.append(v)
                    else:
                        if base_name + '.mask' in sd and is_weight:
                            mask = sd[base_name+'.mask'].to(device)
                            weights.append(v[torch.where(mask==1.0)])
                        else:
                            weights.append(v)
            return torch.cat([w.reshape(bs, -1) for w in weights], 1)

    model_type = 'resnet8'
    net_dataset = NetDataset(model_type=model_type, return_hparams=True)

    # for computing some statistics of the networks in the dataset
    num_params = []
    test_accs = []
    hparams = []
    params = []
    device = torch.device('cuda')
    for i in range(len(net_dataset)):
        if i % 1000 == 0:
            print("Network number:", i)
        data = net_dataset[i]
        sd = data[0]
        num_params.append(sum([v.numel() for v in sd.values()]))
        test_accs.append(data[1])
        param_vec = sd_to_input(sd, 1, device, use_fixed=False).cpu()
        params.append(param_vec)
        hparams.append(data[2])
    
    plt.hist(test_accs, bins=30)
    plt.savefig(f'test_accs_hist_{model_type}.png')

    all_params = torch.cat(params, 0).numpy()
    all_test_accs = np.array(test_accs)
    all_hparams = np.array(hparams)
    np.savez(f'/net_data/{model_type}_data.npz', params=all_params, test_accs=all_test_accs, hparams=all_hparams)
    
