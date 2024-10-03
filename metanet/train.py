import functools
import argparse
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data_utils import NetDataset
from models import DMCCNN, DSSeqReadout

def sd_to_input(sd, bs, device, use_fixed=False):
    """
        use_fixed determines whether untrained entries of asym networks are used
    """
    is_sigma_asym = 'nonlin.F_mat' in sd
    asym = 'linear.mask' in sd or is_sigma_asym
    if not asym:
        weights = torch.cat([v.reshape(bs, -1) for v in sd.values()], 1)
    elif is_sigma_asym:
        weights = torch.cat([v.reshape(bs, -1) for (k,v) in sd.items() if "F_mat" not in k], 1)
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
        weights = torch.cat([w.reshape(bs, -1) for w in weights], 1)
    return weights

def vector_to_stats(v):
    ''' 
        v is B x d
        output is B x 7
    '''
    mean = v.mean(1, keepdim=True)
    var = v.var(1, keepdim=True)
    min_val = v.min(1, keepdim=True).values
    max_val = v.max(1, keepdim=True).values
    q25 = v.quantile(.25, dim=1, keepdim=True)
    q50 = v.quantile(.50, dim=1, keepdim=True)
    q75 = v.quantile(.75, dim=1, keepdim=True)
    return torch.cat([mean, var, min_val, max_val, q25, q50, q75], 1)

def sd_to_stats(sd, bs, device, use_fixed=False):
    is_sigma_asym = 'nonlin.F_mat' in sd
    asym = 'linear.mask' in sd or is_sigma_asym
    stats = []
    if not asym:
        for (k,v) in sd.items():
            v = v.to(device)
            v = v.reshape(bs, -1)
            stats.append(vector_to_stats(v))
    elif is_sigma_asym:
        for (k,v) in sd.items():
            if "F_mat" in k:
                continue
            v = v.to(device)
            v = v.reshape(bs, -1)
            stats.append(vector_to_stats(v))
    else:
        for (k,v) in sd.items():
            v = v.to(device)
            base_name = k[:k.rfind('.')] # e.g. for layer3.0.ln1.weight, this would be layer3.0.ln1
            is_weight = k[k.rfind('.')+1:] == 'weight'
            # change the fixed idx to what the normal mask prescribes
            if base_name + '.mask' in sd and is_weight:
                mask = sd[base_name+'.mask'].to(device)
                n_mask = sd[base_name+'.normal_mask'].to(device)
                fixed_idx = torch.where(mask == 0.0)
                v[fixed_idx] = n_mask[fixed_idx]

            if 'mask' not in k:
                if use_fixed:
                    stats.append(vector_to_stats(v.reshape(bs, -1)))
                else:
                    if base_name + '.mask' in sd and is_weight:
                        mask = sd[base_name+'.mask'].to(device)
                        v_use = v[torch.where(mask==1.0)]
                        if v_use.numel() == 0:
                            empty_stats = torch.zeros(bs, 7, device=device)
                            stats.append(empty_stats)
                        else:
                            v_use = v_use.reshape(bs, -1)
                            stats.append(vector_to_stats(v_use))
                    else:
                        stats.append(vector_to_stats(v))

    return torch.cat(stats, 1)

def train(model, train_loader, criterion, optimizer, device, weight_processor):
    model.train()
    total_loss = 0.0
    for sd, y in train_loader:
        bs = y.shape[0]
        weights = weight_processor(sd, bs, device)
        weights, y = weights.to(device), y.to(device).float()
        out = model(weights)
        loss = criterion(out.squeeze(1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * bs
    total_loss = total_loss / len(train_loader)
    return total_loss
    

@torch.no_grad()
def evaluate(model, loader, criterion, device, weight_processor):
    model.eval()
    total_loss = 0.0
    preds = []
    truth = []
    for sd, y in loader:
        bs = y.shape[0]
        weights = weight_processor(sd, bs, device)
        weights, y = weights.to(device), y.to(device).float()
        out = model(weights).squeeze(1)
        loss = criterion(out, y)
        total_loss += loss.item() * bs
        preds.append(out.detach().cpu().numpy())
        truth.append(y.detach().cpu().numpy())
    preds, truth = np.concatenate(preds), np.concatenate(truth)
    rsq = r2_score(truth, preds)
    tau = kendalltau(truth, preds).correlation
    total_loss = total_loss / len(loader)
    return total_loss, rsq, tau

def main(args):
    device = torch.device('cuda')
    dataset = NetDataset(model_type=args.dataset, target=args.target)
    n = min(len(dataset), args.max_size)
    dataset = Subset(dataset, range(n))
    print("Num examples:", len(dataset))
    train_prop = .5
    val_prop = .25
    test_prop = .25
    train_num = round(train_prop * n)
    val_num = round(val_prop * n)
    train_dataset = Subset(dataset, range(train_num))
    val_dataset = Subset(dataset, range(train_num, train_num+val_num))
    test_dataset = Subset(dataset, range(train_num+val_num, n))

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    if args.stats_input:
        in_dim = 203 # 29 param tensors * 7 stats per tensor
        weight_processor = sd_to_stats
    elif args.dataset == "smaller_resnet8":
        in_dim = 59944
        weight_processor = functools.partial(sd_to_input, use_fixed=False)
    elif args.dataset == "sigma_asym_resnet8":
        in_dim = 78042
        weight_processor = functools.partial(sd_to_input, use_fixed=False)
    elif args.use_fixed:
        in_dim = 78042 # number of parameters in resnet8
        weight_processor = functools.partial(sd_to_input, use_fixed=True)
    elif args.dataset == "sparse_resnet8":
        in_dim = 60634
        weight_processor = functools.partial(sd_to_input, use_fixed=False)

        
    if args.model == 'linear':
        model = nn.Linear(in_dim, 1)
    elif args.model == 'mlp':
        model = nn.Sequential(nn.Linear(in_dim, args.hidden_dim), nn.ReLU(), nn.LayerNorm(args.hidden_dim), nn.Linear(args.hidden_dim, 1))
    elif args.model == 'dmc':
        model = DMCCNN(1, args.hidden_dim, 1, conv_num_layers=10)
    elif args.model == 'deepsets':
        model = DSSeqReadout(1, args.hidden_dim, 1, num_layers=3)
    elif args.model == 'statnn':
        model = nn.Sequential(nn.Linear(in_dim, args.hidden_dim), nn.ReLU(), nn.LayerNorm(args.hidden_dim), nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU(), nn.LayerNorm(args.hidden_dim), nn.Linear(args.hidden_dim, 1))
    else:
        raise ValueError("Invalid model type")
    model = model.to(device)
    print(model)
    print("Num metanet params:", sum(p.numel() for p in model.parameters()))
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_rsq = -float('inf')
    best_epoch = None
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device, weight_processor)
        val_loss, rsq, tau = evaluate(model, val_loader, criterion, device, weight_processor)
        print(f"Epoch {epoch} | Train {train_loss:.5f} | Val {val_loss:.5f}, R^2 {rsq:.5f}, Tau {tau:.5f}")
        if rsq > best_rsq:
            test_loss, test_rsq, test_tau = evaluate(model, test_loader, criterion, device, weight_processor)
            best_rsq = rsq
            best_epoch = epoch
    print(f"Best Val Epoch: {best_epoch} | Best Val R^2: {best_rsq:.5f}")
    print(f"Test loss {test_loss:.5f} | Test R^2 {test_rsq:.5f} | Test Tau {test_tau:.5f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Metanet training')
    parser.add_argument('--bs', type=int, default=128,)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--dataset', type=str, default='resnet8', choices=['resnet8', 'sparse_resnet8', 'smaller_resnet8', 'sigma_asym_resnet8'])
    parser.add_argument('--stats_input', type=int, default=0, help='whether to take weight statistics as input instead of raw parameters')
    parser.add_argument('--use_fixed', type=int, default=0, help='whether to take the fixed (untrained) asym network weights as input')
    parser.add_argument('--max_size', type=int, default=10000, help='max dataset size')
    parser.add_argument('--target', type=str, default='test_acc', help='regression target')
    args = parser.parse_args()
    if args.dataset == 'resnet8':
        args.use_fixed = 1
    main(args)
