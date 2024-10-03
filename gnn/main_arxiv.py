import argparse

import math
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SimpleConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from interp import interpolate_test
from asym_nets import SparseLinear, AsymNonlin, AsymSwiGLU

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) >= 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

class MyGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, nonlin='gelu', lin_builder=None):
        ''' lin_builder is a function taking in a in_dim and out_dim,
            and returning an nn.Module
        '''
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(MyConv(in_channels, hidden_channels, nonlin=nonlin, C=C_lst[0], lin_builder=lin_builder))
        for i in range(num_layers - 1):
            self.convs.append(
                MyConv(hidden_channels, hidden_channels, nonlin=nonlin, C=C_lst[i+1], lin_builder=lin_builder))
        self.lin = lin_builder(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x = self.lin(x)
        return x.log_softmax(dim=-1)
    
    def count_unused_params(self):
        count = 0
        for node in self.modules():
            if hasattr(node, 'mask'):
                count += (1-node.mask).sum().int().item()
        return count

class MyConv(nn.Module):
    def __init__(self, in_dim, out_dim, nonlin='gelu', C=None, lin_builder=None):
        super().__init__()
        if nonlin == 'gelu':
            nonlin_module = nn.GELU()
        elif nonlin == 'asym_gelu':
            nonlin_module = AsymNonlin(C)
        elif nonlin == 'asym_swiglu':
            nonlin_module = AsymSwiGLU(C)
        self.conv = SimpleConv(aggr='mean', combine_root='sum')
        #self.mlp = nn.Sequential(lin_builder(in_dim, out_dim), nn.LayerNorm(out_dim), nonlin_module)
        self.mlp = nn.Sequential(lin_builder(in_dim, out_dim), nn.BatchNorm1d(out_dim), nonlin_module)
    
    def reset_parameters(self):
        self.mlp[0].reset_parameters()
        self.mlp[1].reset_parameters()

    def forward(self, x, adj_t):
        x = self.conv(x, adj_t)
        x = self.mlp(x)
        return x


def train(model, data, train_idx, optimizer, scheduler):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    out = model(data.x, data.adj_t)
    train_loss = F.nll_loss(out[train_idx], data.y.squeeze(1)[train_idx]).item()
    valid_loss = F.nll_loss(out[valid_idx], data.y.squeeze(1)[valid_idx]).item()
    test_loss = F.nll_loss(out[test_idx], data.y.squeeze(1)[test_idx]).item()
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss

def perm_gnn(model1, model2, data):
    activations = {}
    model1.eval(), model2.eval()
    def activation_hook(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    num_convs = len(model1.convs)
    hooks1 = [model1.convs[i].mlp[0].register_forward_hook(activation_hook(f'{i}_1')) for i in range(num_convs)]
    hooks2 = [model2.convs[i].mlp[0].register_forward_hook(activation_hook(f'{i}_2')) for i in range(num_convs)]

    out1 = model1(data.x, data.adj_t)
    out2 = model2(data.x, data.adj_t)
    [hook.remove() for hook in hooks1]
    [hook.remove() for hook in hooks2]
    perms = []
    for i in range(num_convs):
        sim = activations[f'{i}_1'].T @ activations[f'{i}_2']
        _, perm = linear_sum_assignment(-sim.cpu().numpy())
        perms.append(perm)
    def rowperm_mat(perm, mat):
        temp_mat = mat.clone()
        temp_mat[perm] = mat
        return temp_mat
    def colperm_mat(perm, mat):
        temp_mat = mat.clone()
        temp_mat[:, perm] = mat
        return temp_mat

    for i in range(num_convs):
        perm = perms[i]
        model1.convs[i].mlp[0].weight.data = rowperm_mat(perm, model1.convs[i].mlp[0].weight.data)
        model1.convs[i].mlp[0].bias.data = rowperm_mat(perm, model1.convs[i].mlp[0].bias.data)
        model1.convs[i].mlp[1].weight.data = rowperm_mat(perm, model1.convs[i].mlp[1].weight.data)
        model1.convs[i].mlp[1].bias.data = rowperm_mat(perm, model1.convs[i].mlp[1].bias.data)
        if i != num_convs-1:
            model1.convs[i+1].mlp[0].weight.data = colperm_mat(perm, model1.convs[i+1].mlp[0].weight.data)
        else:
            model1.lin.weight.data = colperm_mat(perm, model1.lin.weight.data)
    print("POST PERM DIST:", (out1 - model1(data.x, data.adj_t)).norm())
    return model1
    

def main(args, device, dataset, data, split_idx):
    
    train_idx = split_idx['train'].to(device)

    if args.model == 'gnn':
        lin_builder = nn.Linear
        model = MyGNN(data.num_features, args.hidden_channels,
                  dataset.num_classes, args.num_layers, args.dropout, lin_builder=lin_builder)
    elif args.model == 'asym_gelu_gnn':
        lin_builder = nn.Linear
        model = MyGNN(data.num_features, args.hidden_channels,
                  dataset.num_classes, args.num_layers, args.dropout, nonlin='asym_gelu', lin_builder=lin_builder)
    elif args.model == 'asym_swiglu_gnn':
        lin_builder = nn.Linear
        model = MyGNN(data.num_features, args.hidden_channels,
                  dataset.num_classes, args.num_layers, args.dropout, nonlin='asym_swiglu', lin_builder=lin_builder)
    elif args.model == 'asym_w_gnn':
        lin_builder = lambda in_dim, out_dim: SparseLinear(in_dim, out_dim, mask_constant=0.5, num_fixed=6, do_normal_mask=True, mask_type='random_subsets')
        model = MyGNN(data.num_features, args.hidden_channels,
                  dataset.num_classes, args.num_layers, args.dropout, nonlin='asym_swiglu', lin_builder=lin_builder)
    else:
        raise ValueError('Invalid model type')
    model = model.to(device)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    if hasattr(model, 'count_unused_params'):
        num_params -= model.count_unused_params()
    print("Num params:", num_params)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    def linear_warmup(step):
        warmup_steps = args.warmup_steps
        if step < warmup_steps:
            return float(step+1) / float(warmup_steps)
        else:
            return 1.0

    for run in range(args.runs):
        model.reset_parameters()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = LambdaLR(optimizer, linear_warmup)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer, scheduler)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss = result
                print(
                      f'Ep: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Val: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%',
                      f'Val loss: {valid_loss:.3f} '
                      f'Test loss: {test_loss:.3f}',
                        )

        logger.print_statistics(run)
    logger.print_statistics()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--model', type=str, default='gnn')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--interp', type=int, default=0)
    parser.add_argument('--rebasin', type=int, default=0)
    parser.add_argument('--warmup_steps', type=int, default=25)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', 
                            transform=T.Compose([T.ToUndirected(), T.ToSparseTensor()]) )

    data = dataset[0]
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-arxiv')
    
    C_lst = [.01*torch.randn(args.hidden_channels, args.hidden_channels) / math.sqrt(args.hidden_channels) for _ in range(args.num_layers)]

    model1 = main(args, device, dataset, data, split_idx)
    if args.interp:
        model2 = main(args, device, dataset, data, split_idx)
        if args.rebasin:
            model1 = perm_gnn(model1, model2, data)
        test_fn = lambda model: test(model, data, split_idx, evaluator)[5]
        print('Interpolating... ')
        results = interpolate_test(model1, model2, test_fn, steps=3, rewarm=True, data=data)
        print('results:', results)
        n = len(results)
        barrier = results[n//2] - .5*(results[0] + results[-1])
        print('Barrier:', barrier)

