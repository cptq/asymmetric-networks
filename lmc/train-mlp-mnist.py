# -*- coding: utf-8 -*-
import math
import itertools
import copy
import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
import random
import numpy as np
from models.models_mlp import SigmaMLP, WMLP, MLP
from LMC_utils import *
import argparse
def train(model, optimizer, batch_size=32, n_epochs=10, lr_schedule = None):
    loss_fn = nn.CrossEntropyLoss()
    train_loss_history = np.zeros([n_epochs, 1])
    valid_accuracy_history = np.zeros([n_epochs, 1])
    valid_loss_history = np.zeros([n_epochs, 1])
    for epoch in range(n_epochs):

        # Train code from CS189
        model.train()

        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        if lr_schedule:
            lr_schedule.step()
        train_loss_history[epoch] = train_loss / len(train_loader.dataset)

        # Track loss each epoch
        print('Train Epoch: %d  Average loss: %.4f' %
              (epoch + 1,  train_loss_history[epoch]))

        model.eval()

        valid_loss = 0
        correct = 0
        valid_loss_fn = nn.CrossEntropyLoss(reduction = 'sum')
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                valid_loss += valid_loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max class score
                correct += pred.eq(target.view_as(pred)).sum().item()

        valid_loss_history[epoch] = valid_loss / len(valid_loader.dataset)

        
        valid_accuracy_history[epoch] = correct / len(valid_loader.dataset)

        print('Valid set: Average loss: %.4f, Accuracy: %d/%d (%.4f)\n' %
              (valid_loss_history[epoch], correct, len(valid_loader.dataset),
              100. * valid_accuracy_history[epoch]))

    return model


def test(model, batch_size=32):

    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction = 'sum')
 
    test_loss = 0
    correct = 0
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max class score
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    
    print('Test set: Average loss: %.4f, Accuracy: %d/%d (%.4f)' %
          (test_loss, correct, len(test_loader.dataset),
          100. * test_accuracy))
    return test_loss, test_accuracy

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
if __name__ == '__main__':
    from torchvision import datasets, transforms
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('-w', '--weight_decay', default=.03, type=float,
                        metavar='W', help='weight decay (default: 0e-4')
    parser.add_argument('-l', '--lr', default=1e-3, type=float,
                        metavar='W', help='lr (default: 1e-3')
    
    parser.add_argument('--lin_n_1', default = 64 , type=int,
                        metavar='LN1', help='lin c 1 (default: 64)')
    parser.add_argument('--lin_c_1', default=1, type=int,
                        metavar='L', help='lin c 1 (default: 1)')
    
    parser.add_argument('--lin_n_2', default=64, type=int,
                        metavar='LC', help='lin n 2 (default: 64)')
    parser.add_argument('--lin_c_2', default=1/2, type=int,
                        metavar='LC', help='lin c 2 (default: 1/2)')
    parser.add_argument('--lin_n_3', default=256, type=int,
                        metavar='LC', help='lin n 3 (default: 256)')
    parser.add_argument('--lin_c_3', default=1/4, type=int,
                        metavar='LC', help='lin c 3 (default: 1/4)')
    parser.add_argument('--lin_n_0', default=64, type=int,
                        metavar='LC', help='lin n 0 (default: 64)')
    parser.add_argument('--lin_c_0', default=1, type=int,
                        metavar='LC', help='lin c 0 (default: 1)')

    parser.add_argument('--symmetry',  default=0, type=int,
                        metavar='s', help='Symmetry: 0 (Standard) 1 (W) 2 (Sigma)')
    args = parser.parse_args()
    normalize = transforms.Normalize((0.1307,), (0.3081,))

   
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]), download=False)
    #USING CS189 CODE
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset,
        [int(len(train_dataset)*0.9), int(len(train_dataset)*0.1)],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory = False
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        pin_memory = False,
        batch_size=128, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        pin_memory = False,
        batch_size=128, shuffle=False
    )

    device = torch.device('cuda')
    in_dim, out_dim = 784,10
    linear_mask_params_0 = {'mask_constant' : args.lin_c_0, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': args.lin_n_0}
    linear_mask_params_1 = {'mask_constant' : args.lin_c_1, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': args.lin_n_1}
    linear_mask_params_2 = {'mask_constant' : args.lin_c_2, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': args.lin_n_2}
    linear_mask_params_3 = {'mask_constant' : args.lin_c_3, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': args.lin_n_3}

    
    mask_params = {
        0 : linear_mask_params_0, 
        1 : linear_mask_params_1, 
        2: linear_mask_params_2,
        3: linear_mask_params_3
    }

    HIDDEN_DIM = 512
    NUM_LAYERS = 4
    if args.symmetry == 0:
        model1 = MLP(in_dim, HIDDEN_DIM, out_dim, NUM_LAYERS, norm='layer').to(device)
        model2 = MLP(in_dim, HIDDEN_DIM, out_dim, NUM_LAYERS, norm='layer').to(device)
    elif args.symmetry == 1:
        model1 = WMLP(in_dim, HIDDEN_DIM, out_dim, NUM_LAYERS, mask_params, norm='layer').to(device)
        model2 = WMLP(in_dim, HIDDEN_DIM, out_dim, NUM_LAYERS, mask_params, norm='layer').to(device)
    else:
        model1 = SigmaMLP(in_dim, HIDDEN_DIM, out_dim, NUM_LAYERS, norm='layer').to(device)
        model2 = SigmaMLP(in_dim, HIDDEN_DIM, out_dim, NUM_LAYERS, norm='layer').to(device)

    naive_num_params = sum(p.numel() for p in model1.parameters())
    num_params = naive_num_params - model1.count_unused_params()
    print('Naive param count:', naive_num_params)
    print('Actual param count:', num_params)

    epochs = args.epochs
        
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=args.lr, weight_decay = args.weight_decay)     


    
    wandb.init(
        # set the wandb project where this run will be logged
        project="PROJECT-NAME",
        config=args
        # track hyperparameters and run metadata
    )
    train(model1, optimizer1, n_epochs = epochs, lr_schedule = None, batch_size=args.batch_size)
    test_loss, test_acc = test(model1)
    wandb.run.summary["test_acc_1"] = test_acc
    wandb.run.summary["test_loss_1"] = test_loss

    train(model2, optimizer2, n_epochs = epochs, lr_schedule = None, batch_size=args.batch_size)
    test_loss, test_acc = test(model2)
    wandb.run.summary["test_acc_2"] = test_acc
    wandb.run.summary["test_loss_2"] = test_loss

    outs = interpolate_test(model1, model2, test)
    wandb.run.summary["worst_loss"] = max(out[0] for out in outs)
    wandb.run.summary["worst_acc"] = min(out[1] for out in outs)
    if args.symmetry > 0:
        for i, ((out_loss, out_acc)) in enumerate(outs):
            wandb.log({
                "interpolation_test_loss": out_loss,
                "interpolation_test_acc": out_acc,
            })
    else:
        device = torch.device('cpu')
        model1 = model1.to(device)
        model2 = model2.to(device)
        from rebasin.weight_matching import *
        ps = mlp_permutation_spec(NUM_LAYERS - 1, norm = True)
        sd1 = model1.state_dict()
        sd2 = model2.state_dict()
        perm = weight_matching(ps,  sd1, sd2)
        sd = apply_permutation(ps, perm, model2.state_dict())
        model2.load_state_dict(sd)
        device = torch.device('cuda')
        
        model1 = model1.to(device)
        model2 = model2.to(device)
        outs_align = (interpolate_test(model1, model2, test))
        for i, ((out_loss, out_acc), (out_loss_align, out_acc_align)) in enumerate(zip(outs, outs_align)):
            wandb.log({
                "interpolation_test_loss": out_loss,
                "interpolation_test_acc": out_acc,
                "interpolation_test_loss_align": out_loss_align,
                "interpolation_test_acc_align": out_acc_align,
                
    
            })
    wandb.finish()


