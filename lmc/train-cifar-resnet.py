
import math
import itertools
import copy



import torch
import torch.nn as nn


import torch.nn.functional as F
import random
from torch.nn import init
import wandb
from LMC_utils import *
from models.models_resnet import *

from torchvision import datasets, transforms
transform_train = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])#https://stackoverflow.com/questions/50710493/cifar-10-meaningless-normalization-values

    ])

train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                               transform=transform_train)
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset,
                                                             [int(len(train_dataset)*0.9), int(len(train_dataset)*0.1)],
                                                             generator=torch.Generator().manual_seed(42))
test_dataset = datasets.CIFAR10('./data', train=False, download=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                                ]))
import numpy as np

transform_img = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
    ])
def train(model, optimizer, batch_size=32, n_epochs=10, lr_schedule = None):

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    valid_loss_fn = nn.CrossEntropyLoss(reduction = 'sum')
    train_loss_history = np.zeros([n_epochs, 1])
    valid_accuracy_history = np.zeros([n_epochs, 1])
    valid_loss_history = np.zeros([n_epochs, 1])
    for epoch in range(n_epochs):

        # Train code from CS189
        model.train()

        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            data = transform_img(data)
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

def test(model, batch_size=128):

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
from itertools import permutations


import argparse
parser = argparse.ArgumentParser(description='Properties for ResNets for CIFAR10 in pytorch')
if __name__ == '__main__':
    from torchvision import datasets, transforms
    parser.add_argument('--symmetry', default=0, type=int, metavar='N',
                        help='sparse?')
    parser.add_argument('--width', default=1, type=int, metavar='N',
                    help='width multiplying factor')
    parser.add_argument('--n_mul', default=1, type=float, metavar='N',
                    help='number of fixed entries multiplying factor')
    parser.add_argument('--c_mul', default=1, type=float, metavar='N',
                    help='std of fixed entries multiplying factor')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of epochs')
    parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                    help='batch_size')
    args = parser.parse_args()
    
    
    device = torch.device('cuda')
    in_dim, out_dim = 3072,10
    wandb.init(
        # set the wandb project where this run will be logged
        project="PROJECT-NAME",
        config = args
        # track hyperparameters and run metadata
    )
    w = args.width
    linear_mask_params = {'mask_constant' : 3*args.c_mul, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': int(8*args.n_mul)}
    conv_mask_params_f = {'mask_constant' : 3*args.c_mul, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': int(12*args.n_mul)}
    
    conv_mask_params_1_conv = {'mask_constant' :3*args.c_mul, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': int(36*args.n_mul)}
    conv_mask_params_1_skip = {'mask_constant' : 3*args.c_mul, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': int(4*args.n_mul)}
    conv_mask_params_1 = {'conv' : conv_mask_params_1_conv, 'skip':conv_mask_params_1_skip}
    
    conv_mask_params_2_conv = {'mask_constant' : 3*args.c_mul, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed':int(54*args.n_mul)}
    conv_mask_params_2_skip = {'mask_constant' : 3*args.c_mul, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': int(6*args.n_mul)}
    conv_mask_params_2 = {'conv' : conv_mask_params_2_conv, 'skip':conv_mask_params_2_skip}
    
    conv_mask_params_3_conv = {'mask_constant' : 3*args.c_mul, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed':int(72*args.n_mul)}
    conv_mask_params_3_skip = {'mask_constant' : 3*args.c_mul, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': int(8*args.n_mul)}
    conv_mask_params_3 = {'conv' : conv_mask_params_3_conv, 'skip':conv_mask_params_3_skip}
    mask_params = {
           'linear' : linear_mask_params, 
           'conv_f' : conv_mask_params_f, 
           'conv_1': conv_mask_params_1,
           'conv_2': conv_mask_params_2,
           'conv_3': conv_mask_params_3
          }
    
    
    if args.symmetry == 0:
        model1 = resnet20(w = w).to(device)
        model2 = resnet20(w = w).to(device)
    elif args.symmetry == 1:
        model1 = w_resnet20(mask_params, w = w).to(device)
        model2 = w_resnet20(mask_params, w = w).to(device)
    elif args.symmetry == 2:
        model1 = sigma_resnet20(mask_params, w = w).to(device)
        model2 = sigma_resnet20(mask_params, w = w).to(device)
    
    else:
        raise ValueError('Invalid model type')
    
    naive_num_params = sum(p.numel() for p in model1.parameters())
    num_params = naive_num_params - model1.count_unused_params()
    print('Naive param count:', naive_num_params)
    print('Actual param count:', num_params)
    epochs = args.epochs
    # if args.warm_up:
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-4, weight_decay = 0.00)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4, weight_decay = 0.00)     
    lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[i for i in range(1,21)],  gamma = 1.259)
    lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[i for i in range(1,21)],  gamma = 1.259)

    train(model1, optimizer1, n_epochs = epochs, lr_schedule = lr_scheduler1, batch_size=args.batch_size)
    test_loss, test_acc = test(model1)
    wandb.run.summary["test_acc_1"] = test_acc
    wandb.run.summary["test_loss_1"] = test_loss

    train(model2, optimizer2, n_epochs = epochs, lr_schedule = lr_scheduler2, batch_size=args.batch_size)
    test_loss, test_acc = test(model2)
    wandb.run.summary["test_acc_2"] = test_acc
    wandb.run.summary["test_loss_2"] = test_loss

    outs = interpolate_test(model1, model2, test)
    wandb.run.summary["worst_test_loss"] = max(out[0] for out in outs)
    wandb.run.summary["worst_test_acc"] = min(out[1] for out in outs)
    if args.symmetry == 0:
        from rebasin.weight_matching import *
        d = torch.device('cpu')
    
        model1 = model1.to(d)
        model2 = model2.to(d)
        
        sd1 = model1.state_dict()
        sd2 = model2.state_dict()
        ps = resnet20_permutation_spec()
        
        perm = weight_matching(ps,  sd1, sd2)
        sd = apply_permutation(ps, perm, sd2)
        model2.load_state_dict(sd)
        device = torch.device('cuda')
        model1 = model1.to(device)
        model2 = model2.to(device)
        
        outs_align = interpolate_test(model1, model2, test)
        wandb.run.summary["worst_test_loss_align"] = max(out[0] for out in outs_align)
        wandb.run.summary["worst_test_acc_align"] = min(out[1] for out in outs_align)
        
    wandb.finish()
