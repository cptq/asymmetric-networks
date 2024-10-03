import logging
from types import SimpleNamespace
from pathlib import Path
import time
import argparse
from argparse import Namespace
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
import argparse
from sparse import *
from alg import elrg
from fit import fit, DL, get_pred
from torchmetrics.classification import MulticlassCalibrationError
import wandb
logging.basicConfig(
    format="[%(asctime)s, %(levelname)s] %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

torch.backends.cudnn.benchmark = True

def fully_connected(in_dim, out_dim, num_layers, size):
    
    layers = [nn.Flatten(), nn.Linear(in_dim, size), nn.LayerNorm(size), nn.ReLU(inplace=True)]
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(size, size),nn.LayerNorm(size), nn.ReLU(inplace=True)])
    layers.append(nn.Linear(size, out_dim))
    return nn.Sequential(*layers)

def sparse_fully_connected(in_dim, out_dim, num_layers, size, n = 8, c=.5):
    print(f"n={n}, c={c}")
    layers = [nn.Flatten(), SparseLinear(in_dim, size, mask_num = 0, mask_constant = c, num_fixed = n), nn.LayerNorm(size), nn.ReLU(inplace=True)]
    for i in range(num_layers - 1):
        layers.extend([SparseLinear(size, size, mask_num = i+1, mask_constant = c, num_fixed = n),nn.LayerNorm(size), nn.ReLU(inplace=True)])
    layers.append(SparseLinear(size, out_dim, mask_num = num_layers, mask_constant = c, num_fixed = n))
    return nn.Sequential(*layers)

def get_transform(pad, crop, stats, flip):
    tfm = [
        T.Pad(pad, padding_mode="reflect"),
        T.RandomCrop(crop),
    ]

    multiplier = 4

    if flip:
        tfm += [T.RandomHorizontalFlip(0.5)]
        multiplier *= 2

    base = [T.ToTensor(), T.Normalize(*stats)]

    return (
        T.Compose(base + tfm),
        T.Compose(base),
        multiplier
    )


IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CIFAR_STATS = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])
MNIST_STATS = ([0.1307], [0.3081])

MNISTS = ['MNIST', 'KMNIST', 'FashionMNIST', 'EMNIST']

TRANSFORMS = \
    {"CIFAR10":
         get_transform(4, 32, CIFAR_STATS, flip=True),
     "CIFAR100":
         get_transform(4, 32, CIFAR_STATS, flip=True),
     "SVHN":
         get_transform(4, 32, IMAGENET_STATS, flip=False),
     "STL10":
         get_transform(12, 96, IMAGENET_STATS, flip=True),

     }

TRANSFORMS.update({name: get_transform(4, 28, MNIST_STATS, flip=False)
                   for name in MNISTS})

NUM_CLASS = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "SVHN": 10,
    "STL10": 10,
    "MNIST": 10,
    "FashionMNIST": 10,
    "KMNIST": 10,
}


def get_stats(ds):
    if ds in MNISTS:
        return MNIST_STATS
    elif ds in ['CIFAR10', 'CIFAR100']:
        return CIFAR_STATS
    else:
            return IMAGENET_STATS



def get_dl(local_storage_path,
           dataset,
           data_transform,
           transforms,
           train,
           batch_size,
           device):
    
    
    def get_ds_kwargs(dataset, train):
        if dataset in ['SVHN', 'STL10']:
            kwargs = {'split': 'train' if train else 'test'}
        elif dataset == 'EMNIST':
            kwargs = {'split': 'letters', 'train': train}
        else:
            kwargs = {'train': train}
        return kwargs
    
    kwargs = get_ds_kwargs(dataset, train)

    ds = getattr(torchvision.datasets, dataset)(
        local_storage_path,
        download=True,
        transform=transforms,
        **kwargs)


    if not data_transform:

        if dataset in ['CIFAR10', 'CIFAR100']:
            return DL(torch.from_numpy(ds.data).float().transpose(1, 3),
                      torch.tensor(ds.targets),
                      batch_size=batch_size,
                      device=device)

        elif dataset in ['SVHN', 'STL10']:
            return DL(torch.from_numpy(ds.data).float(),
                      torch.tensor(ds.labels),
                      batch_size=batch_size)

        elif dataset in ['KMNIST', 'EMNIST', 'FashionMNIST', 'MNIST']:
            return DL(ds.train_data[:, None].float(),
                      ds.targets,
                      batch_size=batch_size,
                      device=device)

        else:
            raise NotImplementedError()

    return torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=12 if train else 8,
        shuffle=train,
    )


t=[]
def train_model(config):

    num_class = NUM_CLASS[config.dataset]
    device = torch.device(config.device)
                         
    if config.model == 'fc':
        model = fully_connected(3072, num_class, 32, 256)
    elif config.model == 'sfc':
        model = sparse_fully_connected(3072, num_class, config.depth, 256, c = config.c, n=config.n)
    elif config.model == 'resnet':
        model = resnet56()
    elif config.model =='sparse_resnet':
        model = sparse_resnet56(mask_params)
        naive_num_params = sum(p.numel() for p in model.parameters())
        num_params = naive_num_params - model.count_unused_params()
        print('Naive param count:', naive_num_params)
        print('Actual param count:', num_params)
    else:
        model = getattr(torchvision.models,
                        config.model)(num_classes=num_class)

    model = nn.Sequential(model,
                          nn.LogSoftmax(dim=-1))

    train_tfm, valid_tfm, multiplier = TRANSFORMS[config.dataset]

    data = SimpleNamespace(
        train_dl=get_dl(local_storage_path="/tmp",
                        dataset=config.dataset,
                        data_transform=config.data_transform,
                        train=True,
                        batch_size=config.batch_size,
                        device=device,
                        transforms=train_tfm),
        valid_dl=get_dl(local_storage_path="/tmp",
                        dataset=config.dataset,
                        data_transform=config.data_transform,
                        train=False,
                        batch_size=config.batch_size // 10,
                        device=device,
                        transforms=valid_tfm)
    )

    model.to(device)
    num_data = len(data.train_dl.dataset) * \
               (multiplier if config.data_transform else 1)
    model = elrg(model, config, num_data)
    # return model
    results = fit(
        model=model,
        data=data,
        loss_func=F.nll_loss,
        num_updates=config.num_updates,
        keep_curve=False,
        device=device,
        log_steps=100,
        num_burnin_steps = 10000 * config.burn_in
    )
    wandb.run.summary["test loss"] = results.loss
    wandb.run.summary["test acc"] = results.er
    wandb.run.summary["ECE"] = results.ece  
    
    

    logging.info(
        f"Final test nll {results.loss:.3f} "
        f"error rate {results.er:.2f}"
        f"ECE {results.ece:.2f}"

    )
    t.append(
        f"Final test nll {results.loss:.3f} "
        f"error rate {results.er:.2f}"
    
    )
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--sparse",
                    default=0,
                    type=int)
parser.add_argument("--n",
                    default=0,
                    type=int)
parser.add_argument("--num_updates",
                    default=10000,
                    type=int)
parser.add_argument("--rank",
                    default=4,
                    type=int)
parser.add_argument("--c",
                    default=.5,
                    type=float)
parser.add_argument("--depth",
                    default=8,
                    type=int)
args = parser.parse_args()

n=args.n * args.sparse
c = args.c
wandb.init(project='PROJECT-NAME-MLP', config = args)
linear_mask_params = {'mask_constant' : c, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': 10*n}
conv_mask_params_f = {'mask_constant' : c, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': 12*n}

conv_mask_params_1_conv = {'mask_constant' :c, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': 8*n}
conv_mask_params_1_skip = {'mask_constant' :c, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': 5*n}
conv_mask_params_1 = {'conv' : conv_mask_params_1_conv, 'skip':conv_mask_params_1_skip}

conv_mask_params_2_conv = {'mask_constant' : c, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed':12*n}
conv_mask_params_2_skip = {'mask_constant' : c, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': 6*n}
conv_mask_params_2 = {'conv' : conv_mask_params_2_conv, 'skip':conv_mask_params_2_skip}

conv_mask_params_3_conv = {'mask_constant' : c, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': 16*n}
conv_mask_params_3_skip = {'mask_constant' : c, 'mask_type' : 'random_subsets', 'do_normal_mask' : True, 'num_fixed': 8*n}
conv_mask_params_3 = {'conv' : conv_mask_params_3_conv, 'skip':conv_mask_params_3_skip}
mask_params = {
       'linear' : linear_mask_params, 
       'conv_f' : conv_mask_params_f, 
       'conv_1': conv_mask_params_1,
       'conv_2': conv_mask_params_2,
       'conv_3': conv_mask_params_3
      }
args = Namespace(dataset='CIFAR10', model='sfc', n=n, c=c, depth=args.depth, rank=args.rank, learn_diag=False, num_updates=args.num_updates, burn_in = 0, data_transform=True, batch_size=250, lr=0.0005, device='cuda', num_test_samples=25, test_batch_size=512, scale_prior=True, q_init_logvar=-10.0, prior_precision=.01)
logging.info(f"Config: {args}")
model=train_model(config=args)

wandb.finish()